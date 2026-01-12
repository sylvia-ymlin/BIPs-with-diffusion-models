
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
from skimage import data, transform
import time

# Add src to path
import sys
sys.path.append(os.getcwd())

from src.solver import DiffusionSolver
from src.operators.operators import SuperResolutionOperator, InpaintingOperator, MRIOperator, PhaseRetrievalOperator
from src.baselines import TVReconstruction

def get_real_world_images(device, size=256):
    """
    Get 3 representative images: Human (Astronaut), Cat (Chelsea), Texture (Coffee).
    """
    images = []
    
    # 1. Astronaut (Human Face proxy)
    img_astro = data.astronaut()
    img_astro = transform.resize(img_astro, (size, size))
    # to tensor [-1, 1], [B, C, H, W]
    t_astro = torch.from_numpy(img_astro).permute(2, 0, 1).float().unsqueeze(0)
    t_astro = (t_astro * 2) - 1
    images.append(('Astronaut', t_astro.to(device)))
    
    # 2. Cat (Chelsea)
    img_cat = data.chelsea()
    img_cat = transform.resize(img_cat, (size, size))
    t_cat = torch.from_numpy(img_cat).permute(2, 0, 1).float().unsqueeze(0)
    t_cat = (t_cat * 2) - 1
    images.append(('Cat', t_cat.to(device)))
    
    # 3. Coffee (Texture / Structure)
    img_coffee = data.coffee()
    img_coffee = transform.resize(img_coffee, (size, size))
    t_coffee = torch.from_numpy(img_coffee).permute(2, 0, 1).float().unsqueeze(0)
    t_coffee = (t_coffee * 2) - 1
    images.append(('Coffee', t_coffee.to(device)))
    
    return images

def visualize(task, name, x_gt, y_meas, result, save_path):
    # Unwrap
    x_dps = result['dps']['mean']
    x_smc = result['smc']['mean']
    var_smc = result['smc']['var']
    x_tv = result['tv']['mean']
    
    def to_img(t): 
        # t is [C, H, W] or [1, C, H, W]
        if t.ndim == 4: t = t.squeeze(0)
        img = (t.detach().cpu().numpy() + 1) / 2
        img = np.clip(img, 0, 1)
        return img.transpose(1, 2, 0)
        
    def to_var(t):
        if t.ndim == 4: t = t.squeeze(0)
        # Average variance across channels for visualization
        v = t.detach().cpu().numpy().mean(axis=0) 
        return v
    
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: GT, Input (if visualizable), TV, DPS
    axs[0, 0].imshow(to_img(x_gt)); axs[0, 0].set_title(f"{name} (GT)")
    axs[0, 0].axis('off')

    # Input (y_meas)
    # If y is image-like (SuperRes, Inpainting) show it. 
    # If k-space (MRI, PR), show magnitude log or something.
    if task in ['SuperRes', 'Inpainting']:
        axs[0, 1].imshow(to_img(y_meas)); axs[0, 1].set_title("Input (Degraded)")
    else:
        # Show Log Magnitude for freq domain
        axs[0, 1].text(0.5, 0.5, "Freq/K-space Data", ha='center')
    axs[0, 1].axis('off')
        
    axs[0, 2].imshow(to_img(x_tv)); axs[0, 2].set_title("Classical TV-L2")
    axs[0, 2].axis('off')
    
    axs[0, 3].imshow(to_img(x_dps)); axs[0, 3].set_title("DPS (Gradient)")
    axs[0, 3].axis('off')
    
    # Row 2: GT (Ref), SMC Mean, SMC Var, SMC Error
    axs[1, 0].imshow(to_img(x_gt)); axs[1, 0].set_title("GT (Ref)")
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(to_img(x_smc)); axs[1, 1].set_title("SMC Mean (Ours)")
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(to_var(var_smc), cmap='hot'); axs[1, 2].set_title("SMC Uncertainty")
    axs[1, 2].axis('off')
    
    # Error Map
    err = np.abs(to_img(x_smc) - to_img(x_gt)).mean(axis=2)
    axs[1, 3].imshow(err, cmap='inferno'); axs[1, 3].set_title("Error Map")
    axs[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running on {DEVICE}")
    
    # Load Model (HuggingFace)
    # Try CelebA-HQ 256. If fails, fallback? 
    # For astronaut/cat, ImageNet would be better, but we only have DDPM checkpoints easily available for specific domains.
    # actually google/ddpm-celebahq-256 generates faces.
    # google/ddpm-cifar10-32 is too small (32x32).
    # google/ddpm-church-256 could work for buildings.
    # Let's try CelebA-HQ 256. It might hallucinate faces on the cat/astronaut, 
    # WHICH IS AN INTERESTING RESULT (Prior Mismatch bias)!
    model_id = "google/ddpm-celebahq-256"
    
    config_dps = {'T': 1000, 'method': 'dps', 'scale': 0.5} # T must match model usually? Or we respace. HF pipeline handles T=1000.
    config_smc = {'T': 1000, 'method': 'smc', 'batch_size': 4, 'step_size': 0.1} # Lower batch for SMC due to memory
        
    print(f"Loading Model {model_id}...")
    try:
        solver_dps = DiffusionSolver(model_id, DEVICE, config_dps)
        solver_smc = DiffusionSolver(model_id, DEVICE, config_smc)
    except Exception as e:
        print(f"CRITICAL: Failed to load model. {e}")
        return

    # Data
    images = get_real_world_images(DEVICE, size=256)
    
    # Tasks
    tasks = [
        ('SuperRes', SuperResolutionOperator(4, DEVICE)),
        ('MRI', MRIOperator(acceleration=4.0, center_fraction=0.1, device=DEVICE))
        # Skipping Inpainting/PhaseRetrieval for brevity in this run
    ]
    
    results = {}
    
    for task_name, operator in tasks:
        print(f"\n=== Task: {task_name} ===")
        
        for img_name, x_gt in images:
            print(f"  Image: {img_name}")
            
            # Prepare Measurement
            y = operator.forward(x_gt)
            
            # 1. TV
            print("    Running TV...")
            tv_start = time.time()
            # Need slight adaptation for TV shape
            from src.baselines import TVReconstruction
            # Ensure TV works with single image batch=1
            tv_solver = TVReconstruction(DEVICE, lambda_tv=0.05, num_steps=100)
            x_tv = tv_solver.solve(y, operator, x_gt.shape) 
            x_tv_mean = x_tv # Det already
            
            # 2. DPS
            print("    Running DPS...")
            x_dps, _ = solver_dps.sample(y, operator, x_gt.shape)
            x_dps_mean = x_dps
            
            # 3. SMC
            print("    Running SMC...")
            x_smc, ess = solver_smc.sample(y, operator, (config_smc['batch_size'], *x_gt.shape[1:]))
            x_smc_mean = x_smc.mean(dim=0, keepdim=True)
            x_smc_var = x_smc.var(dim=0, keepdim=True)
            
            result_entry = {
                'tv': {'mean': x_tv_mean},
                'dps': {'mean': x_dps_mean},
                'smc': {'mean': x_smc_mean, 'var': x_smc_var, 'ess': ess}
            }
            
            save_path = f"results/real_{task_name}_{img_name}.png"
            visualize(task_name, f"{task_name} - {img_name}", x_gt, y, result_entry, save_path)
            print(f"    Saved to {save_path}")

if __name__ == "__main__":
    main()
