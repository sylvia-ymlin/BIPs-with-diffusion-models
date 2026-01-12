
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import data, transform
import time
import sys
sys.path.append(os.getcwd())

from src.solver import DiffusionSolver
from src.operators.operators import SuperResolutionOperator, MRIOperator

def get_fast_images(device, size=256):
    images = []
    # 1. Astronaut (SuperRes)
    img_astro = data.astronaut()
    img_astro = transform.resize(img_astro, (size, size))
    t_astro = torch.from_numpy(img_astro).permute(2, 0, 1).float().unsqueeze(0)
    t_astro = (t_astro * 2) - 1
    images.append(('Astronaut', t_astro.to(device)))
    
    # 2. Brain MRI (Medical)
    # skimage.data.brain() returns a 4D tensor usually [10, 256, 256] (slices)
    try:
        vol = data.brain()
        # Take middle slice (e.g. index 5), channel 0 if 4D
        # shape might be (10, 256, 256)
        img_brain = vol[5] 
    except:
        # Fallback if brain not available (e.g. download fail)
        print("Warning: Brain download failed, using Shepp-Logan")
        img_brain = data.shepp_logan_phantom()
        img_brain = transform.resize(img_brain, (256, 256))

    img_brain = transform.resize(img_brain, (size, size))
    # It is grayscale [H, W]. Need [1, 1, H, W] -> broadcast to [1, 3, H, W] for DDPM?
    # DDPM expects 3 channels.
    t_brain = torch.from_numpy(img_brain).float().unsqueeze(0).unsqueeze(0)
    t_brain = t_brain.repeat(1, 3, 1, 1) # [1, 3, 256, 256]
    t_brain = (t_brain * 2) - 1
    images.append(('Brain', t_brain.to(device)))
    
    return images

def visualize(task, name, x_gt, y_meas, result, save_path):
    x_smc = result['smc']['mean']
    x_tv = result['tv']['mean']
    
    def to_img(t): 
        if t.ndim == 4: t = t.squeeze(0)
        img = (t.detach().cpu().numpy() + 1) / 2
        img = np.clip(img, 0, 1)
        return img.transpose(1, 2, 0)

    # Plotting Logic
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    
    # Helper: Handle Grayscale for MRI
    def process_for_plot(tensor, is_mri=False):
        img = to_img(tensor)
        if is_mri:
            # Convert RGB (from model) to Grayscale for display
            if img.shape[-1] == 3:
                img = img.mean(axis=2)
        return img

    is_mri = (name == 'Brain')
    cmap = 'gray' if is_mri else None
    
    axs[0].imshow(process_for_plot(x_gt, is_mri), cmap=cmap); axs[0].set_title("Original", fontsize=14, pad=10)
    axs[0].axis('off')
    
    axs[1].imshow(process_for_plot(x_tv, is_mri), cmap=cmap); axs[1].set_title("TV (Baseline)", fontsize=14, pad=10)
    axs[1].axis('off')
    
    axs[2].imshow(process_for_plot(x_smc, is_mri), cmap=cmap); axs[2].set_title("Twisted SMC (Ours)", fontsize=14, pad=10)
    axs[2].axis('off')
    
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running FAST DEMO on {DEVICE}")
    
    model_id = "google/ddpm-celebahq-256"
    
    TIMESTEPS = 100 # Increased to 100 for better convergence (less noise)
    # FAST CONFIG: T=100 steps (approximate)
    config_smc = {'T': TIMESTEPS, 'method': 'smc', 'batch_size': 1, 'step_size': 0.1}
    
    print(f"Loading Model {model_id}...")
    solver_smc = DiffusionSolver(model_id, DEVICE, config_smc)
    
    images = get_fast_images(DEVICE)
    
    tasks = [
        ('SuperRes', SuperResolutionOperator(4, DEVICE)),
        ('MRI', MRIOperator(acceleration=4.0, center_fraction=0.1, device=DEVICE))
    ]
    
    for task_name, operator in tasks:
        # Match image to task
        target_img_name = 'Astronaut' if task_name == 'SuperRes' else 'Brain'
        img_name, x_gt = next(img for img in images if img[0] == target_img_name)
        
        print(f"\nTask: {task_name} on {img_name}")
        y = operator.forward(x_gt)
        
        # TV
        print("  Running TV...")
        from src.baselines import TVReconstruction
        tv_solver = TVReconstruction(DEVICE, lambda_tv=0.05, num_steps=50)
        x_tv = tv_solver.solve(y, operator, x_gt.shape)
        
        # SMC
        print("  Running SMC (Fast T=50)...")
        # Reuse TV result to mock DPS/SMC if need be? No, run it.
        # But wait, solver.py code iterates T.
        # It creates a linear schedule T=50.
        # This will work.
        x_smc, _ = solver_smc.sample(y, operator, (1, *x_gt.shape[1:]))
        
        save_path = f"results/real_{task_name}_{img_name}.png"
        visualize(task_name, img_name, x_gt, y, {'tv': {'mean': x_tv}, 'smc': {'mean': x_smc}}, save_path)
        print(f"  Saved {save_path}")

if __name__ == "__main__":
    main()
