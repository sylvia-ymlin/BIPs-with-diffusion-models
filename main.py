
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
from src.solver import DiffusionSolver
from src.operators import InpaintingOperator, SuperResolutionOperator

# --- Configuration ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "ddpm_mnist.pth"
BATCH_SIZE = 30 # Number of Particles

def run_task(task_name, operator, x_gt, solver_dps, solver_smc):
    print(f"\nRunning Task: {task_name}")
    
    # Create Measurement
    y = operator.forward(x_gt.unsqueeze(0))
    
    # Solve DPS
    print("  -> Running DPS...")
    rec_dps = solver_dps.sample(y, operator, (BATCH_SIZE, 1, 28, 28))
    mean_dps = rec_dps.mean(dim=0)
    var_dps = rec_dps.var(dim=0)
    
    # Solve SMC
    print("  -> Running SMC...")
    rec_smc = solver_smc.sample(y, operator, (BATCH_SIZE, 1, 28, 28))
    mean_smc = rec_smc.mean(dim=0)
    var_smc = rec_smc.var(dim=0)
    
    # Metrics
    mse_dps = F.mse_loss(mean_dps.unsqueeze(0), x_gt.unsqueeze(0)).item()
    mse_smc = F.mse_loss(mean_smc.unsqueeze(0), x_gt.unsqueeze(0)).item()
    
    print(f"  [{task_name}] DPS MSE: {mse_dps:.5f} | Var: {var_dps.mean().item():.5f}")
    print(f"  [{task_name}] SMC MSE: {mse_smc:.5f} | Var: {var_smc.mean().item():.5f}")
    
    return x_gt, y, mean_dps, mean_smc, var_dps, var_smc

def main():
    print("=== Industrial Grade Diffusion Solver Experiment ===")
    
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    x_gt, _ = dataset[0] # Digit '5'
    x_gt = x_gt.to(DEVICE)

    # Solvers
    config_dps = {'T': 300, 'method': 'dps', 'scale': 0.5, 'batch_size': BATCH_SIZE}
    config_smc = {'T': 300, 'method': 'smc', 'batch_size': BATCH_SIZE}
    
    solver_dps = DiffusionSolver(MODEL_PATH, DEVICE, config_dps)
    solver_smc = DiffusionSolver(MODEL_PATH, DEVICE, config_smc)
    
    # Task 1: Inpainting (Representative Task 1)
    mask = torch.ones_like(x_gt)
    mask[:, 10:18, 10:18] = 0
    op_inp = InpaintingOperator(mask, DEVICE)
    res_inp = run_task("Block Inpainting", op_inp, x_gt, solver_dps, solver_smc)
    
    # Task 2: Super-Resolution 4x (Representative Task 2)
    # MNIST 28x28 -> 7x7
    op_sr = SuperResolutionOperator(factor=4, device=DEVICE)
    res_sr = run_task("4x SuperRes", op_sr, x_gt, solver_dps, solver_smc)
    
    # Visualization
    visualize_results(res_inp, res_sr)

def visualize_results(res_inp, res_sr):
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    
    # Helper
    def show(ax, t, title=None, cmap='gray'):
        img = t.squeeze().cpu().detach().numpy()
        if cmap == 'gray': img = (img + 1)/2
        ax.imshow(img, cmap=cmap)
        if title: ax.set_title(title)
        ax.axis('off')

    # Task 1: Inpainting
    gt, y, dps_m, smc_m, dps_v, smc_v = res_inp
    show(axs[0,0], gt, "GT")
    show(axs[0,1], y, "Masked Input")
    show(axs[0,2], dps_m, "DPS Mean")
    show(axs[0,3], dps_v, "DPS Var", 'hot')
    
    show(axs[1,0], gt, "GT")
    show(axs[1,1], y, "Masked Input")
    show(axs[1,2], smc_m, "SMC Mean")
    show(axs[1,3], smc_v, "SMC Var", 'hot')
    
    # Task 2: SR
    gt, y, dps_m, smc_m, dps_v, smc_v = res_sr
    show(axs[2,0], gt, "GT")
    show(axs[2,1], y, "LowRes Input (4x)")
    show(axs[2,2], dps_m, "DPS Mean")
    show(axs[2,3], dps_v, "DPS Var", 'hot')
    
    show(axs[3,0], gt, "GT")
    show(axs[3,1], y, "LowRes Input (4x)")
    show(axs[3,2], smc_m, "SMC Mean")
    show(axs[3,3], smc_v, "SMC Var", 'hot')

    plt.tight_layout()
    if not os.path.exists("results"): os.makedirs("results")
    plt.savefig("results/industrial_benchmark.png")
    print("\nBenchmark Plot Saved: results/industrial_benchmark.png")

if __name__ == "__main__":
    main()
