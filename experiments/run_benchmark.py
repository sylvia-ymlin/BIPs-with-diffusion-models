
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
import time

# Import Industrial Modules
from src.solver import DiffusionSolver
from src.operators.operators import InpaintingOperator, SuperResolutionOperator, PhaseRetrievalOperator, MRIOperator
from src.metrics.evaluator import calculate_psnr, calculate_ssim, calculate_variance_map
from src.baselines import TVReconstruction

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "ddpm_mnist.pth"
BATCH_SIZE = 20 # Lower batch size for safety

def run_benchmark_task(task_name, operator, x_gt, solver_dps, solver_smc):
    print(f"\n--- Running Benchmark: {task_name} ---")
    
    # 1. Create Measurement
    y = operator.forward(x_gt.unsqueeze(0))
    
    # 2. Run DPS
    print("  -> Execution: DPS Strategy")
    start_time = time.time()
    samples_dps, _ = solver_dps.sample(y, operator, (BATCH_SIZE, 1, 28, 28))
    time_dps = time.time() - start_time
    mean_dps = samples_dps.mean(dim=0)
    
    # 3. Run SMC
    print("  -> Execution: SMC Strategy")
    start_time = time.time()
    samples_smc, ess_log_smc = solver_smc.sample(y, operator, (BATCH_SIZE, 1, 28, 28))
    time_smc = time.time() - start_time
    mean_smc = samples_smc.mean(dim=0)
    
    # 4. Evaluation
    psnr_dps = calculate_psnr(mean_dps.unsqueeze(0), x_gt.unsqueeze(0))
    ssim_dps = calculate_ssim(mean_dps.unsqueeze(0), x_gt.unsqueeze(0))
    var_dps = calculate_variance_map(samples_dps)

    psnr_smc = calculate_psnr(mean_smc.unsqueeze(0), x_gt.unsqueeze(0))
    ssim_smc = calculate_ssim(mean_smc.unsqueeze(0), x_gt.unsqueeze(0))
    var_smc = calculate_variance_map(samples_smc)
    
    
    # 5. Run Classical Baseline (TV)
    print("  -> Execution: Classical TV Baseline")
    start_time = time.time()
    tv_solver = TVReconstruction(DEVICE, lambda_tv=0.1, num_steps=200)
    samples_tv = tv_solver.solve(y, operator, (BATCH_SIZE, 1, 28, 28))
    time_tv = time.time() - start_time
    mean_tv = samples_tv.mean(dim=0) # TV is deterministic, but solve returns batch
    
    psnr_tv = calculate_psnr(mean_tv.unsqueeze(0), x_gt.unsqueeze(0))
    ssim_tv = calculate_ssim(mean_tv.unsqueeze(0), x_gt.unsqueeze(0))
    
    print(f"  [Result] TV  -> PSNR: {psnr_tv:.2f} dB | SSIM: {ssim_tv:.4f} | Time: {time_tv:.2f}s")
    print(f"  [Result] DPS -> PSNR: {psnr_dps:.2f} dB | SSIM: {ssim_dps:.4f} | Var: {var_dps:.5f} | Time: {time_dps:.2f}s")
    print(f"  [Result] SMC -> PSNR: {psnr_smc:.2f} dB | SSIM: {ssim_smc:.4f} | Var: {var_smc:.5f} | Time: {time_smc:.2f}s")
    
    return {
        'y': y,
        'tv': {'mean': mean_tv, 'time': time_tv},
        'dps': {'mean': mean_dps, 'var': samples_dps.var(dim=0), 'time': time_dps},
        'smc': {'mean': mean_smc, 'var': samples_smc.var(dim=0), 'time': time_smc, 'ess': ess_log_smc}
    }

def visualize_task(task_name, result, x_gt):
    fig, axs = plt.subplots(2, 5, figsize=(16, 7))
    
    def to_img(t): return (t.squeeze().detach().cpu().numpy() + 1)/2
    def to_var(t): return t.squeeze().detach().cpu().numpy()
    
    # Row 1: Baselines
    axs[0, 0].imshow(to_img(x_gt), cmap='gray'); axs[0, 0].set_title(f"Ground Truth")
    axs[0, 1].imshow(to_img(result['y']), cmap='gray'); axs[0, 1].set_title("Observation")
    axs[0, 2].imshow(to_img(result['tv']['mean']), cmap='gray'); axs[0, 2].set_title("Classical TV-L2")
    axs[0, 3].imshow(to_img(result['dps']['mean']), cmap='gray'); axs[0, 3].set_title("DPS (Mode)")
    axs[0, 4].imshow(to_var(result['dps']['var']), cmap='hot'); axs[0, 4].set_title("DPS Variance")
    
    # Row 2: SMC
    axs[1, 0].imshow(to_img(x_gt), cmap='gray'); axs[1, 0].set_title(f"Ground Truth")
    axs[1, 1].imshow(to_img(result['y']), cmap='gray'); axs[1, 1].set_title("Observation")
    axs[1, 2].axis('off') # Spacer
    axs[1, 3].imshow(to_img(result['smc']['mean']), cmap='gray'); axs[1, 3].set_title("SMC Mean (Posterior)")
    
    # Error Map
    diff = torch.abs(result['smc']['mean'] - x_gt).detach().cpu().numpy().squeeze()
    axs[1, 4].imshow(diff, cmap='hot'); axs[1, 4].set_title("SMC Error |x_hat - x|")

    plt.suptitle(f"Task: {task_name} - TV vs. DPS vs. Twisted SMC", fontsize=16)
    plt.tight_layout()
    
    if not os.path.exists("results"): os.makedirs("results")
    path = f"results/benchmark_{task_name.lower()}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Visualization saved to {path}")

def visualize_ess(task_name, ess_log):
    plt.figure(figsize=(8, 4))
    plt.plot(ess_log, label='Effective Sample Size (ESS)')
    plt.axhline(y=10.0, color='r', linestyle='--', label='Resampling Threshold')
    plt.xlabel('Diffusion Step (Reverse)')
    plt.ylabel('ESS')
    plt.title(f'SMC Particle Health: {task_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = f"results/ess_{task_name.lower()}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"ESS Plot saved to {path}")

def main():
    print("Initializing Benchmark Suite...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    x_gt, _ = dataset[0]
    x_gt = x_gt.to(DEVICE)
    
    # Configs
    cfg_dps = {'T': 300, 'method': 'dps', 'scale': 0.5, 'batch_size': BATCH_SIZE}
    cfg_smc = {'T': 300, 'method': 'smc', 'batch_size': BATCH_SIZE, 'step_size': 0.1}
    
    solver_dps = DiffusionSolver(MODEL_PATH, DEVICE, cfg_dps)
    solver_smc = DiffusionSolver(MODEL_PATH, DEVICE, cfg_smc)
    
    # 1. Inpainting
    mask = torch.ones_like(x_gt)
    mask[:, 10:18, 10:18] = 0
    res_inp = run_benchmark_task("Inpainting", InpaintingOperator(mask, DEVICE), x_gt, solver_dps, solver_smc)
    visualize_task("Inpainting", res_inp, x_gt)
    if 'ess' in res_inp['smc']:
        visualize_ess("Inpainting", res_inp['smc']['ess'])
    
    # Run Super-Resolution Task
    print("\n--- Running Benchmark: SuperRes ---")
    res_sr = run_benchmark_task("SuperRes", SuperResolutionOperator(4, DEVICE), x_gt, solver_dps, solver_smc)
    visualize_task("SuperRes (4x)", res_sr, x_gt)
    
    # Run MRI Reconstruction Task
    print("\n--- Running Benchmark: MRI Reconstruction ---")
    # Acceleration 4x, 10% center (FastMRI standard-ish)
    mri_op = MRIOperator(acceleration=4.0, center_fraction=0.1, device=DEVICE) 
    res_mri = run_benchmark_task("MRI", mri_op, x_gt, solver_dps, solver_smc)
    visualize_task("MRI Reconstruction", res_mri, x_gt)

    # Run Phase Retrieval Task
    print("\n--- Running Benchmark: PhaseRetrieval ---")
    res_pr = run_benchmark_task("PhaseRetrieval", PhaseRetrievalOperator(oversampling_ratio=2.0, device=DEVICE), x_gt, solver_dps, solver_smc)
    visualize_task("PhaseRetrieval", res_pr, x_gt)
    if 'ess' in res_pr['smc']:
        visualize_ess("PhaseRetrieval", res_pr['smc']['ess'])


if __name__ == "__main__":
    main()
