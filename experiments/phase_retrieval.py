import torch
from diffusion_solver import DiffusionSolver
from operators.operators import PhaseRetrievalOperator
from metrics.metrics import psnr, ssim, lpips

def run_phase_retrieval_experiment(config, score_model, dataset):
    operator = PhaseRetrievalOperator()
    solver = DiffusionSolver(config, score_model, operator)
    results = []
    for i, (measured_img, gt_img) in enumerate(dataset):
        y = operator(measured_img)
        recon = solver.sample(y, gt_img.shape)
        results.append({
            'psnr': psnr(recon, gt_img).item(),
            'ssim': ssim(recon, gt_img),
            'lpips': lpips(recon, gt_img),
        })
        torch.save(recon.cpu(), f"results/phase_recon_{i}.pt")
    print('Phase retrieval results:', results)
    return results
