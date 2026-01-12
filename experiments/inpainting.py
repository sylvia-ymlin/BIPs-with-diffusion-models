import torch
from diffusion_solver import DiffusionSolver
from operators.operators import InpaintingOperator
from metrics.metrics import psnr, ssim, lpips

def run_inpainting_experiment(config, score_model, dataset, mask):
    operator = InpaintingOperator(mask)
    solver = DiffusionSolver(config, score_model, operator)
    results = []
    for i, (corrupted_img, gt_img) in enumerate(dataset):
        y = operator(corrupted_img)
        recon = solver.sample(y, gt_img.shape)
        results.append({
            'psnr': psnr(recon, gt_img).item(),
            'ssim': ssim(recon, gt_img),
            'lpips': lpips(recon, gt_img),
        })
        torch.save(recon.cpu(), f"results/inpaint_recon_{i}.pt")
    print('Inpainting results:', results)
    return results
