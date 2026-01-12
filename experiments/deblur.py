import torch
from diffusion_solver import DiffusionSolver
from operators.operators import GaussianDeblurOperator
from metrics.metrics import psnr, ssim, lpips

def run_deblur_experiment(config, score_model, dataset, kernel):
    operator = GaussianDeblurOperator(kernel)
    solver = DiffusionSolver(config, score_model, operator)
    results = []
    for i, (blurred_img, gt_img) in enumerate(dataset):
        y = operator(blurred_img)
        recon = solver.sample(y, gt_img.shape)
        results.append({
            'psnr': psnr(recon, gt_img).item(),
            'ssim': ssim(recon, gt_img),
            'lpips': lpips(recon, gt_img),
        })
        torch.save(recon.cpu(), f"results/deblur_recon_{i}.pt")
    print('Deblur results:', results)
    return results
