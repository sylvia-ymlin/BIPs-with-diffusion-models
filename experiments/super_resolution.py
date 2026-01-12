import torch
from diffusion_solver import DiffusionSolver
from operators.operators import InpaintingOperator, SuperResolutionOperator, GaussianDeblurOperator, PhaseRetrievalOperator
from metrics.metrics import psnr, ssim, lpips, fid, sliced_wasserstein

# 示例：4x超分辨率任务

def run_super_resolution_experiment(config, score_model, dataset):
    operator = SuperResolutionOperator(scale=4)
    solver = DiffusionSolver(config, score_model, operator)
    results = []
    for i, (lr_img, hr_img) in enumerate(dataset):
        y = operator(lr_img)
        recon = solver.sample(y, hr_img.shape)
        results.append({
            'psnr': psnr(recon, hr_img).item(),
            'ssim': ssim(recon, hr_img),
            'lpips': lpips(recon, hr_img),
            # 可扩展fid、sw等
        })
        # 保存重建图片
        torch.save(recon.cpu(), f"results/sr_recon_{i}.pt")
    # 汇总结果
    print('Super-resolution results:', results)
    return results

# 其他任务（修复、去模糊、相位恢复）可仿照此结构实现
