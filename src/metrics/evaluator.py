
import torch
import torch.nn.functional as F
import numpy as np

def calculate_psnr(img1, img2):
    """Calculates PSNR between two tensors [B, C, H, W] in range [-1, 1]"""
    mse = F.mse_loss(img1, img2, reduction='none').mean(dim=[1,2,3])
    # dynamic range is 2 (-1 to 1)
    return 10 * torch.log10(4.0 / mse).mean().item()

def calculate_ssim(img1, img2, window_size=11):
    """
    Simplified SSIM implementation for tensors.
    """
    # Map from [-1, 1] to [0, 1]
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Mean
    mu1 = F.avg_pool2d(img1, window_size, 1, window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Variance
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
               
    return ssim_map.mean().item()

def calculate_variance_map(samples):
    """Calculates pixel-wise variance from samples [B, C, H, W]"""
    return samples.var(dim=0).mean().item()
