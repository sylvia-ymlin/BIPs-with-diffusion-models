import torch
import torch.nn.functional as F

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim(img1, img2):
    # 这里只写接口，建议用skimage或torchmetrics实现
    return 0.0

def lpips(img1, img2):
    # 这里只写接口，建议用官方LPIPS库
    return 0.0

def fid(fake_imgs, real_imgs):
    # 这里只写接口，建议用官方FID实现
    return 0.0

def sliced_wasserstein(fake_samples, real_samples):
    # 这里只写接口
    return 0.0

# 自动保存定量与定性结果
import os
import torchvision.utils as vutils
def save_results(img_tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if img_tensor.ndim == 4:
        # 批量保存为网格
        vutils.save_image(img_tensor, save_path, nrow=4, normalize=True, scale_each=True)
    else:
        vutils.save_image(img_tensor, save_path, normalize=True, scale_each=True)

def save_metrics(metrics_dict, save_path):
    import json
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
