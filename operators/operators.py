import torch

class InpaintingOperator:
    def __init__(self, mask):
        self.mask = mask
    def __call__(self, x):
        return self.mask * x

class SuperResolutionOperator:
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, x):
        # 假设x为NCHW，简单下采样
        return torch.nn.functional.interpolate(x, scale_factor=1/self.scale, mode='bicubic')

class GaussianDeblurOperator:
    def __init__(self, kernel):
        self.kernel = kernel
    def __call__(self, x):
        # 频域卷积，略
        return x

class PhaseRetrievalOperator:
    def __call__(self, x):
        # 取模的傅里叶变换
        return torch.abs(torch.fft.fft2(x))
