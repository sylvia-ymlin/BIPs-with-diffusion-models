
import torch
import torch.nn.functional as F

class MeasurementOperator:
    def forward(self, x):
        raise NotImplementedError
    
    def transpose(self, y):
        raise NotImplementedError

class InpaintingOperator(MeasurementOperator):
    def __init__(self, mask, device):
        self.mask = mask.to(device)
    
    def forward(self, x):
        return x * self.mask
        
class SuperResolutionOperator(MeasurementOperator):
    def __init__(self, factor, device):
        self.factor = factor
        self.device = device
    
    def forward(self, x):
        # Downsample
        return F.interpolate(x, scale_factor=1/self.factor, mode='bilinear', align_corners=False)
    
    def transpose(self, y):
        # Naive upsample
        return F.interpolate(y, scale_factor=self.factor, mode='nearest')

class PhaseRetrievalOperator(MeasurementOperator):
    def __init__(self, device):
        self.device = device
        
    def forward(self, x):
        return torch.abs(torch.fft.fft2(x))
