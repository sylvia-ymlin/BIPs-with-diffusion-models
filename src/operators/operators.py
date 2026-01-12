
import torch
import torch.nn.functional as F
import abc

class MeasurementOperator(abc.ABC):
    def __init__(self, device):
        self.device = device
        
    @abc.abstractmethod
    def forward(self, x):
        pass

class InpaintingOperator(MeasurementOperator):
    def __init__(self, mask, device):
        super().__init__(device)
        self.mask = mask.to(device)
    
    def forward(self, x):
        return x * self.mask

class SuperResolutionOperator(MeasurementOperator):
    def __init__(self, factor, device):
        super().__init__(device)
        self.factor = factor
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=1/self.factor, mode='bilinear', align_corners=False, antialias=True)
        
    def transpose(self, y):
        return F.interpolate(y, scale_factor=self.factor, mode='nearest')

# Assuming LinearOperator is a base class similar to MeasurementOperator,
# or that it should inherit from MeasurementOperator or abc.ABC.
# For the purpose of this edit, I will define a placeholder LinearOperator
# if it's not already defined, or assume it's meant to be MeasurementOperator
# if the context implies it. Given the instruction, I will use LinearOperator
# as the base class for the new definitions. Since LinearOperator is not
# defined, I will make it inherit from abc.ABC for now to ensure syntactical correctness.
# If LinearOperator is meant to be MeasurementOperator, this would need adjustment.
class LinearOperator(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def forward(self, x):
        pass

class PhaseRetrievalOperator(LinearOperator):
    def __init__(self, oversampling_ratio=2.0, device='cpu'):
        super().__init__()
        self.oversampling_ratio = oversampling_ratio
        self.device = device
    
    def forward(self, x):
        # |F(x)|
        # Pad for oversampling
        pad = int((self.oversampling_ratio - 1) * x.shape[-1] / 2)
        x_pad = F.pad(x, (pad, pad, pad, pad))
        fft = torch.fft.fft2(x_pad)
        return torch.abs(fft)
    
    def distance(self, y_pred, y_meas):
        # || |F(x)| - y ||^2
        return torch.sum((y_pred - y_meas)**2)

class MRIOperator(LinearOperator):
    """
    Simulates Compressed Sensing MRI:
    y = M * F(x)
    where F is 2D FFT, M is a binary k-space mask.
    The measurement y is complex-valued k-space data.
    """
    def __init__(self, acceleration=4.0, center_fraction=0.1, device='cpu'):
        super().__init__()
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.device = device
        self.mask = None
        
    def _create_mask(self, shape):
        if self.mask is not None and self.mask.shape[-2:] == shape[-2:]:
            return self.mask
            
        b, c, h, w = shape
        num_cols = w
        num_low_freqs = int(round(num_cols * self.center_fraction))
        
        # Create a mask with random columns but keeping center columns
        prob = (num_cols / self.acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask_vector = torch.rand(num_cols) < prob
        
        # Always keep center
        pad = (num_cols - num_low_freqs + 1) // 2
        mask_vector[pad:pad+num_low_freqs] = True
        
        # Reshape to (1, 1, 1, W) and broadcast to (1, 1, H, W)
        self.mask = mask_vector.view(1, 1, 1, w).expand(1, 1, h, w).float().to(self.device)
        return self.mask

    def forward(self, x):
        # y = Mask * FFT(x)
        mask = self._create_mask(x.shape)
        # Use ortho norm to preserve energy
        fft = torch.fft.fft2(x, norm='ortho')
        return mask * fft
        
    def transpose(self, y):
        # Zero-filled reconstruction: IFFT(y) (since Mask is its own transpose, and y is already masked)
        return torch.fft.ifft2(y, norm='ortho').real.float()

    def distance(self, x_pred, y_meas):
        # Likelihood in k-space: || Mask * F(x) - y_meas ||^2
        # Complex difference
        y_pred = self.forward(x_pred)
        # We process complex values as 2-channel real for distance if needed, or just sum of magnitude squared
        diff = y_pred - y_meas
        return torch.sum(diff.real**2 + diff.imag**2)

class GaussianBlurOperator(MeasurementOperator):
    def __init__(self, kernel_size, sigma, device):
        super().__init__(device)
        # Create Gaussian Kernel
        self.kernel = self._get_gaussian_kernel(kernel_size, sigma).to(device)
        self.padding = kernel_size // 2
        
    def _get_gaussian_kernel(self, size, sigma):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords**2) / (2*sigma**2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1) # 2D
        return kernel.unsqueeze(0).unsqueeze(0) # [1, 1, K, K]

    def forward(self, x):
        # Depthwise conv
        return F.conv2d(x, self.kernel, padding=self.padding)
