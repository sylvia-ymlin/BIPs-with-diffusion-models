
import torch
import torch.nn.functional as F
from .base import SamplingStrategy

class DPSStrategy(SamplingStrategy):
    def __init__(self, step_size=0.5):
        self.scale = step_size

    def update_step(self, x_t, noise_pred, t, y, operator, model_params):
        betas, sqrt_recip_alphas, posterior_variance, alphas_cumprod = model_params
        idx = t[0].item()
        
        # 1. Estimate x0 (Tweedie)
        sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[idx])
        sqrt_alpha = torch.sqrt(alphas_cumprod[idx])
        x_0_hat = (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha
        
        # 2. Guidance (Likelihood Gradient)
        # Assuming linear approx for stability in demo
        measurement = operator.forward(x_0_hat)
        
        # Handling different operator outputs shapes (e.g. SR)
        diff = y - measurement
        
        # Simple back-projection for demo (replace with autograd for rigorous)
        if hasattr(operator, 'transpose'):
            grad_approx = - operator.transpose(diff)
        elif hasattr(operator, 'mask'):
            grad_approx = - diff
        else:
             # Fallback: assume same dimension
             grad_approx = - diff

        # 3. Step
        mean = sqrt_recip_alphas[idx] * (x_t - betas[idx] * noise_pred / sqrt_one_minus)
        mean = mean - self.scale * grad_approx
        
        noise = torch.randn_like(x_t) if idx > 0 else 0
        x_prev = mean + torch.sqrt(posterior_variance[idx]) * noise
        return torch.clamp(x_prev, -4, 4)
