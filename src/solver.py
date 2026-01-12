
import torch
import os
import torch.nn.functional as F
from .model import SimpleUnet
from .sampling_strategies.dps import DPSStrategy
from .sampling_strategies.smc import SMCStrategy

class DiffusionSolver:
    def __init__(self, model_path, device, config):
        self.device = device
        self.config = config
        self._load_model(model_path)
        self.model.eval()
        self.params = self._get_schedule(config['T'])
        self.strategy = self._get_strategy(config['method'])
    
    def _load_model(self, path):
        if "/" in path:
            # Assume HuggingFace Model ID
            print(f"Loading pretrained model from HuggingFace: {path}")
            try:
                from diffusers import UNet2DModel
                self.model = UNet2DModel.from_pretrained(path).to(self.device)
                self.is_diffusers = True
            except Exception as e:
                # Fallback or Error
                print(f"Failed to load from HF: {e}")
                raise e
        elif os.path.exists(path):
            self.model = SimpleUnet().to(self.device)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.is_diffusers = False
        else:
             # Create a dummy initialized model for testing logic if no file
             print(f"Warning: Model {path} not found. Using random initialized SimpleUnet for structure testing.")
             self.model = SimpleUnet().to(self.device)
             self.is_diffusers = False

    def _get_schedule(self, T):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T).to(self.device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        return betas, sqrt_recip_alphas, posterior_variance, alphas_cumprod

    def _get_strategy(self, method):
        if method == 'dps':
            return DPSStrategy(self.config.get('scale', 0.5))
        elif method == 'smc':
            return SMCStrategy(self.config['batch_size'], step_size=self.config.get('step_size', 0.1))
        else:
            raise ValueError(f"Unknown method {method}")

    def sample(self, y, operator, shape):
        x = torch.randn(shape, device=self.device)
        T = self.config['T']
        
        # Expand y if needed
        if y.shape[0] != shape[0]:
            if y.is_complex():
                 y = torch.view_as_complex(torch.stack((y.real.repeat(shape[0], 1, 1, 1), y.imag.repeat(shape[0], 1, 1, 1)), dim=-1))
            else:
                y = y.repeat(shape[0], 1, 1, 1)

        ess_log = []
        for i in reversed(range(0, T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            
            # 1. Gradient Checkpointing (Optional - Simulated here for architecture demo)
            # In a real heavy model, wrapping model(x,t) in checkpoint
            
            # 2. Mixed Precision (AutoCast)
            with torch.cuda.amp.autocast(enabled=False): # Setup for future use
                 with torch.no_grad():
                    if self.is_diffusers:
                         # Diffusers UNet returns a class with .sample attribute
                         noise_pred = self.model(sample=x, timestep=t).sample
                    else:
                         noise_pred = self.model(x, t)
            
            # 3. Strategy Update
            # Some strategies return extra info (e.g. SMC returns ESS)
            out = self.strategy.update_step(x, noise_pred, t, y, operator, self.params)
            
            if isinstance(out, tuple):
                x, info = out
                if 'ess' in info:
                    ess_log.append(info['ess'])
            else:
                x = out
            
        return x, ess_log
