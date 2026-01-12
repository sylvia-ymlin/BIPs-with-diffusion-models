import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
from model import SimpleUnet

# --- Configuration ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "ddpm_mnist.pth"
T = 300
NUM_SAMPLES = 50

def get_schedule():
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, T).to(DEVICE)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return betas, sqrt_recip_alphas, posterior_variance, alphas_cumprod

class SamplingStrategy:
    def step(self, x_t, noise_pred, t, y, mask, model_params):
        raise NotImplementedError

class DPSStrategy(SamplingStrategy):
    def step(self, x_t, noise_pred, t, y, mask, model_params):
        betas, sqrt_recip_alphas, posterior_variance, alphas_cumprod = model_params
        idx = t[0].item()
        
        sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[idx])
        sqrt_alpha = torch.sqrt(alphas_cumprod[idx])
        x_0_hat = (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha
        
        # Reduced scale and clamping
        scale = 0.5
        measurement_error = y - (x_0_hat * mask)
        grad = - measurement_error 
        
        mean = sqrt_recip_alphas[idx] * (x_t - betas[idx] * noise_pred / sqrt_one_minus)
        mean = mean - scale * grad 
        
        noise = torch.randn_like(x_t) if idx > 0 else 0
        x_prev = mean + torch.sqrt(posterior_variance[idx]) * noise
        x_prev = torch.clamp(x_prev, -3.0, 3.0) # Safety Clamp
        return x_prev

class SMCStrategy(SamplingStrategy):
    def step(self, x_t, noise_pred, t, y, mask, model_params):
        betas, sqrt_recip_alphas, posterior_variance, alphas_cumprod = model_params
        idx = t[0].item()

        sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[idx])
        sqrt_alpha = torch.sqrt(alphas_cumprod[idx])
        
        mean = sqrt_recip_alphas[idx] * (x_t - betas[idx] * noise_pred / sqrt_one_minus)
        noise = torch.randn_like(x_t) if idx > 0 else 0
        x_new = mean + torch.sqrt(posterior_variance[idx]) * noise
        
        # Resample every 20 steps, gentle weights
        if idx % 20 == 0:
            x_0_hat = (x_new - sqrt_one_minus * noise_pred) / sqrt_alpha
            dist = torch.sum(((x_0_hat * mask) - y)**2, dim=[1,2,3])
            
            # Gentle weighting to preserve diversity
            log_weights = - dist * 0.5 
            
            log_weights = log_weights - torch.max(log_weights)
            weights = torch.exp(log_weights)
            weights = weights / (torch.sum(weights) + 1e-8)
            
            indices = torch.multinomial(weights, NUM_SAMPLES, replacement=True)
            x_new = x_new[indices]
            
        return x_new

class DiffusionSolver:
    def __init__(self, model_path, device="cpu", strategy="dps"):
        self.device = device
        self.model = SimpleUnet().to(device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise FileNotFoundError("Model not found")
        self.model.eval()
        self.params = get_schedule()
        
        if strategy == "dps": self.strategy = DPSStrategy()
        elif strategy == "smc": self.strategy = SMCStrategy()
        else: raise ValueError("Unknown strategy")

    def solve(self, y, mask, shape):
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(0, T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            with torch.no_grad():
                noise_pred = self.model(x, t)
            x = self.strategy.step(x, noise_pred, t, y, mask, self.params)
        return x

def run_experiment():
    print("Initializing Experiment...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    x_gt, _ = dataset[0]
    x_gt = x_gt.to(DEVICE).unsqueeze(0)
    
    mask = torch.ones_like(x_gt)
    mask[:, :, 10:18, 10:18] = 0
    y = x_gt * mask
    y_batch = y.repeat(NUM_SAMPLES, 1, 1, 1)
    
    # Run
    print("Running DPS...")
    solver_dps = DiffusionSolver(MODEL_PATH, DEVICE, strategy="dps")
    samples_dps = solver_dps.solve(y_batch, mask, (NUM_SAMPLES, 1, 28, 28))
    
    print("Running SMC...")
    solver_smc = DiffusionSolver(MODEL_PATH, DEVICE, strategy="smc")
    samples_smc = solver_smc.solve(y_batch, mask, (NUM_SAMPLES, 1, 28, 28))
    
    # Metric Calculation
    mean_dps = samples_dps.mean(dim=0)
    var_dps = samples_dps.var(dim=0)
    mean_smc = samples_smc.mean(dim=0)
    var_smc = samples_smc.var(dim=0)
    
    # MSE
    mse_dps = F.mse_loss(mean_dps.unsqueeze(0), x_gt)
    mse_smc = F.mse_loss(mean_smc.unsqueeze(0), x_gt)
    
    print(f"\n=== QUANTITATIVE ANALYSIS ===")
    print(f"DPS - MSE: {mse_dps.item():.4f}, Mean Var: {var_dps.mean().item():.4f}")
    print(f"SMC - MSE: {mse_smc.item():.4f}, Mean Var: {var_smc.mean().item():.4f}")
    
    # Plot
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    def to_img(t): return (t.squeeze().cpu().numpy() + 1)/2
    def to_var(t): return t.squeeze().cpu().numpy()
    
    axs[0,0].imshow(to_img(x_gt), cmap="gray"); axs[0,0].set_title("Original")
    axs[0,1].imshow(to_img(y), cmap="gray"); axs[0,1].set_title("Corrupted")
    axs[0,2].imshow(to_img(mean_dps), cmap="gray"); axs[0,2].set_title("DPS Mean")
    axs[0,3].imshow(to_var(var_dps), cmap="hot"); axs[0,3].set_title(f"DPS Var\n({var_dps.mean().item():.4f})")
    
    axs[1,0].imshow(to_img(x_gt), cmap="gray")
    axs[1,1].imshow(to_img(y), cmap="gray")
    axs[1,2].imshow(to_img(mean_smc), cmap="gray"); axs[1,2].set_title("SMC Mean")
    axs[1,3].imshow(to_var(var_smc), cmap="hot"); axs[1,3].set_title(f"SMC Var\n({var_smc.mean().item():.4f})")
    
    plt.tight_layout()
    if not os.path.exists("results"): os.makedirs("results")
    plt.savefig("results/bias_analysis_comparison.png")
    print("Saved plot.")

if __name__ == "__main__":
    run_experiment()
