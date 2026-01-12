
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleUnet
import os
from tqdm import tqdm

# Configuration
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 3
T = 300  # Diffusion steps
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def train():
    print("Starting training script...")
    # 1. Data (MNIST)
    print("Loading data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Model & Optimizer
    model = SimpleUnet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3. Diffusion Schedule
    betas = linear_beta_schedule(T).to(DEVICE)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 4. Training Loop
    print(f"Training on {DEVICE} for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(loader)
        for x, _ in pbar:
            x = x.to(DEVICE)
            t = torch.randint(0, T, (x.size(0),), device=DEVICE).long()
            
            # Forward Diffusion
            noise = torch.randn_like(x)
            sqrt_alphabar = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
            sqrt_one_minus_alphabar = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
            x_t = sqrt_alphabar * x + sqrt_one_minus_alphabar * noise
            
            # Predict Noise
            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    # 5. Save Model
    torch.save(model.state_dict(), "ddpm_mnist.pth")
    print("Model saved to ddpm_mnist.pth")

if __name__ == "__main__":
    train()
