
import torch
import torch.nn as nn
import math

class SimpleUnet(nn.Module):
    """
    A minimal U-Net implementation for DDPM on small images (MNIST/CIFAR).
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels)
        )

        self.down1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1, stride=2)
        self.down3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1, stride=2)

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2 + base_channels * 2, base_channels, 2, stride=2)
        self.out = nn.Conv2d(base_channels + base_channels, out_channels, 3, padding=1)

        self.act = nn.SiLU()

    def forward(self, x, t):
        # Time embedding
        t = t.float().view(-1, 1)
        t_emb = self.time_mlp(t)
        t_emb = t_emb[:, :, None, None]

        # Down
        x1 = self.act(self.down1(x))
        x2 = self.act(self.down2(x1 + t_emb))
        x3 = self.act(self.down3(x2))

        # Up
        x_up1 = self.act(self.up1(x3))
        x_up1 = torch.cat([x_up1, x2], dim=1)
        
        x_up2 = self.act(self.up2(x_up1))
        x_up2 = torch.cat([x_up2, x1], dim=1)

        return self.out(x_up2)
