
import torch
import torch.nn.functional as F
import torch.optim as optim

class TVReconstruction:
    """
    Classical Total Variation (TV) Regularization Baseline.
    Solves: min_x ||A(x) - y||^2 + lambda * ||grad(x)||_1
    """
    def __init__(self, device, lambda_tv=0.01, num_steps=500, lr=0.01):
        self.device = device
        self.lambda_tv = lambda_tv
        self.num_steps = num_steps
        self.lr = lr

    def solve(self, y, operator, shape):
        # Initialize x (e.g., with adjoint or zero)
        # We start with random noise or zero to show convergence
        # Better: Start with adjoint for fairer comparison
        # But A^T is hard to define generically here without explicit method.
        # We start with simple interpolation for SR or masked for Inpainting if possible.
        # For generic operator, we start with random.
        x = torch.zeros(shape, device=self.device, requires_grad=True)
        
        optimizer = optim.Adam([x], lr=self.lr)
        
        for i in range(self.num_steps):
            optimizer.zero_grad()
            
            # Measurement Consistency
            # Use operator's distance metric if available (e.g. for complex MRI data)
            if hasattr(operator, 'distance'):
                loss_meas = operator.distance(x, y)
            else:
                y_pred = operator.forward(x)
                loss_meas = F.mse_loss(y_pred, y)
            
            # TV Regularization
            # approximations of horizontal and vertical gradients
            dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
            dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
            loss_tv = torch.mean(dx) + torch.mean(dy)
            
            loss = loss_meas + self.lambda_tv * loss_tv
            
            loss.backward()
            optimizer.step()
            
            # Simple projection to valid range [-1, 1] if needed, 
            # though usually TV is in [0,1]. Our diffusion is [-1, 1].
            with torch.no_grad():
                x.clamp_(-1, 1)
                
        return x.detach()
