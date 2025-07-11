import torch
import torch.nn as nn
from torch.amp import custom_fwd

class Whitening(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim

        # Running statistics in float32
        self.register_buffer("running_mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("running_cov", torch.eye(dim, dtype=torch.float32))

    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)  # Ensure all operations below are in float32

        if self.training:
            mean = x.mean(dim=0)
            x_centered = x - mean

            cov = (x_centered.T @ x_centered) / (x.shape[0] - 1)
            cov += self.eps * torch.eye(self.dim, device=cov.device, dtype=torch.float32)

            U, S, _ = torch.linalg.svd(cov, full_matrices=False)
            zca_matrix = U @ torch.diag(1.0 / torch.sqrt(S + self.eps)) @ U.T

            self.running_mean.copy_(mean.detach())
            self.running_cov.copy_(cov.detach())
        else:
            mean = self.running_mean
            cov = self.running_cov
            cov += self.eps * torch.eye(self.dim, device=cov.device, dtype=torch.float32)

            U, S, _ = torch.linalg.svd(cov, full_matrices=False)
            zca_matrix = U @ torch.diag(1.0 / torch.sqrt(S + self.eps)) @ U.T
            x_centered = x - mean

        x_whitened = x_centered @ zca_matrix
        return x_whitened.to(orig_dtype)
