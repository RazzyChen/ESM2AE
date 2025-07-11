import torch

class SwiGLU(torch.nn.Module):
    """
    SwiGLU activation function.
    """
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)