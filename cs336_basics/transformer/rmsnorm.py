import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones((d_model,)))
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt((1/self.d_model) * torch.einsum('... d->...', x**2) + self.eps)
        result = torch.einsum('... d, ... -> ... d', x, 1/RMS)
        result = torch.einsum("... d, d -> ... d", result, self.gain)
        return result.to(in_dtype)