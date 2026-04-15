import torch
from torch import nn, einsum, cos, sin
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len):
        super().__init__()
        self.seq_len = max_seq_len
        self.d_k = d_k
        pos = torch.arange(max_seq_len).unsqueeze(-1)
        k = torch.arange(1, d_k//2 + 1)
        angle = pos / theta ** ((2*k - 2) / d_k)
        cos_tensor = torch.cos(angle)
        sin_tensor = torch.sin(angle)
        self.register_buffer("cos_tensor", cos_tensor, persistent=False)
        self.register_buffer("sin_tensor", sin_tensor, persistent=False)
    
    def forward(self, x, token_positions):
        x1 = x[..., ::2] # take every 2nd element starting from index 0 
        x2 = x[..., 1::2] # take every 2nd element starting from index 1

        cos_c = self.cos_tensor[token_positions]
        sin_c = self.sin_tensor[token_positions]

        cos_c = rearrange(cos_c, '... i d -> ... 1 i d')
        sin_c = rearrange(sin_c, '... i d -> ... 1 i d')
        rotated_x1 = cos_c * x1 - sin_c * x2
        rotated_x2 = sin_c * x1 + cos_c * x2

        # want (..., d_k)
        rotated = torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(-2)
        return rotated
        