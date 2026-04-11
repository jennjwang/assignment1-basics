import math

from torch import einsum, sqrt
import torch
import torch.nn as nn

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # matmul: einsum('i j, j k -> i k', a, b)
        return einsum('o i, ... i ->  ... o', self.weights, x)
