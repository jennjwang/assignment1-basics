from torch import nn, einsum, sigmoid
import torch

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_ff = d_ff or int(8/3 * d_model)
        self.weights_2 = nn.Parameter(torch.randn(d_model, self.d_ff))
        self.weights_3 = nn.Parameter(torch.randn(self.d_ff, d_model))
        self.weights_1 = nn.Parameter(torch.randn(self.d_ff, d_model))

    def forward(self, x):
        w1_x = einsum('h i, ...i -> ...h', self.weights_1, x)
        silu =  w1_x * sigmoid(w1_x)
        w3_x = einsum('h i, ...i -> ...h', self.weights_3, x)
        product = silu * w3_x
        w2_x = einsum('h i, ...i -> ...h', self.weights_2, product)
        return w2_x
        