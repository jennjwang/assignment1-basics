import torch
from torch import nn
from einops import rearrange, einsum
import math
from cs336_basics.transformer.rope import RoPE

def softmax(x, dim):
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - max_val
    exp_x = torch.exp(x)
    dem = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / dem
    

def scaled_dot_product_attention(Q, K, V, mask=None):
    nom = einsum(Q, K, '... i d, ... j d -> ... i j')
    # print("nom", nom)
    dem = math.sqrt(K.shape[-1])
    scores = nom/dem
    # print("scores before mask", scores)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = (scores).masked_fill(mask == False, float('-inf'))
        else:
            scores = scores + mask
        # print("scores after mask", scores)
    weights = softmax(scores, dim=-1)
    res = einsum(weights, V, '... i j, ... j v -> ... i v')
    return res

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.num_heads
        self.weights_qkv = nn.Parameter(torch.zeros(3, d_model, d_model))
        self.weights_o = nn.Parameter(torch.zeros(d_model, d_model))
        std = math.sqrt(2 / (d_model + d_model))
        torch.nn.init.trunc_normal_(self.weights_qkv, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.weights_o, mean=0.0, std=std, a=-3*std, b=3*std)
        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

    def forward(self, x, token_positions=None):
        # weights_qkv: (3, num_heads, d_k, d_model) → 'nhkm'
        # x:           (batch, seq, d_model)         → 'bsm'
        # output:      (3, num_heads, batch, seq, d_k) → 'nhbsk'
        weights_qkv = rearrange(self.weights_qkv, 'n (num_heads d_k) d_model -> n num_heads d_k d_model', 
                        num_heads=self.num_heads, d_k=self.d_k, d_model=self.d_model)
        qkv = einsum(weights_qkv, x, 'n h k d, b s d -> n h b s k') 
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        
        seq_len = x.shape[-2] # " ... sequence_length d_in"
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        
        # weights_o: (d_model, num_heads, d_k) → 'mhk'
        # attention:   (num_heads, batch, seq, d_k)    → 'h..bsk'
        # output:      (batch, seq, d_model) → 'bsm'
        attention = scaled_dot_product_attention(q, k, v, mask)
        weights_o = rearrange(self.weights_o, 'd_model (num_heads d_k) -> d_model num_heads d_k',
                num_heads=self.num_heads, d_k=self.d_k, d_model=self.d_model)
        res = einsum(weights_o, attention, 'm h k,h ... b s k -> b s m')
        return res


    