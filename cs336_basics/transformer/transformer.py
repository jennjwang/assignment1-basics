from torch import nn
from cs336_basics.transformer.attention import MultiheadAttention
from cs336_basics.transformer.ffn import PositionwiseFeedForward
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.attention import softmax
import torch

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.attention_norm = RMSNorm(d_model=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.ffn_norm = RMSNorm(d_model=d_model)

    def forward(self, x, token_positions=None):
        normed_x = self.attention_norm(x)
        attention_output = x + self.attention(normed_x, token_positions)
        normed_attention = self.ffn_norm(attention_output)
        return attention_output + self.ffn(normed_attention)

        # # layer_norm_ablation
        # attention_output = x + self.attention(x, token_positions)
        # ffn_output = self.ffn(attention_output)
        # res = attention_output + ffn_output
        # return res
        
        # # pre_norm_ablation
        # attention_output = x + self.attention(x, token_positions)
        # normed_x = self.attention_norm(attention_output)
        # attention_output = normed_x + self.ffn(normed_x)
        # return self.ffn_norm(attention_output)

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.context_length = context_length
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for i in range(num_layers)])
        self.norm = RMSNorm(d_model=d_model)
        self.linear = Linear(d_model, vocab_size)
    
    def forward(self, x, token_positions=None):
        token_embed = self.embedding(x)
        for layer in self.transformer_blocks:
            if token_positions is None:
                token_positions = torch.arange(token_embed.shape[-2])
            token_embed = layer(token_embed, token_positions)
        norm = self.norm(token_embed)
        output_embed = self.linear(norm)
        return output_embed


        