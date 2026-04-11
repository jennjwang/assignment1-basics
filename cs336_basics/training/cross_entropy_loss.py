import torch
from cs336_basics.transformer.attention import softmax
from einops import rearrange

def cross_entropy_loss(inputs, targets):
    max_value = torch.max(inputs, dim=1)[0] # (batch size, )
    max_value = rearrange(max_value, 'batch -> batch 1')
    shifted = inputs - max_value
    log_sum = torch.log(torch.sum(torch.exp(shifted), dim=1)) # (batch size, )
    gathered = torch.gather(shifted, 1, targets.unsqueeze(1))
    gathered = rearrange(gathered, 'batch 1 -> batch')
    loss = -(gathered - log_sum)
    return torch.mean(loss)


