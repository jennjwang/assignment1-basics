import torch
from typing import Optional, Callable
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas, eps, weight_decay, lr=1e-3):
        if lr < 0 or betas[0] < 0 or betas[0] > 1 or betas[1] < 0 or betas[1] > 1 or eps < 0 or weight_decay < 0:
            raise ValueError(f"Invalid learning rate: {lr}, beta: {beta}, eps: {eps}, weight_decay: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, betas, eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                if t == 0:
                    m = torch.zeros_like(p)
                    v = torch.zeros_like(p)
                else:
                    m = state['m']
                    v = state['v']
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                lr_t = lr * math.sqrt(1 - betas[1] ** (t + 1)) / (1 - betas[0] ** (t + 1))
                p.data = p.data - lr * weight_decay * p.data # Apply weight decay
                m = betas[0] * m + (1 - betas[0]) * grad # Update the first moment estimate
                v = betas[1] * v + (1 - betas[1]) * grad**2 # Update the second moment estimate
                p.data -= lr_t * m / (torch.sqrt(v) + eps)  # Apply moment-adjusted weight updates
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1  # Increment iteration number.
        return loss
    

