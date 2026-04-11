import math

def learning_rate_schedule(t, max_lr, min_lr, warmup_iters, cos_iters):
    lr_t = min_lr
    if t < warmup_iters:
        lr_t = (t / warmup_iters) * max_lr
    if warmup_iters <= t and t <= cos_iters:
        lr_t = min_lr + 1/2 * (1 + math.cos((t - warmup_iters) / (cos_iters - warmup_iters) * math.pi)) * (max_lr - min_lr)
    if t > cos_iters:
        lr_t = min_lr
    
    return lr_t