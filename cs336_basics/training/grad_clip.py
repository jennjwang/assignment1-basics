# run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
#     """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

#     Args:
#         parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
#         max_l2_norm (float): a positive value containing the maximum l2-norm.

#     The gradients of the parameters (parameter.grad) should be modified in-place.
#     """

import torch

def gradient_clipping(parameters, max_l2_norm):

    grads = [p.grad for p in parameters if p.grad is not None]
    norm = torch.linalg.norm(torch.cat([g.flatten() for g in grads]))
    if norm > max_l2_norm:
        scale = max_l2_norm / (norm + 1e-6)
        for grad in grads:
            grad.data *= scale