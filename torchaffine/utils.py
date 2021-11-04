import torch

def zero(x):
    if x is None:
        x = torch.zeros(1, 2, 1, 1)
    else:
        x = x[..., None, None]
    return x
