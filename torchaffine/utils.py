"""
    Support functions.
"""
import torch

def zero(x):
    if x is None:
        x = torch.zeros(1, 2, 1, 1)
    else:
        x = x[..., None, None]
    return x


def totensor(x, y):
    if isinstance(x, (float, int)):
        x, y = torch.tensor(x), torch.tensor(y)
    if not len(x.shape):
        x, y = x[None], y[None]
    return torch.stack([x, y]).t()
