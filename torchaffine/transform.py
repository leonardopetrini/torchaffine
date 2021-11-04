import torch
from .utils import remap, zero

def transform_coordinates(n, matrix=None, translation=None, center=None):
    """
    int [B, 2, 2] [B, 2] [B, 2]
    """
    if matrix is None:
        matrix = torch.eye(2)[None]
    center, translation = zero(center), zero(translation)
    X = torch.stack(torch.meshgrid(torch.arange(0, n), torch.arange(0, n))).float().flip(0)
    return torch.einsum('bji,jnm->binm', matrix, X - center[0]) + center - translation


def tx(x, new_coordinates):
    return torch.stack([remap(x[i], *new_coordinates[i], interp='linear') for i in range(len(x))])