import math
import torch

def rotation_matrix(theta):
    theta = -theta
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta)
    theta = theta / 180 * math.pi
    sin, cos = theta.sin(), theta.cos()
    return torch.tensor([[cos, -sin], [sin, cos]])


def compression_matrix(s):
    s = -s
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s)
    return torch.tensor([[1 + s, 0], [0, 1 + s]])


def shear_matrix(a, b):
    a, b = -a, -b
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
        b = torch.tensor(b)
    return torch.tensor([[1 + a, b], [b, 1 - a]])


