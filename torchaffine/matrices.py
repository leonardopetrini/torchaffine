"""
    Compute rotation, compression and shear matrices from parameters.
"""
import math
import torch

def rotation_matrix(theta):
    """
    Compute rotation matrices for a batch of angles `theta`.
    :param float or torch.Tensor theta: angle of rotation. Shape: [B,] or []
    :return: rotation matrix. Shape: [B, 2, 2]
    """
    theta = -theta
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta)
    if not len(theta.shape):
        theta = theta[None]
    theta = theta / 180 * math.pi
    sin, cos = theta.sin(), theta.cos()
    return torch.cat([cos, -sin, sin, cos]).reshape(2, 2, -1).permute(2, 0, 1)


def compression_matrix(s):
    """
    Compute compression matrices for a batch of scales `s`.
    :param float or torch.Tensor s: compression scale. Shape: [B,] or []
    :return: compression matrix. Shape: [B, 2, 2]
    """
    s = -s
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s)
    if not len(s.shape):
        s = s[None]
    return torch.einsum('b,ij->bij', 1 + s, torch.eye(2))

def pure_shear_matrix(a, b):
    """
    Compute pure shear matrices for a batch of parameters `a`, `b`.
    :param float or torch.Tensor a: pure shear along x-y axes. Shape: [B,] or []
    :param float or torch.Tensor b: pure shear along diagonals. Shape: [B,] or []
    :return: shear matrix. Shape: [B, 2, 2]
    """
    a, b = -a, -b
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
        b = torch.tensor(b)
    if not len(a.shape):
        a = a[None]
        b = b[None]
    return torch.cat([1+a, b, b, 1-a]).reshape(2, 2, -1).permute(2, 0, 1)


