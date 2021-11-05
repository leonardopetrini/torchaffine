"""
    Compute displacement field and apply it to x.
"""
import torch
from .utils import zero


def displacement_field(n, matrix=None, translation=None, center=None):
    """
    Compute displacement field for given linear transformations.
    :param int n: image size.
    :param torch.Tensor matrix: transformation matrices. Shape: [B, 2, 2]
    :param torch.Tensor translation: translation vectors. Shape: [B, 2]
    :param torch.Tensor center: transformation centers. Shape: [B, 2]
    :return torch.Tensor: displacement field. [B, 2, n, n]
    """
    if matrix is None or isinstance(matrix, (int, float)):
        matrix = torch.zeros(2, 2)[None]
    center, translation = zero(center), zero(translation)

    X = torch.stack(torch.meshgrid(torch.arange(0, n), torch.arange(0, n))).float().flip(0)
    return torch.einsum('bij,bjnm->binm', matrix, X[None] - center) + translation


def apply_displacement(x, tau):
    """
    Apply the displacement field `tau` to the pixels positions of image `x`.
    :param torch.Tensor x: original image. [B, ch, n, n]
    :param torch.Tensor tau: displacement field. [B, n, n]
    :return torch.Tensor: `\tau x` [B, ch, n, n].
    """
    return torch.stack([remap(x[i], *tau[i], interp='linear') for i in range(len(x))])


def remap(a, dx, dy, interp):
    """
    adapted from https://github.com/pcsl-epfl/diffeomorphism
    :param a: Tensor of shape [..., y, x]
    :param dx: Tensor of shape [y, x]
    :param dy: Tensor of shape [y, x]
    :param interp: interpolation method
    """
    n, m = a.shape[-2:]
    assert dx.shape == (n, m) and dy.shape == (n, m), 'Image(s) and displacement fields shapes should match.'

    y, x = torch.meshgrid(torch.arange(n, dtype=dx.dtype), torch.arange(m, dtype=dx.dtype))

    xn = (x - dx).clamp(0, m-1)
    yn = (y - dy).clamp(0, n-1)

    if interp == 'linear':
        xf = xn.floor().long()
        yf = yn.floor().long()
        xc = xn.ceil().long()
        yc = yn.ceil().long()

        xv = xn - xf
        yv = yn - yf

        return (1-yv)*(1-xv)*a[..., yf, xf] + (1-yv)*xv*a[..., yf, xc] + yv*(1-xv)*a[..., yc, xf] + yv*xv*a[..., yc, xc]

    if interp == 'gaussian':
        # can be implemented more efficiently by adding a cutoff to the Gaussian
        sigma = 0.4715

        dx = (xn[:, :, None, None] - x)
        dy = (yn[:, :, None, None] - y)

        c = (-dx**2 - dy**2).div(2 * sigma**2).exp()
        c = c / c.sum([2, 3], keepdim=True)

        return (c * a[..., None, None, :, :]).sum([-1, -2])
