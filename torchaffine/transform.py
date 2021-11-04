import torch
from .utils import zero

def transform_coordinates(n, matrix=None, translation=None, center=None):
    """
    :param int n: image size.
    :param torch.Tensor matrix: transformation matrices. Shape: [B, 2, 2]
    :param torch.Tensor translation: translation vectors. Shape: [B, 2]
    :param torch.Tensor center: transformation centers. Shape: [B, 2]
    :return torch.Tensor: new transformed coordinates. [B, 2, n, n]
    """
    if matrix is None:
        matrix = torch.eye(2)[None]
    center, translation = zero(center), zero(translation)
    X = torch.stack(torch.meshgrid(torch.arange(0, n), torch.arange(0, n))).float().flip(0)
    return torch.einsum('bji,jnm->binm', matrix, X - center[0]) + center - translation


def tx(x, new_coordinates):
    """
    Remap `x` to new coordinates.
    :param torch.Tensor x: original image. [B, ch, n, n]
    :param torch.Tensor new_coordinates: new image coordinates. [B, n, n]
    :return torch.Tensor: x' [B, ch, n, n].
    """
    return torch.stack([remap(x[i], *new_coordinates[i], interp='linear') for i in range(len(x))])

def remap(a, xn, yn, interp):
    """
    adapted from https://github.com/pcsl-epfl/diffeomorphism
    :param a: Tensor of shape [..., y, x]
    :param xn: Tensor of shape [y, x]
    :param yn: Tensor of shape [y, x]
    :param interp: interpolation method
    """
    n, m = a.shape[-2:]
    assert xn.shape == (n, m) and yn.shape == (n, m), 'Image(s) and displacement fields shapes should match.'

    xn = xn.clamp(0, m-1)
    yn = yn.clamp(0, n-1)

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
        y, x = torch.meshgrid(torch.arange(n, dtype=xn.dtype), torch.arange(m, dtype=xn.dtype))
        dx = (xn[:, :, None, None] - x)
        dy = (yn[:, :, None, None] - y)

        c = (-dx**2 - dy**2).div(2 * sigma**2).exp()
        c = c / c.sum([2, 3], keepdim=True)

        return (c * a[..., None, None, :, :]).sum([-1, -2])
