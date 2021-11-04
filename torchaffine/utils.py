import torch

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


def zero(x):
    if x is None:
        x = torch.zeros(1, 2, 1, 1)
    else:
        x = x[..., None, None]
    return x
