from .transform import *
from .matrices import *
from .utils import *


def transform(x, translation_x=0., translation_y=0., rotation=0., compression=0.,
              shear_a=0., shear_b=0., center_x=0., center_y=0.):
    """
    Apply linear transformations to (an) image(s).
    :param torch.Tensor x: batch of input images to transform. Shape: [B, ch, n, n]
    :param torch.Tensor or float translation_x: translation(s) along the x-axis. Shape: [B,] or []
    :param torch.Tensor or float translation_y: translation(s) along the y-axis. Shape: [B,] or []
    :param torch.Tensor or float rotation: rotation angle(s) in degree. Shape: [B,] or []
    :param torch.Tensor or float compression: compression factor. Shape: [B] or []
    :param torch.Tensor or float shear_a: shear factor along cartesian xy coordinates. Shape: [B,] or []
    :param torch.Tensor or float shear_b: shear factor along diagonals. Shape: [B,] or []
    :param torch.Tensor or float center_x: transformation(s) center(s) along the x-axis. Shape: [B,] or []
    :param torch.Tensor or float center_y: transformation(s) center(s) along the y-axis. Shape: [B,] or []
    :return torch.Tensor: transformed image.
    """

    device = x.device

    n = x.shape[-1]
    B = x.shape[0]

    T = totensor(translation_x, translation_y).to(device)
    Ce = totensor(center_x, center_y).to(device)

    R = rotation_matrix(rotation) if rotation else 0
    Co = compression_matrix(compression) if compression else 0
    S = pure_shear_matrix(shear_a, shear_b) if shear_a or shear_b else 0

    matrix = (R + Co + S).to(device)

    tau = displacement_field(n, matrix=matrix, translation=T, center=Ce)
    assert len(tau) == 1 or len(tau) == B, "Batch size of input and transformation(s) must be the same!!"
    if len(tau) == 1:
        tau = tau.expand(B, -1, -1, -1)

    return apply_displacement(x, tau)
