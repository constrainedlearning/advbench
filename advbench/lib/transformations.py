import torch
from numpy import pi
from einops import rearrange, reduce, repeat
from kornia.geometry import warp_affine 
from kornia.geometry.transform import Affine
import functools
import math

def deg2rad(angle):
  return angle*pi/180.0

def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
    if not 2 <= len(tensor.shape) <= 4:
        raise AssertionError(f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}.")
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)
    return center


def angle_to_rotation_matrix(angle, imgs):
    """Create a rotation matrix out of angles in degrees.

    Args:
        angle: tensor of angles in degrees,  shape Bx1.

    Returns:
        tensor of rotation matrices with shape (B, 2, 3).
    """
    B = angle.shape[0]
    ang_rad = deg2rad(angle)
    cos_a = torch.cos(ang_rad)
    sin_a = torch.sin(ang_rad)
    center = _compute_tensor_center(imgs)
    rotat_m = repeat(torch.eye(3), 'd1 d2 -> b d1 d2',b=B).clone()
    a_mat = torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1)
    rotat_m[:, :2, :2] = rearrange(a_mat, 'b (d1 d2) -> b d1 d2', d1=2, d2=2).clone()
    center = repeat(_compute_tensor_center(imgs), 'd -> b d',b=B).clone()
    shift_m = txs_to_translation_matrix(center)  
    shift_m_inv = txs_to_translation_matrix(-center)
    return  shift_m @ rotat_m @ shift_m_inv

def se_matrix(delta, imgs):
  """
  delta: Bx3 (third dimension is rotation , w translation, h traslation)
  returns se: Bx2x3
  """
  angle, txs  = delta[:, 0], delta[:,1:]
  affine = torch.zeros((angle.shape[0],2,3))
  rotat_m = angle_to_rotation_matrix(angle, imgs)
  trans_m = txs_to_translation_matrix(txs)
  return (rotat_m@trans_m)[:,:2,:]

def txs_to_translation_matrix(txs):
    """Create a translation matrix out of translations in pixels.

    Args:
        txs: tensor of  translations in pixels,  shape Bx2
    Returns:
        tensor of translation matrices with shape (B, 3, 3).
    """
    shift_m = repeat(torch.eye(3), 'd1 d2 -> b d1 d2', b=txs.shape[0]).clone()
    shift_m[:, :2, 2] = txs
    return shift_m

def se_transform(imgs, delta):
    return warp_affine(imgs, se_matrix(delta, imgs).to(imgs.device).to(imgs.dtype), imgs.shape[2:]) 

def translation(imgs, delta):
  """
  delta: Bx3 ( flip, w translation, h traslation)
  returns se: Bx2x3
  """
  return Affine(translation = delta.to(imgs.device).to(imgs.dtype))(imgs)

######### DIFFEO FREQUENCY ##############
# From https://github.com/pcsl-epfl/diffeomorphism/blob/main/diff.py
#########                   ##############
@functools.lru_cache()
def scalar_field_modes(n, m, dtype=torch.float64, device='cpu'):
    """
    sqrt(1 / Energy per mode) and the modes
    """
    x = torch.linspace(0, 1, n, dtype=dtype, device=device)
    k = torch.arange(1, m + 1, dtype=dtype, device=device)
    i, j = torch.meshgrid(k, k)
    r = (i.pow(2) + j.pow(2)).sqrt()
    e = (r < m + 0.5) / r
    s = torch.sin(math.pi * x[:, None] * k[None, :])
    return e, s

def scalar_field(delta, n, m, device='cuda'):
    """
    random scalar field of size nxn made of the first m modes
    """
    d = delta.reshape()
    e, s = scalar_field_modes(n, m, dtype=torch.get_default_dtype(), device=device)
    c = torch.randn(m, m, device=device) * e
    return torch.einsum('ij,xi,yj->yx', c, s, s)


def deform(image, delta, cut=15,T=8e-6, interp='linear'):
    """
    1. Sample a displacement field tau: R2 -> R2, using tempertature `T` and cutoff `cut`
    2. Apply tau to `image`
    :param img Tensor: square image(s) [..., y, x]
    :param T float: temperature
    :param cut int: high frequency cutoff
    """
    n = image.shape[-1]
    assert image.shape[-2] == n, 'Image(s) should be square.'

    device = image.device.type

    # Sample dx, dy
    # u, v are defined in [0, 1]^2
    # dx, dx are defined in [0, n]^2
    u = scalar_field(n, cut, device)  # [n,n]
    v = scalar_field(n, cut, device)  # [n,n]
    dx = T ** 0.5 * u * n
    dy = T ** 0.5 * v * n

    # Apply tau
    return remap(image, dx, dy, interp)


def remap(a, dx, dy, interp):
    """
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