from kornia.geometry import rotate
import torch
from torchvision.transforms import Pad
from advbench.lib.transformations import se_transform, translation
from advbench.datasets import FFCV_AVAILABLE

class Perturbation():
    def __init__(self, epsilon):
        self.eps = epsilon
    def clamp_delta(self, delta):
        raise NotImplementedError
    def perturb_img(self, imgs, delta):
        return self._perturb(imgs, delta)
                
    def _perturb(self, imgs, delta):
        raise NotImplementedError
    def delta_init(self, imgs):
        raise NotImplementedError

class Linf(Perturbation):
    def __init__(self, epsilon):
        super(Linf, self).__init__(epsilon)
        self.dim = None
        self.names = ['Linf']
    def clamp_delta(self, delta, imgs):
        """Clamp delta so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""
        eps = self.eps
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(delta, -imgs, 1-imgs)
        return delta

    def _perturb(self, imgs, delta):
        if self.dim is None:
            self.dim = imgs.shape[1:]
        return imgs + delta

    def delta_init(self, imgs):
        if self.dim is None:
            self.dim = imgs.shape[1:]
        return 0.001 * torch.randn(imgs.shape, dtype=imgs.dtype, device=imgs.device)

class Translation(Perturbation):
    def __init__(self, epsilon):
        super(Translation, self).__init__(epsilon)
        self.dim = 2
        self.names = [ 'Tx', 'Ty']
    def clamp_delta(self, delta, imgs):
        """Clamp delta so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""
        for i in range(self.dim):
            delta[:, i] = torch.clamp(delta[:, i], - self.eps[i], self.eps[i])
        return delta

    def _perturb(self, imgs, delta):
        return translation(imgs, delta)

    def delta_init(self, imgs):
        delta_init = torch.empty(imgs.shape[0], self.dim, device=imgs.device, dtype=imgs.dtype)
        for i in range(self.dim):
            eps = self.eps[i]
            delta_init[:,i] =   2*eps* torch.randn(imgs.shape[0], device = imgs.device, dtype=imgs.dtype)-eps
        return delta_init
        
class Rotation(Perturbation):
    def __init__(self, epsilon):
        super(Rotation, self).__init__(epsilon)
        self.dim = 1
        self.names = ['Angle']
    def clamp_delta(self, delta, imgs):
        """Clamp delta so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""
        delta = torch.clamp(delta, -self.eps, self.eps)
        return delta

    def _perturb(self, imgs, delta):
        if delta.dim() > 1:
            return rotate(imgs, delta.squeeze())
        else:
            return rotate(imgs, delta)
        
    def delta_init(self, imgs):
        eps = self.eps
        delta_init =   2*eps* torch.rand(imgs.shape[0], dtype=imgs.dtype, device=imgs.device)-eps
        return delta_init

class SE(Perturbation):
    def __init__(self, epsilon):
        super(SE, self).__init__(epsilon)
        self.dim = 3
        self.names = ['Angle', 'Tx', 'Ty']
    def clamp_delta(self, delta, imgs):
        """Clamp delta so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""
        for i in range(self.dim):
            delta[:, i] = torch.clamp(delta[:, i], - self.eps[i], self.eps[i])
        return delta

    def _perturb(self, imgs, delta):
        return se_transform(imgs, delta)

    def delta_init(self, imgs):
        delta_init = torch.empty(imgs.shape[0], self.dim, device=imgs.device, dtype=imgs.dtype)
        for i in range(self.dim):
            eps = self.eps[i]
            delta_init[:,i] =   2*eps* torch.randn(imgs.shape[0], device = imgs.device, dtype=imgs.dtype)-eps
        return delta_init
