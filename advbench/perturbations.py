from kornia.geometry import rotate
import torch
from torchvision.transforms import Pad
from advbench.lib.transformations import se_transform, translation, translate_pointcloud
from kornia.geometry.transform import crop_by_boxes
from kornia.geometry.bbox import bbox_generator
from einops import repeat
from advbench.datasets import FFCV_AVAILABLE
try:
    from libcpab import Cpab
except:
    print("CPAB not available")
    CPAB_AVAILABLE = False

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


class CPAB(Perturbation):
    def __init__(self, epsilon, tesselation=10):
        super(CPAB, self).__init__(epsilon)
        self.names = "norm"
        self.T = Cpab([tesselation, tesselation], backend="pytorch", device='gpu', zero_boundary=True)
        _verbose = True
        self.dim = self.T.identity().shape[1]
        self.norm = epsilon

    def clamp_delta(self, delta, imgs):
        """Clamp delta so that (1) the diffeomorphism has bounded derivative."""
        #norm = torch.linalg.norm(delta, dim=1)/self.dim
        norm, _ = torch.max(torch.abs(delta), dim=1)
        delta_norm = torch.div(delta.T, norm).T
        return (norm>self.norm)[:, None] * self.norm * delta_norm + (norm<self.norm)[:, None]*delta
        
    def _perturb(self, imgs, delta):
        if FFCV_AVAILABLE:
            return self.T.transform_data(imgs.float().cuda(), delta.float().cuda(), outsize = tuple(imgs.shape[2:])).half()
        else:
            return self.T.transform_data(imgs, delta, outsize = imgs.shape[2:])

    def delta_init(self, imgs):
        delta_init = self.T.sample_transformation(imgs.shape[0])
        linf, _ = torch.max(torch.abs(delta_init), dim=1)
        #delta_init = [d*self.norm for d in delta_init]
        return delta_init#torch.stack(delta_init)

class Crop(Perturbation):
    def __init__(self, epsilon):
        # Epsilon stores padding
        super(Crop, self).__init__([epsilon, epsilon])
        self.dim = 2
        self.names = ['Tx', 'Ty']
        self.pad = Pad(self.eps, fill=0, padding_mode='constant')
        #self.eps = [2*e for e in self.eps]
        self.indexes = []
    
    def clamp_delta(self, delta, imgs):
        """Clamp delta."""
        for i in range(self.dim):
            delta[:, i] = torch.clamp(delta[:, i], -self.eps[i], self.eps[i])
        return delta

    def _perturb(self, imgs, delta):
        h, w = imgs.shape[-2:]
        padded_images = self.pad(imgs)
        grid = self.crop_grid(imgs,delta)
        perturbed_imgs = torch.nn.functional.grid_sample(padded_images, grid)
        return perturbed_imgs.to(dtype=imgs.dtype, device=imgs.device)

    def delta_init(self, imgs):
        dims = imgs.shape[2:]
        delta_init = torch.empty(imgs.shape[0], self.dim, device=imgs.device, dtype=imgs.dtype)
        for i in range(self.dim):
            delta_init[:,i] =  torch.round((2*self.eps[i])*(torch.rand(imgs.shape[0], device = imgs.device, dtype=imgs.dtype)-0.5))
        self.grid = self.build_grid(imgs.shape[2], imgs.shape[2]).repeat(imgs.size(0),1,1,1).to(device=imgs.device)
        return delta_init

    def build_grid(self, source_size, target_size):
        k = 1#float(target_size)/float(source_size)
        direct = torch.linspace(-k,k,target_size).unsqueeze(0).repeat(target_size,1).unsqueeze(-1)
        full = torch.cat([direct,direct.transpose(1,0)],dim=2).unsqueeze(0)
        return full
    
    def crop_grid(self, x, delta):
        try:
            grid = self.grid.clone()
        except:
            grid = self.build_grid(x.shape[2], x.shape[2]).repeat(x.size(0),1,1,1).to(device=x.device)
        #Add shifts by x
        grid[:,:,:,0] = grid[:,:,:,0]+ (delta[:, 0]).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))/x.size(-2)
        #Add shifts by y
        grid[:,:,:,1] = grid[:,:,:,1]+ (delta[:, 1]).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))/x.size(-1)
        return grid
'''
class Crop(Perturbation):
    def __init__(self, epsilon):
        # Epsilon stores padding
        super(Crop, self).__init__([epsilon, epsilon])
        self.dim = 2
        self.names = ['Tx', 'Ty']
        self.pad = Pad(self.eps, fill=0, padding_mode='constant')
        #self.eps = [2*e for e in self.eps]
        self.indexes = []
    
    def clamp_delta(self, delta, imgs):
        """Clamp delta."""
        for i in range(self.dim):
            delta[:, i] = torch.clamp(torch.round(delta[:, i]), -self.eps[i], self.eps[i]).to(dtype=torch.int64)
        return delta

    def _perturb(self, imgs, delta):
        delta = delta.to(dtype=torch.int64)
        h, w = imgs.shape[-2:]
        perturbed_imgs = torch.empty_like(imgs)
        padded_imgs = self.pad(imgs)
        for i in range(imgs.shape[0]):
            dh, dw = self.eps[0]+delta[i,0], self.eps[1]+delta[i,1]
            perturbed_imgs[i] = padded_imgs[i,:,dh:h+dh, dw:w+dw]
        return perturbed_imgs

    def delta_init(self, imgs):
        delta_init = torch.empty(imgs.shape[0], self.dim, device=imgs.device, dtype=imgs.dtype)
        for i in range(self.dim):
            delta_init[:,i] =  torch.round(self.eps[i]*2*(torch.rand(imgs.shape[0], device = imgs.device, dtype=imgs.dtype)-0.5))
        #self.grid = []
        #for i in range(2):
            #self.grid.append(torch.arange(imgs.shape[-2-i]+self.eps[i], device=imgs.device))       
        return delta_init
    

class Crop_and_Flip(Crop):
    def __init__(self, epsilon):
        super(Crop_and_Flip, self).__init__(epsilon)
        self.dim = 3
        self.names = ['Tx', 'Ty', 'Flip']
        self.pad = Pad(self.eps.copy(), fill=0, padding_mode='constant')
        self.eps.append(1)
        self.indexes = []
    
    def clamp_delta(self, delta, imgs):
        """Clamp delta."""
        for i in range(2):
            delta[:, i] = torch.clamp(delta[:, i], -self.eps[i], self.eps[i]).to(dtype=torch.int64)
        delta[:, 2] = torch.sign(delta[:, 2])
        return delta

    def _perturb(self, imgs, delta):
        delta = delta.to(dtype=torch.int64)
        h, w = imgs.shape[-2:]
        perturbed_imgs = torch.empty_like(imgs)
        padded_imgs = self.pad(imgs)
        for i in range(imgs.shape[0]):
            dh, dw = self.eps[0]+delta[i,0], self.eps[1]+delta[i,1]
            perturbed_imgs[i] = padded_imgs[i,:,dh:h+dh, dw:w+dw]
        perturbed_imgs[delta[:,2]>0] = torch.flip(perturbed_imgs[delta[:,2]>0], dims=[-1])
        return perturbed_imgs

    def delta_init(self, imgs):
        dims = imgs.shape[2:]
        delta_init = torch.empty(imgs.shape[0], self.dim, device=imgs.device, dtype=imgs.dtype)
        for i in range(2):
            delta_init[:,i] =  torch.round((2*self.eps[i])*(torch.rand(imgs.shape[0], device = imgs.device, dtype=imgs.dtype)-0.5))
        delta_init[:,2] = torch.sign(torch.randn(imgs.shape[0], device = imgs.device, dtype=imgs.dtype))
        return delta_init

'''

class Crop_and_Flip(Crop):
    def __init__(self, epsilon):
        super(Crop_and_Flip, self).__init__(epsilon)
        self.dim = 3
        self.names = ['Tx', 'Ty', 'Flip']
        self.pad = Pad(self.eps.copy(), fill=0, padding_mode='constant')
        self.eps.append(1)
        self.indexes = []
    
    def clamp_delta(self, delta, imgs):
        """Clamp delta."""
        for i in range(2):
            delta[:, i] = torch.clamp(delta[:, i], -self.eps[i], self.eps[i])
        delta[:, 2] = torch.sign(delta[:, 2])
        return delta

    def _perturb(self, imgs, delta):
        h, w = imgs.shape[-2:]
        delta_cp = delta.clone()
        delta_cp[:, 0] = delta[:, 0]*delta[:, 2]
        grid = self.crop_grid(imgs, delta_cp[:,:2])
        perturbed_imgs = torch.nn.functional.grid_sample(imgs, grid.to(dtype=imgs.dtype))
        return perturbed_imgs.to(dtype=imgs.dtype, device=imgs.device)

    def delta_init(self, imgs):
        dims = imgs.shape[2:]
        delta_init = torch.empty(imgs.shape[0], self.dim, device=imgs.device, dtype=imgs.dtype)
        for i in range(2):
            delta_init[:,i] =  (2*self.eps[i])*(torch.rand(imgs.shape[0], device = imgs.device, dtype=imgs.dtype)-0.5)
        delta_init[:,2] = torch.sign(torch.randn(imgs.shape[0], device = imgs.device, dtype=imgs.dtype))
        self.grid = self.build_grid(imgs.shape[2], imgs.shape[2]).repeat(imgs.size(0),1,1,1).to(device=imgs.device)
        return delta_init

'''
class Crop(Perturbation):
    def __init__(self, epsilon):
        # Epsilon stores padding
        super(Crop, self).__init__([epsilon, epsilon])
        self.dim = 2
        self.names = ['Tx', 'Ty']
        self.pad = Pad(self.eps, fill=0, padding_mode='constant')
        #self.eps = [2*e for e in self.eps]
        self.indexes = []
    
    def clamp_delta(self, delta, imgs):
        """Clamp delta."""
        for i in range(self.dim):
            delta[:, i] = torch.clamp(torch.round(delta[:, i]), -self.eps[i], self.eps[i]).to(dtype=torch.int64)
        return delta

    def _perturb(self, imgs, delta):
        delta = delta.to(dtype=torch.int64)
        h, w = imgs.shape[-2:]
        perturbed_imgs = torch.empty_like(imgs)
        padded_imgs = self.pad(imgs)
        dh, dw = self.eps[0]+delta[:,0], self.eps[1]+delta[:,1]
        source_box = bbox_generator(dw, dh, torch.tensor(w, device=imgs.device), torch.tensor(h, device=imgs.device))
        perturbed_imgs = crop_by_boxes(padded_imgs, source_box, self.destination_box)   
        return perturbed_imgs

    def delta_init(self, imgs):
        delta_init = torch.empty(imgs.shape[0], self.dim, device=imgs.device, dtype=imgs.dtype)
        for i in range(self.dim):
            delta_init[:,i] =  self.eps[i]*2*(torch.rand(imgs.shape[0], device = imgs.device, dtype=imgs.dtype)-0.5)
        self.destination_box = (bbox_generator(torch.tensor(0), torch.tensor(0), torch.tensor(imgs.shape[-2]), torch.tensor(imgs.shape[-1]))).expand(imgs.shape[0], -1,-1)
        #self.grid = []
        #for i in range(2):
            #self.grid.append(torch.arange(imgs.shape[-2-i]+self.eps[i], device=imgs.device))       
        return delta_init
    

class Crop_and_Flip(Crop):
    def __init__(self, epsilon):
        super(Crop_and_Flip, self).__init__(epsilon)
        self.dim = 3
        self.names = ['Tx', 'Ty', 'Flip']
        self.pad = Pad(self.eps.copy(), fill=0, padding_mode='constant')
        self.eps.append(1)
        self.indexes = []
    
    def clamp_delta(self, delta, imgs):
        """Clamp delta."""
        for i in range(2):
            delta[:, i] = torch.clamp(delta[:, i], -self.eps[i], self.eps[i]).to(dtype=torch.int64)
        delta[:, 2] = torch.sign(delta[:, 2])
        return delta

    def _perturb(self, imgs, delta):
        delta = delta.to(dtype=torch.int64)
        h, w = imgs.shape[-2:]
        perturbed_imgs = torch.empty_like(imgs)
        padded_imgs = self.pad(imgs)
        for i in range(imgs.shape[0]):
            dh, dw = self.eps[0]+delta[i,0], self.eps[1]+delta[i,1]
            perturbed_imgs[i] = padded_imgs[i,:,dh:h+dh, dw:w+dw]
        perturbed_imgs[delta[:,2]>0] = torch.flip(perturbed_imgs[delta[:,2]>0], dims=[-1])
        return perturbed_imgs

    def delta_init(self, imgs):
        dims = imgs.shape[2:]
        delta_init = torch.empty(imgs.shape[0], self.dim, device=imgs.device, dtype=imgs.dtype)
        for i in range(2):
            delta_init[:,i] =  torch.round((2*self.eps[i])*(torch.rand(imgs.shape[0], device = imgs.device, dtype=imgs.dtype)-0.5))
        delta_init[:,2] = torch.sign(torch.randn(imgs.shape[0], device = imgs.device, dtype=imgs.dtype))
        return delta_init
'''

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

class PointcloudTranslation(Translation):
    def __init__(self, epsilon):
        super(PointcloudTranslation, self).__init__(epsilon)
        self.dim = 6
        self.names = [ 'xyz1', 'xyz2']

    def _perturb(self, points, delta):
        return translate_pointcloud(points, delta)
    def delta_init(self, imgs):
        delta_init = torch.empty(imgs.shape[0], self.dim, device=imgs.device, dtype=imgs.dtype)
        for i in range(self.dim):
            eps = self.eps[i]
            delta_init[:,i] =   2*eps* (torch.rand(imgs.shape[0], device = imgs.device, dtype=imgs.dtype)-0.5)
        return delta_init