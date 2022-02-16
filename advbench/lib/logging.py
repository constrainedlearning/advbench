from selectors import EpollSelector
from scipy import ma
import torch
from torch.cuda.amp import autocast
from advbench import attacks
from einops import repeat, rearrange
import torch.nn.functional as F
from tqdm import tqdm

class PerturbationEval():
    def __init__(self, algorithm, loader, max_perturbations=None, batched=True):
        self.algorithm = algorithm
        self.classifier = self.algorithm.classifier
        self.hparams = self.algorithm.hparams
        self.device = self.algorithm.device
        self.loader = loader
        self.max_perturbations = max_perturbations
        self.perturbation = self.algorithm.attack.perturbation
        self.dim = self.perturbation.dim
        self.batched = batched
        
    def eval_perturbed(self, single_img=False, batches=1):
        self.grid = self.get_grid()
        self.grid_size = self.grid.shape[0]
        self.algorithm.classifier.eval()
        self.algorithm.export()
        adv_losses = []
        with torch.no_grad():
            if single_img:
                    imgs, labels = self.loader.dataset[0]
                    imgs, labels = imgs.unsqueeze(0).to(self.device), torch.tensor([labels]).to(self.device)
                    with autocast():
                        adv_losses = self.step(imgs, labels)[0]
            else:
                for idx, batch in tqdm(enumerate(self.loader)):
                    if idx < batches:
                        imgs, labels = batch
                        imgs, labels = imgs.to(self.device), labels.to(self.device)
                        with autocast():
                            adv_losses.append(self.step(imgs, labels))
                    else:
                        break
        self.algorithm.unexport()
        self.algorithm.classifier.train()
        self.loader.shuffle = True
        if batches>1 or not single_img:
            adv_losses = torch.concat(adv_losses, dim=0).mean(dim=0)
        return self.grid, adv_losses
    
    def step(self, imgs, labels):
        batch_size = imgs.shape[0]
        if self.batched:
            adv_imgs = self.perturbation.perturb_img(
                repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=self.grid_size),
                repeat(self.grid, 'S D -> (B S) D', B=batch_size, D=self.dim, S=self.grid_size))
            adv_loss = F.cross_entropy(self.classifier(adv_imgs), repeat(labels, 'B -> (B S)', S=self.grid_size), reduction="none")
            adv_loss = rearrange(adv_loss, '(B S) -> B S', B=batch_size, S=self.grid_size)
        else:
            adv_loss = torch.empty((batch_size, self.grid_size), device=imgs.device)
            for s in range(self.grid_size):
                grid = repeat(self.grid[s], 'D -> B D', B=batch_size, D=self.dim)
                adv_imgs = self.perturbation.perturb_img(imgs, grid)
                angle_loss = F.cross_entropy(self.classifier(adv_imgs), labels, reduction="none")
                adv_loss[:, s] = angle_loss
        return adv_loss

    def get_grid(self):
        pass


class GridEval(PerturbationEval):
    def __init__(self,algorithm, loader, max_perturbations=None):
        super(GridEval, self).__init__(algorithm, loader, max_perturbations=max_perturbations)
        self.attack = attacks.Grid_Search(algorithm.classifier, algorithm.hparams, algorithm.device, perturbation=algorithm.perturbation_name, grid_size=max_perturbations)
    def get_grid(self):
        return self.attack.grid

class AngleGrid(PerturbationEval):
    def __init__(self,algorithm, loader, tx=0, ty=0, max_perturbations=None, batched=False):
        super(AngleGrid, self).__init__(algorithm, loader, max_perturbations=max_perturbations, batched=False)
        self.attack = attacks.Grid_Search(algorithm.classifier, algorithm.hparams, algorithm.device, perturbation="Rotation", grid_size=max_perturbations)
        self.tx = tx
        self.ty = ty
    def get_grid(self):
        angle_grid = self.attack.grid
        ones = torch.ones_like(angle_grid)
        grid = torch.column_stack([angle_grid, self.tx*ones, self.ty*ones])
        return grid