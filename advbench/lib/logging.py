from selectors import EpollSelector
from scipy import ma
import torch
from advbench import attacks
from einops import repeat, rearrange
import torch.nn.functional as F

class PerturbationEval():
    def __init__(self, algorithm, loader, max_perturbations=None):
        self.algorithm = algorithm
        self.classifier = self.algorithm.classifier
        self.hparams = self.algorithm.hparams
        self.device = self.algorithm.device
        self.loader = loader
        self.max_perturbations = max_perturbations
        self.perturbation = self.algorithm.attack.perturbation
        self.dim = self.perturbation.dim
        
    def eval_perturbed(self, single_img=False, batches=1):
        self.grid = self.get_grid()
        self.grid_size = self.grid.shape[0]
        self.algorithm.classifier.eval()
        adv_losses = []
        with torch.no_grad():
            if single_img:
                    imgs, labels = self.loader.dataset[0]
                    imgs, labels = imgs.unsqueeze(0).to(self.device), torch.tensor([labels]).to(self.device)
                    adv_losses = self.step(imgs, labels)[0]
            else:
                for idx, batch in enumerate(self.loader):
                    if idx < batches:
                        imgs, labels = batch
                        imgs, labels = imgs.to(self.device), labels.to(self.device)
                        adv_losses.append(self.step(imgs, labels))
                    else:
                        break
        self.algorithm.classifier.train()
        self.loader.shuffle = True
        if batches>1 or not single_img:
            adv_losses = torch.concat(adv_losses, dim=0).mean(dim=0)
        return self.grid, adv_losses
    
    def step(self, imgs, labels):
        batch_size = imgs.shape[0]
        adv_imgs = self.perturbation.perturb_img(
            repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=self.grid_size),
            repeat(self.grid, 'S D -> (B S) D', B=batch_size, D=self.dim, S=self.grid_size))
        adv_loss = F.cross_entropy(self.classifier(adv_imgs), repeat(labels, 'B -> (B S)', S=self.grid_size), reduction="none")
        adv_loss = rearrange(adv_loss, '(B S) -> B S', B=batch_size, S=self.grid_size)
        return adv_loss

    def get_grid(self):
        pass


class GridEval(PerturbationEval):
    def __init__(self,algorithm, loader, max_perturbations=None):
        super(GridEval, self).__init__(algorithm, loader, max_perturbations=max_perturbations)
        self.attack = attacks.Grid_Search(algorithm.classifier, algorithm.hparams, algorithm.device, perturbation=algorithm.perturbation_name, grid_size=max_perturbations)
    def get_grid(self):
        return self.attack.grid