import os, sys
from math import sqrt
try:
    import hamiltorch
    HAMILTORCH_AVAILABLE = True
except ImportError:
    HAMILTORCH_AVAILABLE = False
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.laplace import Laplace
from einops import rearrange, reduce, repeat

from advbench import perturbations

class Attack(nn.Module):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(Attack, self).__init__()
        self.classifier = classifier
        self.hparams = hparams
        self.device = device
        eps = self.hparams['epsilon']
        self.perturbation = vars(perturbations)[perturbation](eps)
    def forward(self, imgs, labels):
        raise NotImplementedError

class Attack_Linf(Attack):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(Attack_Linf, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
    
class PGD_Linf(Attack_Linf):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(PGD_Linf, self).__init__(classifier, hparams, device, perturbation=perturbation)
    
    def forward(self, imgs, labels):
        self.classifier.eval()
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        for _ in range(self.hparams['pgd_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            delta += self.hparams['pgd_step_size']* torch.sign(grad)
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
            
        self.classifier.train()
        return adv_imgs.detach(), delta.detach()    # this detach may not be necessary

class TRADES_Linf(Attack_Linf):
    def __init__(self, classifier, hparams,  device, perturbation='Linf'):
        super(TRADES_Linf, self).__init__(classifier, hparams, device, perturbation=perturbation)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')  # AR: let's write a method to do the log-softmax part

    def forward(self, imgs, labels):
        self.classifier.eval()
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        for _ in range(self.hparams['trades_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = self.kl_loss_fn(
                    F.log_softmax(self.classifier(adv_imgs), dim=1),   # AR: Note that this means that we can't have softmax at output of classifier
                    F.softmax(self.classifier(imgs), dim=1))
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            delta += self.hparams['trades_step_size']* torch.sign(grad)
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
        
        self.classifier.train()
        return adv_imgs.detach(), delta.detach() # this detach may not be necessary

class FGSM_Linf(Attack):
    def __init__(self, classifier,  hparams, device,  perturbation='Linf'):
        super(FGSM_Linf, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def forward(self, imgs, labels):
        self.classifier.eval()

        imgs.requires_grad = True
        adv_loss = F.cross_entropy(self.classifier(imgs), labels)
        grad = torch.autograd.grad(adv_loss, [imgs])[0].detach()
        delta = self.hparams['epsilon'] * grad.sign()
        delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()

        return adv_imgs.detach(), delta.detach()

class LMC_Gaussian_Linf(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(LMC_Gaussian_Linf, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        #self.step = self.perturbation.eps*self.hparams['g_dale_step_size']
        #self.noise_coeff = self.perturbation.eps*self.hparams['g_dale_noise_coeff']
    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        for _ in range(self.hparams['g_dale_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            noise = torch.randn_like(delta).to(self.device).detach()
            delta += self.step.to(self.device) * grad + self.noise_coeff.to(self.device) * noise
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()

        return adv_imgs.detach(), delta.detach()

class LMC_Laplacian_Linf(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(LMC_Laplacian_Linf, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        if isinstance(self.perturbation.eps, torch.Tensor):
                self.perturbation.eps.to(device)
        if isinstance(self.perturbation.eps, list):
            eps = torch.tensor(self.perturbation.eps).to(device)
        else:
            eps = self.perturbation.eps
        self.step = (eps*self.hparams['l_dale_step_size'])
        if isinstance(self.step, torch.Tensor):
                self.step = self.step.to(device)
        self.noise_coeff = (eps*self.hparams['l_dale_noise_coeff'])
    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        noise_dist = Laplace(torch.tensor(0.), torch.tensor(1.))
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        for _ in range(self.hparams['l_dale_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            noise = noise_dist.sample(grad.shape).to(self.device)
            delta += self.step * torch.sign(grad) + self.noise_coeff * noise
            #delta += self.hparams['l_dale_step_size']* torch.sign(grad) + self.hparams['l_dale_noise_coeff']* noise
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()
        return adv_imgs.detach(), delta.detach()
class MCMC(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf', acceptance_meter=None):
        super(MCMC, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        if self.hparams['mcmc_proposal']=='Laplace':
            if isinstance(self.perturbation.eps, list):
                eps = torch.tensor(self.perturbation.eps).to(device)
            else:
                eps = self.perturbation.eps
            if isinstance(eps, torch.Tensor):
                eps = eps.to(device)
            self.noise_dist = Laplace(torch.zeros(self.perturbation.dim, device=device), self.hparams['mcmc_dale_scale']*eps)
        else:
            raise NotImplementedError
        self.get_proposal = lambda x: x + self.noise_dist.sample([x.shape[0]]).to(x.device)
        if acceptance_meter is not None:
            self.log_acceptance=True
            self.acceptance_meter = acceptance_meter
        else:
            self.log_acceptance = False

    def forward(self, imgs, labels):
        self.classifier.eval()
        with torch.no_grad():
            delta = self.perturbation.delta_init(imgs).to(imgs.device)
            delta = self.perturbation.clamp_delta(delta, imgs)
            adv_imgs = self.perturbation.perturb_img(imgs, delta)
            last_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            ones = torch.ones_like(last_loss)
            for _ in range(self.hparams['mcmc_dale_n_steps']):
                proposal = self.get_proposal(delta)
                if torch.allclose(delta, self.perturbation.clamp_delta(delta, adv_imgs)):
                    adv_imgs = self.perturbation.perturb_img(imgs, delta)
                    proposal_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
                    acceptance_ratio = (
                        torch.minimum((proposal_loss / last_loss), ones)
                    )
                    if self.log_acceptance:
                        self.acceptance_meter.update(acceptance_ratio.mean().item(), n=1)
                    accepted = torch.bernoulli(acceptance_ratio).bool()
                    delta[accepted] = proposal[accepted].type(delta.dtype)
                    last_loss[accepted] = proposal_loss[accepted]
                elif self.log_acceptance:
                    self.acceptance_meter.update(0, n=1)
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()
        return adv_imgs.detach(), delta.detach()


class Grid_Search(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf', grid_size=None):
        super(Grid_Search, self).__init__(classifier,  hparams, device,  perturbation=perturbation)        
        self.perturbation_name = perturbation
        self.dim = self.perturbation.dim
        if grid_size is None:
            self.grid_size = self.hparams['grid_size']
        else:
            self.grid_size = grid_size
        self.grid_steps = int(self.grid_size**(1/self.dim))
        self.grid_size = self.grid_steps**self.dim
        self.grid_shape = [self.grid_size, self.dim]
        self.epsilon = self.hparams['epsilon']
        self.make_grid()
    
    def make_grid(self):
        grids = []
        for idx in range(self.dim):
            if isinstance(self.epsilon, float) or isinstance(self.epsilon, int):
                eps = self.epsilon
            else:
                eps = self.epsilon[idx]
                
            step = 2*eps/self.grid_steps
            if self.perturbation_name == 'CPAB':
                grids.append(torch.arange(0, eps, step=step, device=self.device))
            else:   
                grids.append(torch.arange(-eps, eps, step=step, device=self.device))
        self.grid = torch.cartesian_prod(*grids)

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        with torch.no_grad():
            adv_imgs = self.perturbation.perturb_img(
                repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=self.grid_size),
                repeat(self.grid, 'S D -> (B S) D', B=batch_size, D=self.dim, S=self.grid_size))
            adv_loss = F.cross_entropy(self.classifier(adv_imgs), repeat(labels, 'B -> (B S)', S=self.grid_size), reduction="none")
        adv_loss = rearrange(adv_loss, '(B S) -> B S', B=batch_size, S=self.grid_size)
        max_idx = torch.argmax(adv_loss,dim=-1)
        delta = self.grid[max_idx]
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
        self.classifier.train()
        return adv_imgs.detach(), delta.detach()

class Worst_Of_K(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Worst_Of_K, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        delta = self.perturbation.delta_init(imgs)
        steps = self.hparams['worst_of_k_steps']
        with torch.no_grad():
            repeated_images = repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=steps)
            delta = self.perturbation.delta_init(repeated_images).to(imgs.device)
            delta = self.perturbation.clamp_delta(delta, repeated_images)
            adv_imgs = self.perturbation.perturb_img(repeated_images, delta)
            adv_loss = F.cross_entropy(self.classifier(adv_imgs), repeat(labels, 'B -> (B S)', S=steps), reduction="none")
            adv_loss = rearrange(adv_loss, '(B S) -> B S', B=batch_size, S=steps)
            max_idx = torch.argmax(adv_loss, dim=-1)
            delta = delta[max_idx]
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
        self.classifier.train()
        return adv_imgs.detach(), delta.detach()

class Rand_Aug(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Rand_Aug, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def forward(self, imgs, labels):
        batch_size = imgs.size(0)
        self.classifier.eval()
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
        self.classifier.train()
        return adv_imgs.detach(), delta.detach()

class Rand_Aug_Batch(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Rand_Aug_Batch, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def forward(self, imgs, labels):
        batch_size = imgs.size(0)
        repeated_images = repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=self.hparams['perturbation_batch_size'])
        delta = self.perturbation.delta_init(repeated_images).to(imgs.device)
        delta = self.perturbation.clamp_delta(delta, repeated_images)
        adv_imgs = self.perturbation.perturb_img(repeated_images, delta)
        new_labels = repeat(labels, 'B -> (B S)', S=self.hparams['perturbation_batch_size'])
        return adv_imgs.detach(), delta.detach(), new_labels.detach()

class Dist_Batch(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Dist_Batch, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def sample(self, delta):
        raise NotImplementedError

    def forward(self, imgs, labels):
        batch_size = imgs.size(0)
        repeated_images = repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=self.hparams['perturbation_batch_size'])
        delta = self.perturbation.delta_init(repeated_images).to(imgs.device)
        delta = self.sample(delta)
        delta = self.perturbation.clamp_delta(delta, repeated_images)
        adv_imgs = self.perturbation.perturb_img(repeated_images, delta)
        new_labels = repeat(labels, 'B -> (B S)', S=self.hparams['perturbation_batch_size'])
        return adv_imgs.detach(), delta.detach(), new_labels.detach()
    
class Gaussian_Batch(Dist_Batch):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Gaussian_Batch, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        self.sigma = hparams["gaussian_attack_std"]*self.perturbation.eps

    def sample(self, delta):
        return torch.randn_like(delta)*self.sigma

class Laplacian_Batch(Dist_Batch):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Laplacian_Batch, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        self.scale = hparams["laplacian_attack_std"]*self.perturbation.eps/sqrt(2)

    def sample(self, delta):
        return torch.distributions.laplace.Laplace(torch.zeros_like(delta), self.scale).sample().to(device=delta.device, dtype=delta.dtype)
    

class Grid_Batch(Grid_Search):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Grid_Batch, self).__init__(classifier,  hparams, device,  perturbation=perturbation, grid_size = hparams['perturbation_batch_size'])

    def forward(self, imgs, labels):
        batch_size = imgs.size(0)
        rep_grid = repeat(self.grid, 'S D -> (B S) D', B=batch_size, D=self.dim, S=self.grid_size)
        rep_imgs = repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=self.grid_size)
        delta = self.perturbation.clamp_delta(rep_grid, rep_imgs)
        adv_imgs = self.perturbation.perturb_img(
                rep_imgs,
                rep_grid)
        new_labels = repeat(labels, 'B -> (B S)', S=self.grid_size)
        
        return adv_imgs.detach(), delta, new_labels.detach()

if HAMILTORCH_AVAILABLE:
    class NUTS(Attack_Linf):
        def __init__(self, classifier,  hparams, device, perturbation='Linf'):
            super(NUTS, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
            self.infty = 10e8 #torch.tensor(float('inf')).to(device)
            self.burn = self.hparams['n_burn']
            self.eps = hparams['epsilon']

        def forward(self, imgs, labels):
            self.classifier.eval()
            batch_size = imgs.size(0)
            total_size = 1
            img_dims = tuple(d for d in range(1,imgs.dim()))
            for i in imgs.size():
                total_size = total_size*i
            params_init = 0.001*torch.rand(total_size).to(self.device)
            def log_prob(delta):
                delta = delta.reshape(imgs.shape)
                adv_imgs = imgs+torch.clamp(delta, min=-self.eps, max=self.eps)
                loss = 1 - torch.softmax(self.classifier(adv_imgs), dim=1)[range(batch_size), labels]
                log_loss = torch.log(loss)
                #log_loss[torch.amax(torch.abs(delta),img_dims)>self.eps] = - self.infty
                return log_loss.sum()
            self.blockPrint()
            delta = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init,
                                    num_samples=self.burn+self.hparams['n_dale_n_steps'],
                                    step_size=self.hparams['n_dale_step_size'],
                                    burn = self.burn,
                                    num_steps_per_sample=7,
                                    desired_accept_rate=0.8)[-1]
            self.enablePrint()
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            adv_imgs = imgs + delta.reshape(imgs.shape)
            self.classifier.train()
            return adv_imgs.detach(), delta.detach()
        # Disable
        def blockPrint(self):
            sys.stdout = open(os.devnull, 'w')

        # Restore
        def enablePrint(self):
            sys.stdout = sys.__stdout__

