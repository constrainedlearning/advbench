import os, sys
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.optim import Adam
from einops import rearrange, reduce, repeat

from advbench import perturbations
from advbench.lib.manifool.functions.algorithms.manifool import manifool
from advbench.datasets import FFCV_AVAILABLE

torch.backends.cudnn.benchmark = True

class Attack(object):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(Attack, self).__init__()
        self.classifier = classifier
        self.hparams = hparams
        self.device = device
        eps = self.hparams['epsilon']
        self.perturbation_name = perturbation
        self.perturbation = vars(perturbations)[perturbation](eps)
    
    def __call__(self, imgs, labels):
        return self.forward(imgs, labels)
        
    def forward(self, imgs, labels):
        if self.classifier.training:
            training = True
            self.classifier.eval()
        else:
            training = False
        attack_tuple = self.attack(imgs, labels)
        if training:
            self.classifier.train()
        return attack_tuple
    
    def attack(self, imgs, labels):
        raise NotImplementedError

class Attack_Linf(Attack):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(Attack_Linf, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        if isinstance(self.perturbation.eps, torch.Tensor):
                self.perturbation.eps.to(device)
        if isinstance(self.perturbation.eps, list):
            eps = torch.tensor(self.perturbation.eps).to(device)
        else:
            eps = self.perturbation.eps
        self.step = (eps*self.hparams['pgd_step_size'])
        self.batched = hparams['batched']

class Fo(Attack_Linf):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(Fo, self).__init__(classifier, hparams, device, perturbation=perturbation)
        
    def attack(self, imgs, labels):
        highest_loss = torch.zeros(imgs.shape[0], device = imgs.device)
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        worst_delta = torch.empty_like(delta)
        for _ in range(self.hparams['fo_restarts']):
            delta = self.perturbation.delta_init(imgs).to(imgs.device)
            delta, adv_loss = self.optimize_delta(imgs, labels, delta)
            worst_delta[adv_loss>highest_loss] = delta[adv_loss>highest_loss]
            highest_loss[adv_loss>highest_loss] = adv_loss[adv_loss>highest_loss]
        adv_imgs = self.perturbation.perturb_img(imgs, worst_delta)
        return adv_imgs.detach(), worst_delta.detach()

class Fo_PGD(Fo):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(Fo_PGD, self).__init__(classifier, hparams, device, perturbation=perturbation)
    
    def optimize_delta(self, imgs, labels, delta): 
        for _ in range(self.hparams['pgd_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = self.classifier.loss(self.classifier(adv_imgs), labels, reduction='none')
                mean = adv_loss.mean()
            grad = torch.autograd.grad(mean, [delta])[0].detach()
            delta.requires_grad_(False)
            delta += self.step*torch.sign(grad)
            delta = self.perturbation.clamp_delta(delta, imgs)
        return delta, adv_loss   # this detach may not be necessary

class Fo_Adam(Fo):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(Fo_Adam, self).__init__(classifier, hparams, device, perturbation=perturbation)
    
    def optimize_delta(self, imgs, labels, delta):
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        opt = Adam([delta], lr = self.hparams['fo_adam_step_size'], betas = (0.9, 0.999))
        for _ in range(self.hparams['fo_n_steps']):
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = self.classifier.loss(self.classifier(adv_imgs), labels, reduction='none')
                mean = adv_loss.mean()
            mean.backward()
            opt.step()
            delta = self.perturbation.clamp_delta(delta, imgs)
        return delta, adv_loss

class Fo_SGD(Fo):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Fo_SGD, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        if isinstance(self.perturbation.eps, torch.Tensor):
                self.perturbation.eps.to(device)
        if isinstance(self.perturbation.eps, list):
            eps = torch.tensor(self.perturbation.eps).to(device)
        else:
            eps = self.perturbation.eps
        self.step = (eps*self.hparams['fo_sgd_step_size'])
    def optimize_delta(self, imgs, labels, delta):
        batch_size = imgs.size(0)
        velocity=0
        for t in range(self.hparams['fo_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = self.classifier.loss(self.classifier(adv_imgs), labels, reduction='none')
                mean = - adv_loss.mean()
            torch.nn.utils.clip_grad_norm_(delta, 1, norm_type=2.0)
            grad = torch.autograd.grad(mean, [delta])[0].detach()
            delta.requires_grad_(False)
            grad = torch.clip(grad, min=-1, max=1)
            velocity = self.hparams['fo_sgd_momentum']*velocity+grad
            if t<self.hparams['fo_n_steps']-1:     
                delta += self.step*velocity
                delta = self.perturbation.clamp_delta(delta, imgs)
        return delta.detach(), adv_loss

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
    def attack(self, imgs, labels):
        batch_size = imgs.size(0)
        noise_dist = Laplace(torch.tensor(0.), torch.tensor(1.))
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        for _ in range(self.hparams['l_dale_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = self.classifier.loss(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            noise = noise_dist.sample(grad.shape).to(self.device)
            delta += self.step * torch.sign(grad) + self.noise_coeff * noise
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
        return adv_imgs.detach(), delta.detach()

class MH(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf', acceptance_meter=None):
        super(MH, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        if self.hparams['mh_proposal']=='Laplace':
            if isinstance(self.perturbation.eps, list):
                eps = torch.tensor(self.perturbation.eps).to(device)
            else:
                eps = self.perturbation.eps
            if isinstance(eps, torch.Tensor):
                eps = eps.to(device)
            self.noise_dist = Laplace(torch.zeros(self.perturbation.dim, device=device), self.hparams['mh_dale_scale']*eps)
            self.eps = eps
        else:
            raise NotImplementedError
        self.get_proposal = lambda x: x + self.noise_dist.sample([x.shape[0]]).to(x.device)
        if acceptance_meter is not None:
            self.log_acceptance=True
            self.acceptance_meter = acceptance_meter
        else:
            self.log_acceptance = False

    def attack(self, imgs, labels):
        with torch.no_grad():
            delta = self.perturbation.delta_init(imgs).to(imgs.device)
            delta = self.perturbation.clamp_delta(delta, imgs)
            adv_imgs = self.perturbation.perturb_img(imgs, delta)
            last_loss = self.classifier.loss(self.classifier(adv_imgs), labels)
            ones = torch.ones_like(last_loss)
            noise_dist = Laplace(torch.zeros(delta.shape[1:], device=imgs.device), self.hparams['mh_dale_scale']*self.eps)
            for _ in range(self.hparams['mh_dale_n_steps']):
                proposal = delta +  noise_dist.sample([delta.shape[0]])*self.eps
                if torch.allclose(proposal, self.perturbation.clamp_delta(proposal, adv_imgs)):
                    adv_imgs = self.perturbation.perturb_img(imgs, proposal)
                    proposal_loss = self.classifier.loss(self.classifier(adv_imgs), labels)
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
            grids.append(torch.arange(-eps, eps, step=step, device=self.device))
        self.grid = torch.cartesian_prod(*grids)

    def attack(self, imgs, labels):
        batch_size = imgs.size(0)
        with torch.no_grad():
            if len(imgs.shape) == 4:
                repeated_imgs = repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=self.grid_size)
            elif len(imgs.shape) == 3:
                repeated_imgs = repeat(imgs, 'B C P -> (B S) C P', B=batch_size, S=self.grid_size)
            else:
                raise NotImplementedError
            adv_imgs = self.perturbation.perturb_img(
                repeated_imgs,
                repeat(self.grid, 'S D -> (B S) D', B=batch_size, D=self.dim, S=self.grid_size))
            if len(labels.shape) == 1:
                repeated_labels = repeat(labels, 'B -> (B S)', S=self.grid_size)
            else:
                repeated_labels = repeat(labels, 'B D -> (B S) D', S=self.grid_size)
            adv_loss = self.classifier.loss(self.classifier(adv_imgs), repeated_labels, reduction="none")
        adv_loss = rearrange(adv_loss, '(B S) -> B S', B=batch_size, S=self.grid_size)
        max_idx = torch.argmax(adv_loss,dim=-1)
        delta = self.grid[max_idx]
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
        return adv_imgs.detach(), delta.detach()

class Worst_Of_K(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Worst_Of_K, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def attack(self, imgs, labels):
        with torch.no_grad():
            batch_size = imgs.size(0)
            delta = self.perturbation.delta_init(imgs)
            steps = self.hparams['worst_of_k_steps']
            if self.batched:
                with torch.no_grad():
                    if len(imgs.shape) == 4:
                        repeated_images = repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=steps)
                    elif len(imgs.shape) == 3:
                        repeated_images = repeat(imgs, 'B C P -> (B S) C P', B=batch_size, S=steps)
                    else:
                        raise NotImplementedError
                    delta = self.perturbation.delta_init(repeated_images).to(imgs.device)
                    delta = self.perturbation.clamp_delta(delta, repeated_images)
                    adv_imgs = self.perturbation.perturb_img(repeated_images, delta)
                    if len(labels.shape) == 1:
                        new_labels = repeat(labels, 'B -> (B S)', S=steps)
                    else:
                        new_labels = repeat(labels, 'B D -> (B S) D', S=steps)
                    adv_loss = self.classifier.loss(self.classifier(adv_imgs), new_labels, reduction="none")
                    adv_loss = rearrange(adv_loss, '(B S) -> B S', B=batch_size, S=steps)
                    max_idx = torch.argmax(adv_loss, dim=-1)
                    delta = delta[max_idx]
            else:
                worst_loss = -1
                with torch.no_grad():
                    for _ in range(steps):
                        delta = self.perturbation.delta_init(imgs).to(imgs.device)
                        delta = self.perturbation.clamp_delta(delta, imgs)
                        adv_imgs = self.perturbation.perturb_img(imgs, delta)
                        adv_loss = self.classifier.loss(self.classifier(adv_imgs), labels)
                        if adv_loss>worst_loss:
                            worst_loss = adv_loss
                            worst_delta = delta
                    delta = worst_delta
            adv_imgs = self.perturbation.perturb_img(imgs, delta)
            return adv_imgs.detach(), delta.detach()

class Rand_Aug(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Rand_Aug, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def attack(self, imgs, labels):
        with torch.no_grad():
            batch_size = imgs.size(0)
            delta = self.perturbation.delta_init(imgs).to(imgs.device)
            delta = self.sample(delta)
            delta = self.perturbation.clamp_delta(delta, imgs)
            adv_imgs = self.perturbation.perturb_img(imgs, delta)
        return adv_imgs.detach(), delta.detach()
    
    def sample(self, delta):
        return delta

class Gaussian_aug(Rand_Aug):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Gaussian_aug, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        self.sigma = hparams["gaussian_attack_std"]*self.perturbation.eps

    def sample(self, delta):
        return torch.randn_like(delta)*self.sigma

class Laplace_aug(Rand_Aug):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Laplace_aug, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        self.scale = hparams["laplacian_attack_std"]*self.perturbation.eps/sqrt(2)

    def sample(self, delta):
        return Laplace(torch.zeros_like(delta), self.scale).sample().to(device=delta.device, dtype=delta.dtype)

class Beta_aug(Rand_Aug):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Beta_aug, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        self.alpha = hparams["beta_attack_alpha"]
        self.beta= hparams["beta_attack_beta"]

    def sample(self, delta):
        ones = torch.ones_like(delta)
        return Beta(ones*self.alpha, ones*self.beta).sample()#.to(device=delta.device, dtype=delta.dtype)

class Rand_Aug_Batch(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Rand_Aug_Batch, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def attack(self, imgs, labels):
        with torch.no_grad():
            batch_size = imgs.size(0)
            if len(imgs.shape) == 4:
                repeated_images = repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=self.hparams['perturbation_batch_size'])
            elif len(imgs.shape) == 3:
                repeated_images = repeat(imgs, 'B C P -> (B S) C P', B=batch_size, S=self.hparams['perturbation_batch_size'])
            else:
                raise NotImplementedError
            delta = self.perturbation.delta_init(repeated_images).to(imgs.device)
            delta = self.perturbation.clamp_delta(delta, repeated_images)
            adv_imgs = self.perturbation.perturb_img(repeated_images, delta)
            if len(labels.shape) == 1:
                new_labels = repeat(labels, 'B -> (B S)', S=self.hparams['perturbation_batch_size'])
            else:
                new_labels = repeat(labels, 'B D -> (B S) D', S=self.hparams['perturbation_batch_size'])
            return adv_imgs.detach(), delta.detach(), new_labels.detach()

class Dist_Batch(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Dist_Batch, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def sample(self, delta):
        raise NotImplementedError

    def attack(self, imgs, labels):
        with torch.no_grad():
            batch_size = imgs.size(0)
            if len(imgs.shape) == 4:
                repeated_images = repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=self.hparams['perturbation_batch_size'])
            elif len(imgs.shape) == 3:
                repeated_images = repeat(imgs, 'B C P -> (B S) C P', B=batch_size, S=self.hparams['perturbation_batch_size'])
            else:
                raise NotImplementedError
            delta = self.perturbation.delta_init(repeated_images).to(imgs.device)
            delta = self.sample(delta)
            delta = self.perturbation.clamp_delta(delta, repeated_images)
            adv_imgs = self.perturbation.perturb_img(repeated_images, delta)
            if len(labels.shape) == 1:
                new_labels = repeat(labels, 'B -> (B S)', S=self.hparams['perturbation_batch_size'])
            else:
                new_labels = repeat(labels, 'B D -> (B S) D', S=self.hparams['perturbation_batch_size'])
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
        return Laplace(torch.zeros_like(delta), self.scale).sample().to(device=delta.device, dtype=delta.dtype)
    

class Grid_Batch(Grid_Search):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Grid_Batch, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def attack(self, imgs, labels):
        with torch.no_grad():
            batch_size = imgs.size(0)
            rep_grid = repeat(self.grid, 'S D -> (B S) D', B=batch_size, D=self.dim, S=self.grid_size)
            if len(imgs.shape) == 4:
                rep_imgs = repeat(imgs, 'B W H C -> (B S) W H C', B=batch_size, S=self.grid_size)
            elif len(imgs.shape) == 3:
                rep_imgs = repeat(imgs, 'B C P -> (B S) C P', B=batch_size, S=self.grid_size)
            else:
                raise NotImplementedError
            delta = self.perturbation.clamp_delta(rep_grid, rep_imgs)
            adv_imgs = self.perturbation.perturb_img(
                rep_imgs,
                rep_grid)
            if len(labels.shape) == 1:
                new_labels = repeat(labels, 'B -> (B S)', S=self.hparams['perturbation_batch_size'])
            else:
                new_labels = repeat(labels, 'B D -> (B S) D', S=self.hparams['perturbation_batch_size'])
        
            return adv_imgs.detach(), delta, new_labels.detach()
