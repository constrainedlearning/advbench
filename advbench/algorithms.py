from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pandas as pd
from numpy.random import binomial
from torch.cuda.amp import GradScaler, autocast
from advbench.datasets import FFCV_AVAILABLE
from torch.nn.utils import clip_grad_norm_

from advbench import attacks, networks, optimizers, perturbations
from advbench.lib import meters

ALGORITHMS = [
    'ERM',
    'PGD',
    'FGSM',
    'TRADES',
    'ALP',
    'CLP',
    'Gaussian_DALE',
    'Laplacian_DALE',
    'Discrete_DALE',
    'Gaussian_DALE_PD',
    'Gaussian_DALE_PD_Reverse',
    'MH_DALE_PD_Reverse',
    'KL_DALE_PD',
    'Worst_Of_K',
    'Augmentation',
    'Batch_Augmentation',
    'Grid_Search',
    'Batch_Grid',
    'Uniform_DALE_PD_Reverse',
    'Worst_DALE_PD_Reverse',
    'PGD_DALE_PD_Reverse'
]

class Algorithm(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.classifier = networks.Classifier(
            input_shape, num_classes, hparams)
        #summary(self.classifier.to(device), input_size=input_shape)
        if hparams['optimizer']=="SGD":
            self.optimizer = optimizers.Optimizer(
             self.classifier, hparams)
        elif hparams['optimizer']=="SFCNN":
            self.optimizer = optimizers.SFCNN_Optimizer(self.classifier, hparams)
        else:
            print("Optimizer not suported")
            raise NotImplementedError
        self.device = device
        
        self.meters = OrderedDict()
        self.meters['loss'] = meters.AverageMeter()
        self.meters_df = None
        self.perturbation_name = perturbation
        if FFCV_AVAILABLE:
            self.scaler = GradScaler()
        
        self.label_smoothing = hparams['label_smoothing']
        if 'clip_grad' in hparams:
            self.clip_grad = hparams['clip_grad']
        else:
            self.clip_grad = False

    def step(self, imgs, labels):
        raise NotImplementedError

    def predict(self, imgs):
        return self.classifier(imgs)

    def reset_meters(self):
        for meter in self.meters.values():
            meter.reset()

    def meters_to_df(self, epoch):
        if self.meters_df is None:
            keys = []
            for key, val in self.meters.items():
                if val.print:
                    keys.append(key)
            columns = ['Epoch'] + keys
            self.meters_df = pd.DataFrame(columns=columns)
            self.meters_df_keys = keys
        metrics = []
        for key in self.meters_df_keys:
            metrics.append(self.meters[key].avg)
        values = [epoch] + metrics
        self.meters_df.loc[len(self.meters_df)] = values
        return self.meters_df
    def export(self):
        pass
    def unexport(self):
        pass

class ERM(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(ERM, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Rand_Aug(self.classifier, self.hparams, device, perturbation=perturbation)
    def step(self, imgs, labels):
        self.optimizer.zero_grad(set_to_none=True)
        if FFCV_AVAILABLE:
            with autocast():
                loss = self.classifier.loss(self.predict(imgs), labels  )
                self.scaler.scale(loss).backward()
                if self.clip_grad:
                    clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            loss = self.classifier.loss(self.predict(imgs), labels  )
            loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
        self.optimizer.step()
        
        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class Adversarial(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Adversarial, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.penalty = hparams["adv_penalty"]

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                adv_imgs, deltas = self.attack(imgs, labels)
                self.optimizer.zero_grad()
                adv_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
                if self.penalty>0:
                    clean_loss = self.classifier.loss(self.predict(imgs), labels  )
                    loss = clean_loss+adv_loss*self.penalty
                else:
                    loss = adv_loss
                self.scaler.scale(loss).backward()
                if self.clip_grad:
                    clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =   self.attack(imgs, labels)
            self.optimizer.zero_grad()
            adv_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
            if self.penalty>0:
                clean_loss = self.classifier.loss(self.predict(imgs), labels  )
                loss = clean_loss+adv_loss*self.penalty
            else:
                loss = adv_loss
            loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
            self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class Adversarial_PGD(Adversarial):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Adversarial_PGD, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Fo_PGD(self.classifier, self.hparams, device, perturbation=perturbation)

class Adversarial_SGD(Adversarial):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Adversarial_SGD, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Fo_SGD(self.classifier, self.hparams, device, perturbation=perturbation)

class Adversarial_Adam(Adversarial):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Adversarial_Adam, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Fo_Adam(self.classifier, self.hparams, device, perturbation=perturbation)

class Adversarial_Smoothed(Adversarial):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Adversarial_Smoothed, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.LMC_Laplacian_Linf(self.classifier, self.hparams, device, perturbation=perturbation)

class Adversarial_Smoothed_MH(Adversarial):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Adversarial_Smoothed_MH, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.MH(self.classifier, self.hparams, device, perturbation=perturbation)

class Adversarial_Worst_Of_K(Adversarial):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Adversarial_Worst_Of_K, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Worst_Of_K(self.classifier, self.hparams, device, perturbation=perturbation)

class Gaussian_DALE(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Gaussian_DALE, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.LMC_Gaussian_Linf(self.classifier, self.hparams, device, perturbation=perturbation)
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['robust loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                adv_imgs, deltas = self.attack(imgs, labels)
                adv_imgs, deltas =   self.attack(imgs, labels)
                self.optimizer.zero_grad()
                clean_loss = self.classifier.loss(self.predict(imgs), labels  )
                robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
                total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
                self.scaler.scale(total_loss).backward()
                if self.clip_grad:
                    clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =   self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = self.classifier.loss(self.predict(imgs), labels  )
            robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
            total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
            total_loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
            self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))

class Laplacian_DALE(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Laplacian_DALE, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.LMC_Laplacian_Linf(self.classifier, self.hparams, device, perturbation=perturbation)
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['robust loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                adv_imgs, deltas =   self.attack(imgs, labels)
                self.optimizer.zero_grad()
                clean_loss = self.classifier.loss(self.predict(imgs), labels  )
                robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
                total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
                self.scaler.scale(total_loss).backward()
                if self.clip_grad:
                    clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =   self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = self.classifier.loss(self.predict(imgs), labels  )
            robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
            total_loss = robust_loss + self.hparams['l_dale_nu'] * clean_loss
            total_loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
            self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))

class PrimalDualBase(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(PrimalDualBase, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.dual_params = {'dual_var': torch.tensor(init).to(self.device)}
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['robust loss'] = meters.AverageMeter()
        self.meters['dual variable'] = meters.AverageMeter()
        self.meters['delta L1-border'] = meters.AverageMeter()
        perturbation = vars(perturbations)[perturbation](0)
        self.meters['delta hist'] = meters.WBDeltaMeter(names = perturbation.names, dims = perturbation.dim)

class Gaussian_DALE_PD(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=1.0):
        super(Gaussian_DALE_PD, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.LMC_Gaussian_Linf(self.classifier, self.hparams, device, perturbation=perturbation)
        self.pd_optimizer = optimizers.PrimalDualOptimizer(
            parameters=self.dual_params,
            margin=self.hparams['g_dale_pd_margin'],
            eta=self.hparams['g_dale_pd_eta'])

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                adv_imgs, deltas = self.attack(imgs, labels)
                self.optimizer.zero_grad()
                clean_loss = self.classifier.loss(self.predict(imgs), labels  )
                robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
                total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
                self.scaler.scale(total_loss).backward()
                if self.clip_grad:
                    clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = self.classifier.loss(self.predict(imgs), labels  )
            robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
            total_loss = robust_loss + self.dual_params['dual_var'] * clean_loss
            total_loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
            self.optimizer.step()
        
        self.pd_optimizer.step(clean_loss.detach())
        
        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=imgs.size(0))
        self.meters['delta L1-border'].update((torch.abs(deltas)-self.hparams['epsilon']).mean().item(), n=imgs.size(0))
        self.meters['delta hist'].update(deltas.cpu())        
        #print(deltas[0])
        
class Gaussian_DALE_PD_Reverse(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=1.0):
        super(Gaussian_DALE_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.LMC_Gaussian_Linf(self.classifier, self.hparams, device, perturbation=perturbation)
        self.pd_optimizer = optimizers.PrimalDualOptimizer(
            parameters=self.dual_params,
            margin=self.hparams['g_dale_pd_inv_margin'],
            eta=self.hparams['g_dale_pd_inv_eta'])

    def step(self, imgs, labels):
        adv_imgs, deltas =self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = self.classifier.loss(self.predict(imgs), labels  )
        robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
        total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
        total_loss.backward()
        if self.clip_grad:
            clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
        self.optimizer.step()
        self.pd_optimizer.step(robust_loss.detach())
        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=1)
        self.meters['delta L1-border'].update((torch.abs(deltas)-self.hparams['epsilon']).mean().item(), n=imgs.size(0))
        self.meters['delta hist'].update(deltas.cpu())

class Laplacian_DALE_PD_Reverse(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(Laplacian_DALE_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.LMC_Laplacian_Linf(self.classifier, self.hparams, device, perturbation=perturbation)
        self.pd_optimizer = optimizers.PrimalDualOptimizer(
            parameters=self.dual_params,
            margin=self.hparams['l_dale_pd_inv_margin'],
            eta=self.hparams['l_dale_pd_inv_eta'])

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                adv_imgs, deltas = self.attack(imgs, labels)
                self.optimizer.zero_grad()
                clean_loss = self.classifier.loss(self.predict(imgs), labels  )
                robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
                total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
                self.scaler.scale(total_loss).backward()
                if self.clip_grad:
                    clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = self.classifier.loss(self.predict(imgs), labels  )
            robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
            total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
            total_loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
            self.optimizer.step()
        self.pd_optimizer.step(robust_loss.detach())
        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=1)
        self.meters['delta L1-border'].update((torch.abs(deltas)-self.hparams['epsilon']).mean().item(), n=imgs.size(0))
        self.meters['delta hist'].update(deltas.cpu())

class Worst_DALE_PD_Reverse(Laplacian_DALE_PD_Reverse):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(Worst_DALE_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.Worst_Of_K(self.classifier, self.hparams, device, perturbation=perturbation)

class PGD_DALE_PD_Reverse(Laplacian_DALE_PD_Reverse):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(PGD_DALE_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.Fo_PGD(self.classifier, self.hparams, device, perturbation=perturbation)

class Adam_DALE_PD_Reverse(Laplacian_DALE_PD_Reverse):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(Adam_DALE_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.Fo_Adam(self.classifier, self.hparams, device, perturbation=perturbation)

class Laplacian_PD_Reverse(Laplacian_DALE_PD_Reverse):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(Laplacian_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.Laplace_aug(self.classifier, self.hparams, device, perturbation=perturbation)

class Gaussian_PD_Reverse(Laplacian_DALE_PD_Reverse):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(Gaussian_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.Gaussian_aug(self.classifier, self.hparams, device, perturbation=perturbation)

class Beta_PD_Reverse(Laplacian_DALE_PD_Reverse):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(Beta_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.Beta_aug(self.classifier, self.hparams, device, perturbation=perturbation)
        
class MH_DALE_PD_Reverse(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(MH_DALE_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.meters['acceptance rate'] = meters.AverageMeter()
        self.attack = attacks.MH(self.classifier, self.hparams, device, perturbation=perturbation, acceptance_meter=self.meters['acceptance rate'])
        self.pd_optimizer = optimizers.PrimalDualOptimizer(
            parameters=self.dual_params,
            margin=self.hparams['g_dale_pd_inv_margin'],
            eta=self.hparams['g_dale_pd_inv_eta'])

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                adv_imgs, deltas = self.attack(imgs, labels)
                self.optimizer.zero_grad()
                clean_loss = self.classifier.loss(self.predict(imgs), labels  )
                robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
                total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
                self.scaler.scale(total_loss).backward()
                if self.clip_grad:
                    clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = self.classifier.loss(self.predict(imgs), labels  )
            robust_loss = self.classifier.loss(self.predict(adv_imgs), labels  )
            total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
            total_loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
            self.optimizer.step()
        self.pd_optimizer.step(robust_loss.detach())
        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=1)
        self.meters['delta L1-border'].update((torch.abs(deltas)-self.hparams['epsilon']).mean().item(), n=imgs.size(0))
        self.meters['delta hist'].update(deltas.cpu())

class KL_DALE_PD(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(KL_DALE_PD, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.TRADES_Linf(self.classifier, self.hparams, device, perturbation=perturbation)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.pd_optimizer = optimizers.PrimalDualOptimizer(
            parameters=self.dual_params,
            margin=self.hparams['g_dale_pd_margin'],
            eta=self.hparams['g_dale_pd_eta'])

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                adv_imgs, deltas = self.attack(imgs, labels)
                self.optimizer.zero_grad()
                clean_loss = self.classifier.loss(self.predict(imgs), labels  )
                robust_loss = self.kl_loss_fn(
                F.log_softmax(self.predict(adv_imgs), dim=1),
                F.softmax(self.predict(imgs), dim=1))
                total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
                self.scaler.scale(total_loss).backward()
                if self.clip_grad:
                    clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = self.classifier.loss(self.predict(imgs), labels  )
            robust_loss = self.kl_loss_fn(
                F.log_softmax(self.predict(adv_imgs), dim=1),
                F.softmax(self.predict(imgs), dim=1))
            total_loss = robust_loss + self.dual_params['dual_var'] * clean_loss
            total_loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
            self.optimizer.step()
        self.pd_optimizer.step(clean_loss.detach())

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=1)



        self.penalty = hparams["adv_penalty"]

class Grid_Search(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Grid_Search, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Grid_Search(self.classifier, self.hparams, device, perturbation=perturbation)

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                with torch.no_grad():
                    adv_imgs, deltas =   self.attack(imgs, labels)
                    self.optimizer.zero_grad()
                    loss = self.classifier.loss(self.predict(adv_imgs), labels  )
                    self.scaler.scale(loss).backward()
                    if self.clip_grad:
                        clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
        else:
            with torch.no_grad():
                adv_imgs, deltas =   self.attack(imgs, labels)
            self.optimizer.zero_grad()
            loss = self.classifier.loss(self.predict(adv_imgs), labels  )
            loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
            self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class Augmentation(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Augmentation, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Rand_Aug(self.classifier, self.hparams, device, perturbation=perturbation)
        self.p = hparams['augmentation_prob']
    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                if binomial(1, self.p):
                    adv_imgs, _ =   self.attack(imgs, labels)
                else:
                    adv_imgs = imgs
                self.optimizer.zero_grad()
                loss = self.classifier.loss(self.predict(adv_imgs), labels  )
                self.scaler.scale(loss).backward()
                if self.clip_grad:
                    clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:        
            if binomial(1, self.p):
                adv_imgs, _ =   self.attack(imgs, labels)
            else:
                adv_imgs = imgs
            self.optimizer.zero_grad()
            loss = self.classifier.loss(self.predict(adv_imgs), labels  )
            loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
            self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class Laplacian(Augmentation):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Laplacian, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Laplace_aug(self.classifier, self.hparams, device, perturbation=perturbation)
        self.p = 1

class Gaussian(Augmentation):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Gaussian, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Gaussian_aug(self.classifier, self.hparams, device, perturbation=perturbation)
        self.p = 1

class Batch_Random(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Batch_Random, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Rand_Aug_Batch(self.classifier, self.hparams, device, perturbation=perturbation)
    def step(self, imgs, labels):
        adv_imgs, deltas, new_labels =   self.attack(imgs, labels)
        self.optimizer.zero_grad()
        loss = self.classifier.loss(self.predict(adv_imgs), new_labels  )
        loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class Batch_Grid(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Batch_Grid, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Grid_Batch(self.classifier, self.hparams, device, perturbation=perturbation)
    
    def step(self, imgs, labels):
        adv_imgs, deltas, new_labels =  self.attack(imgs, labels)
        self.optimizer.zero_grad()
        loss = self.classifier.loss(self.predict(adv_imgs), new_labels  )
        loss.backward()
        if self.clip_grad:
            clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
        self.optimizer.step()
        self.meters['loss'].update(loss.item(), n=imgs.size(0))


class Discrete_DALE(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0, batched=False):
        if perturbation != 'SE':
            raise NotImplementedError
        super(Discrete_DALE, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.Grid_Batch(self.classifier, hparams, device, perturbation=perturbation)
        translations = []
        for idx in (1, 2):
            eps = hparams['epsilon'][idx]
            step = 2*eps/hparams['d_num_translations']
            translations.append(torch.arange(-eps, eps, step=step, device=self.device))
        eps = hparams['epsilon'][0]
        step = 2*eps/hparams['d_num_rotations']
        self.translations = translations
        self.rotation = torch.arange(-eps, eps, step=step, device=self.device)
        grids =  [self.rotation] + translations
        self.grid = torch.cartesian_prod(*grids)
        self.attack.grid = self.grid
        self.attack.grid_size = self.grid.shape[0]
        self.dual_params = {'dual_var': torch.ones(self.grid.shape[0]).to(self.device)*init}
        self.pd_optimizer = optimizers.PrimalDualOptimizer(parameters=self.dual_params,
                                                            margin=self.hparams['d_dale_pd_inv_margin'],
                                                            eta=self.hparams['d_dale_pd_inv_eta'])
        loc0 = (int(translations[0].shape[0]//2), int(translations[1].shape[0]//2))
        # Dual plot logger
        self.meters['dual plot'] = meters.WBDualMeter(self.grid,translations, names = "Dual var vs angle",
                                                         locs = [(0,0), loc0, (-1, -1)])
        
    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                with torch.no_grad():
                    adv_imgs, deltas, new_labels = self.attack(imgs, labels)
                self.optimizer.zero_grad()
                clean_loss = self.classifier.loss(self.predict(imgs), labels  )
                robust_loss = self.classifier.loss(self.predict(adv_imgs), new_labels, reduction='none'  )
                robust_loss = rearrange(robust_loss, '(B S) -> B S', B = imgs.shape[0])
                total_loss = clean_loss +  torch.mean(robust_loss@self.dual_params['dual_var'].to(self.device))
                self.scaler.scale(total_loss).backward()
                if self.clip_grad:
                    clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            with torch.no_grad():
                adv_imgs, deltas, new_labels =self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = self.classifier.loss(self.predict(imgs), labels  )
            robust_loss = self.classifier.loss(self.predict(adv_imgs), new_labels, reduction='none'  )
            robust_loss = rearrange(robust_loss, '(B S) -> B S', B = imgs.shape[0])
            total_loss = clean_loss +  torch.mean(robust_loss@self.dual_params['dual_var'].to(self.device))
            total_loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
            self.optimizer.step()
        with torch.no_grad():
            self.pd_optimizer.step(torch.mean(robust_loss, 0).detach())
        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.mean().item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].mean().item(), n=1)
        self.meters['dual plot'].update(self.dual_params['dual_var'])

    def get_grid(self, tx, ty):
        angle_grid = self.attack.grid
        ones = torch.ones_like(angle_grid)
        grid = torch.column_stack([angle_grid, tx*ones, ty*ones])
        return grid
