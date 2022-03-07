from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pandas as pd
from numpy.random import binomial
from torch.cuda.amp import GradScaler, autocast
try:
    import ffcv
    raise ImportError
    FFCV_AVAILABLE=True
    print("*"*80)
    print('FFCV available. Using Low precision operations. May result in numerical instability.')
    print("*"*80)
except ImportError:
    FFCV_AVAILABLE=False

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
    'MCMC_DALE_PD_Reverse',
    'KL_DALE_PD',
    'Worst_Of_K',
    'Augmentation',
    'Batch_Augmentation',
    'Grid_Search',
    'Batch_Grid',
    'NUTS_DALE',
    'Worst_DALE_PD_Reverse',
    'PGD_DALE_PD_Reverse'
]

class Algorithm(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.classifier = networks.Classifier(
            input_shape, num_classes, hparams)
        self.optimizer = optimizers.Optimizer(
            self.classifier, hparams)
        self.device = device
        
        self.meters = OrderedDict()
        self.meters['loss'] = meters.AverageMeter()
        self.meters_df = None
        self.perturbation_name = perturbation
        if FFCV_AVAILABLE:
            self.scaler = GradScaler()

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
                loss = F.cross_entropy(self.predict(imgs), labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            loss = F.cross_entropy(self.predict(imgs), labels)
            loss.backward()
        self.optimizer.step()
        
        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class PGD(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(PGD, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device, perturbation=perturbation)

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                adv_imgs, deltas = self.attack(imgs, labels)
                loss = F.cross_entropy(self.predict(adv_imgs), labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =   self.attack(imgs, labels)
            self.optimizer.zero_grad()
            loss = F.cross_entropy(self.predict(adv_imgs), labels)
            loss.backward()
            self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class FGSM(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(FGSM, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.FGSM_Linf(self.classifier, self.hparams, device, perturbation=perturbation)

    def step(self, imgs, labels):

        adv_imgs, deltas =   self.attack(imgs, labels)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(adv_imgs), labels)
        loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class TRADES(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(TRADES, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')  # TODO(AR): let's write a method to do the log-softmax part
        self.attack = attacks.TRADES_Linf(self.classifier, self.hparams, device, perturbation=perturbation)
        
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['invariance loss'] = meters.AverageMeter()

    def step(self, imgs, labels):

        adv_imgs, deltas =   self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        robust_loss = self.kl_loss_fn(
            F.log_softmax(self.predict(adv_imgs), dim=1),
            F.softmax(self.predict(imgs), dim=1))
        total_loss = clean_loss + self.hparams['trades_beta'] * robust_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['invariance loss'].update(robust_loss.item(), n=imgs.size(0))

        return {'loss': total_loss.item()}

class LogitPairingBase(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(LogitPairingBase, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device, perturbation=perturbation)
        self.meters['logit loss'] = meters.AverageMeter()

    def pairing_loss(self, imgs, adv_imgs):
        logit_diff = self.predict(adv_imgs) - self.predict(imgs)
        return torch.norm(logit_diff, dim=1).mean()

class ALP(LogitPairingBase):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(ALP, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device, perturbation=perturbation)
        self.meters['robust loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs, deltas =   self.attack(imgs, labels)
        self.optimizer.zero_grad()
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        logit_pairing_loss = self.pairing_loss(imgs, adv_imgs)
        total_loss = robust_loss + logit_pairing_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['logit loss'].update(logit_pairing_loss.item(), n=imgs.size(0))

class CLP(LogitPairingBase):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(CLP, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device, perturbation=perturbation)

        self.meters['clean loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs, deltas =   self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        logit_pairing_loss = self.pairing_loss(imgs, adv_imgs)
        total_loss = clean_loss + logit_pairing_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['logit loss'].update(logit_pairing_loss.item(), n=imgs.size(0))

class MART(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(MART, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.kl_loss_fn = nn.KLDivLoss(reduction='none')
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device, perturbation=perturbation)

        self.meters['robust loss'] = meters.AverageMeter()
        self.meters['invariance loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        
        adv_imgs, deltas =   self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_output = self.classifier(imgs)
        adv_output = self.classifier(adv_imgs)
        adv_probs = F.softmax(adv_output, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_label = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(adv_output, labels) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_label)
        nat_probs = F.softmax(clean_output, dim=1)
        true_probs = torch.gather(nat_probs, 1, (labels.unsqueeze(1)).long()).squeeze()
        loss_robust = (1.0 / imgs.size(0)) * torch.sum(
            torch.sum(self.kl_loss_fn(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = loss_adv + self.hparams['mart_beta'] * loss_robust
        loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(loss_robust.item(), n=imgs.size(0))
        self.meters['invariance loss'].update(loss_adv.item(), n=imgs.size(0))


class MMA(Algorithm):
    pass

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
                clean_loss = F.cross_entropy(self.predict(imgs), labels)
                robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
                total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =   self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = F.cross_entropy(self.predict(imgs), labels)
            robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
            total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
            total_loss.backward()
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
                clean_loss = F.cross_entropy(self.predict(imgs), labels)
                robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
                total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =   self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = F.cross_entropy(self.predict(imgs), labels)
            robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
            total_loss = robust_loss + self.hparams['l_dale_nu'] * clean_loss
            total_loss.backward()
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
                clean_loss = F.cross_entropy(self.predict(imgs), labels)
                robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
                total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = F.cross_entropy(self.predict(imgs), labels)
            robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
            total_loss = robust_loss + self.dual_params['dual_var'] * clean_loss
            total_loss.backward()
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
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
        total_loss.backward()
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
                clean_loss = F.cross_entropy(self.predict(imgs), labels)
                robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
                total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = F.cross_entropy(self.predict(imgs), labels)
            robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
            total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
            total_loss.backward()
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
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device, perturbation=perturbation)

class MCMC_DALE_PD_Reverse(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0):
        super(MCMC_DALE_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.meters['acceptance rate'] = meters.AverageMeter()
        self.attack = attacks.MCMC(self.classifier, self.hparams, device, perturbation=perturbation, acceptance_meter=self.meters['acceptance rate'])
        self.pd_optimizer = optimizers.PrimalDualOptimizer(
            parameters=self.dual_params,
            margin=self.hparams['g_dale_pd_inv_margin'],
            eta=self.hparams['g_dale_pd_inv_eta'])

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                adv_imgs, deltas = self.attack(imgs, labels)
                self.optimizer.zero_grad()
                clean_loss = F.cross_entropy(self.predict(imgs), labels)
                robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
                total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = F.cross_entropy(self.predict(imgs), labels)
            robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
            total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
            total_loss.backward()
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
                clean_loss = F.cross_entropy(self.predict(imgs), labels)
                robust_loss = self.kl_loss_fn(
                F.log_softmax(self.predict(adv_imgs), dim=1),
                F.softmax(self.predict(imgs), dim=1))
                total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas =self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = F.cross_entropy(self.predict(imgs), labels)
            robust_loss = self.kl_loss_fn(
                F.log_softmax(self.predict(adv_imgs), dim=1),
                F.softmax(self.predict(imgs), dim=1))
            total_loss = robust_loss + self.dual_params['dual_var'] * clean_loss
            total_loss.backward()
            self.optimizer.step()
        self.pd_optimizer.step(clean_loss.detach())

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=1)


class Worst_Of_K(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Worst_Of_K, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Worst_Of_K(self.classifier, self.hparams, device, perturbation=perturbation)

    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                with torch.no_grad():
                    adv_imgs, deltas =   self.attack(imgs, labels)
                self.optimizer.zero_grad()
                clean_loss = F.cross_entropy(self.predict(imgs), labels)
                robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
                total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:        
            with torch.no_grad():
                adv_imgs, deltas =   self.attack(imgs, labels)
            self.optimizer.zero_grad()
            loss = F.cross_entropy(self.predict(adv_imgs), labels)
            loss.backward()
            self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

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
                    loss = F.cross_entropy(self.predict(adv_imgs), labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
        else:
            with torch.no_grad():
                adv_imgs, deltas =   self.attack(imgs, labels)
            self.optimizer.zero_grad()
            loss = F.cross_entropy(self.predict(adv_imgs), labels)
            loss.backward()
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
                loss = F.cross_entropy(self.predict(adv_imgs), labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:        
            if binomial(1, self.p):
                adv_imgs, _ =   self.attack(imgs, labels)
            else:
                adv_imgs = imgs
            self.optimizer.zero_grad()
            loss = F.cross_entropy(self.predict(adv_imgs), labels)
            loss.backward()
            self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class Batch_Random(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        super(Batch_Random, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
        self.attack = attacks.Rand_Aug_Batch(self.classifier, self.hparams, device, perturbation=perturbation)
    def step(self, imgs, labels):
        adv_imgs, deltas, new_labels =   self.attack(imgs, labels)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(adv_imgs), new_labels)
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
        loss = F.cross_entropy(self.predict(adv_imgs), new_labels)
        loss.backward()
        self.optimizer.step()
        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class NUTS_DALE(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=1.0):
        super(NUTS_DALE, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.NUTS(self.classifier, self.hparams, device, perturbation=perturbation)
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['robust loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs, deltas = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        total_loss = robust_loss + self.hparams['l_dale_nu'] * clean_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))

class GConv(Augmentation):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf'):
        hparams['epsilon_rot'] = 0.0
        super(GConv, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation)
    def export(self):
        self.classifier.export()
        self.classifier.to(self.device)
    def unexport(self):
        self.classifier.unexport()
        self.classifier.to(self.device)

class Discrete_DALE(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device, perturbation='Linf', init=0.0, batched=False):
        if perturbation != 'SE':
            raise NotImplementedError
        super(Discrete_DALE, self).__init__(input_shape, num_classes, hparams, device, perturbation=perturbation, init=init)
        self.attack = attacks.Grid_Search(classifier, hparams, device, perturbation="Rotation", grid_size=hparams['d_num_rotations'])
        translations = []
        for idx in (1, 2):
            eps = hparams['epsilon'][idx]
            step = 2*eps/hparams['d_num_translations']
            translations.append(torch.arange(-eps, eps, step=step, device=self.device))
        self.translations = translations
        self.rotation = self.attack.grid
        grids = translations + self.rotation
        self.grid = torch.cartesian_prod(*grids)
        self.dual_params = {'dual_var': torch.ones(self.attack.grid.size).to(self.device)*init}
        self.pd_optimizer = optimizers.PrimalDualOptimizer(parameters=self.dual_params,
                                                            margin=self.hparams['d_dale_pd_inv_margin'],
                                                            eta=self.hparams['d_dale_pd_inv_eta'])
        # calculate the coordinates of zero translation
        print(self.attack.grid.shape)
        loc0 = (int(self.attack.grid.shape[0]//2), int(self.attack.grid.shape[0]//2))
        # Dual plot logger
        self.meters['dual plot'] = meters.WBDualMeter(self.attack.grid, names = "Dual var vs angle",
                                                         locs = [(0,0), loc0, (-1, -1)])
        
    def step(self, imgs, labels):
        if FFCV_AVAILABLE:
            with autocast():
                adv_imgs, deltas, new_labels = self.attack(imgs, labels)
                self.optimizer.zero_grad()
                clean_loss = F.cross_entropy(self.predict(imgs), labels)
                robust_loss = F.cross_entropy(self.predict(adv_imgs), new_labels)
                total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            adv_imgs, deltas, new_labels =self.attack(imgs, labels)
            self.optimizer.zero_grad()
            clean_loss = F.cross_entropy(self.predict(imgs), labels)
            robust_loss = F.cross_entropy(self.predict(adv_imgs), new_labels, reduction='none')
            robust_loss = rearrange(robust_loss, '(B S) -> B S', B = imgs.shape[0])
            total_loss = clean_loss +  torch.sum(robust_loss@self.dual_params['dual_var'].to(self.device))
            total_loss.backward()
            self.optimizer.step()
        self.pd_optimizer.step(robust_loss.detach())
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
