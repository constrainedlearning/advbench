import hashlib
from re import I
import sys
from functools import wraps
from time import time
from tqdm import tqdm
from einops import rearrange
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from advbench.datasets import FFCV_AVAILABLE
from sklearn.metrics import balanced_accuracy_score

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} took: {te-ts:.3f} sec')
        return result
    return wrap

def seed_hash(*args):
    """Derive an integer hash from all args, for use as a random seed."""

    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

@torch.no_grad()
def accuracy(algorithm, loader, device):
    correct, total = 0, 0
    algorithm_state = algorithm.training
    algorithm.eval()
    algorithm.export()
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        if FFCV_AVAILABLE:
            with autocast():
                output = algorithm.predict(imgs)
        else:
            output = algorithm.predict(imgs)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.size(0)
    assert(not algorithm.classifier.training)
    if algorithm_state:
        algorithm.train()
    algorithm.unexport()
    return 100. * correct / total

@torch.no_grad()
def accuracy_loss(algorithm, loader, device):
    correct, total = 0, 0
    losses = []
    algorithm_state = algorithm.training
    algorithm.eval()
    algorithm.export()
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        if FFCV_AVAILABLE:
            with autocast():
                output = algorithm.predict(imgs)
        else:
            output = algorithm.predict(imgs)
        pred = output.argmax(dim=1, keepdim=True)
        loss = algorithm.classifier.loss(output, labels, reduction='none')
        correct += pred.eq(labels.view_as(pred)).sum().item()
        losses.append(loss.detach().cpu().numpy())
        total += imgs.size(0)
    if algorithm_state:
        algorithm.train()
    algorithm.unexport()
    loss = np.concatenate(losses)
    return 100. * correct / total, np.mean(loss)
@torch.no_grad()
def accuracy_mean_overall(algorithm, loader, device):
    correct, total = 0, 0
    true = []
    preds = []
    algorithm_state = algorithm.training
    algorithm.eval()
    algorithm.export()
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        if FFCV_AVAILABLE:
            with autocast():
                output = algorithm.predict(imgs)
        else:
            output = algorithm.predict(imgs)
        pred = output.argmax(dim=1, keepdim=True)
        true.append(labels.cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.size(0) 
    if algorithm_state:
        algorithm.train()
    algorithm.unexport()
    true = np.concatenate(true)
    preds = np.concatenate(preds)
    mean = balanced_accuracy_score(true, preds)
    return 100. * correct / total, 100. * mean

@torch.no_grad()
def accuracy_mean_overall_loss(algorithm, loader, device, max_batches = None):
    correct, total = 0, 0
    true = []
    preds = []
    losses = []
    algorithm_state = algorithm.training
    algorithm.eval()
    algorithm.export()
    for batch_idx, (imgs, labels) in tqdm(enumerate(loader)):
        if max_batches is not None and batch_idx>max_batches-1:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        if FFCV_AVAILABLE:
            with autocast():
                output = algorithm.predict(imgs)
        else:
            output = algorithm.predict(imgs)
        loss = algorithm.classifier.loss(output, labels, reduction='none')
        pred = output.argmax(dim=1, keepdim=True)
        true.append(labels.cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        correct += pred.eq(labels.view_as(pred)).sum().item()
        losses.append(loss.detach().cpu().numpy())
        total += imgs.size(0) 
    if algorithm_state:
        algorithm.train()
    algorithm.unexport()
    true = np.concatenate(true)
    preds = np.concatenate(preds)
    loss = np.concatenate(losses)
    mean = balanced_accuracy_score(true, preds)
    return 100. * correct / total, 100. * mean, np.mean(loss)

def adv_accuracy(algorithm, loader, device, attack):
    correct, total = 0, 0
    algorithm_state = algorithm.training
    algorithm.eval()
    algorithm.export()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs, _ = attack(imgs, labels)

        with torch.no_grad():
            if FFCV_AVAILABLE:
                with autocast():
                    output = algorithm.predict(adv_imgs)
            else:
                output = algorithm.predict(adv_imgs)
            pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.size(0)
    if algorithm_state:
        algorithm.train()
    algorithm.unexport()

    return 100. * correct / total

def adv_accuracy_loss_delta(algorithm, loader, device, attack, max_batches=None, augs_per_batch=1, batched=False):
    adv_correct, correct, total, total_worst, adv_losses = 0, 0, 0, 0, 0
    losses, deltas, accs = [], [], []
    algorithm_state = algorithm.training
    algorithm.eval()
    algorithm.export()
    with torch.no_grad():
        for batch_idx, (imgs, labels) in tqdm(enumerate(loader)):
            if max_batches is not None and batch_idx>max_batches-1:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            if batched:
                batch_losses , batch_preds, batch_deltas, batch_labels = [], [], [], []
                for _ in range(augs_per_batch):
                    if FFCV_AVAILABLE:
                        with autocast():
                            attacked = attack(imgs, labels)
                            if len(attacked) == 2:
                                adv_imgs, delta = attacked
                            elif len(attacked) == 3:
                                adv_imgs, delta, labels = attacked
                            output = algorithm.predict(adv_imgs)
                    else:
                        assert(not algorithm.classifier.training)
                        attacked = attack(imgs, labels)
                        assert(not algorithm.classifier.training)
                        if len(attacked) == 2:
                            adv_imgs, delta = attacked
                        elif len(attacked) == 3:
                            adv_imgs, delta, labels = attacked
                        output = algorithm.predict(adv_imgs)
                        loss = algorithm.classifier.loss(output, labels, reduction='none')
                        pred = output.argmax(dim=1)
                    batch_losses.append(loss)
                    batch_preds.append(pred)
                    batch_deltas.append(delta)
                    batch_labels.append(labels)
                loss = torch.stack(batch_losses, dim=1)
                pred = torch.stack(batch_preds, dim=1)
                delta = torch.stack(batch_deltas, dim=1)
                labels = torch.stack(batch_labels, dim=1)
                #pred = rearrange(pred, '(B S) -> B S', B=imgs.shape[0])
                eq = pred.eq(labels)
                accs.append(eq.cpu().numpy())
                worst , _ = eq.min(dim = 1)
                adv_correct += worst.sum().item()
                correct += eq.sum().item()
                losses.append(loss.cpu().numpy())
                #loss = rearrange(loss, '(B S) -> B S', B=imgs.shape[0])
                worst_loss, _ = loss.max(dim=1)
                adv_losses += worst_loss.sum().item()
                deltas.append(delta.cpu().numpy())
                total += torch.numel(labels)
                total_worst += imgs.size(0)
            else:
                assert(not algorithm.classifier.training)
                attacked = attack(imgs, labels)
                assert(not algorithm.classifier.training)
                if len(attacked) == 2:
                    adv_imgs, delta = attacked
                elif len(attacked) == 3:
                    adv_imgs, delta, labels = attacked
                output = algorithm.predict(adv_imgs)
                loss = algorithm.classifier.loss(output, labels, reduction='none')
                pred = output.argmax(dim=1)
                pred = rearrange(pred, '(B S) -> B S', B=imgs.shape[0])
                eq = pred.eq(labels.view_as(pred))
                accs.append(eq.view_as(loss).cpu().numpy())
                worst , _ = eq.min(dim = 1)
                adv_correct += worst.sum().item()
                correct += eq.sum().item()
                losses.append(loss.cpu().numpy())
                loss = rearrange(loss, '(B S) -> B S', B=imgs.shape[0])
                worst_loss, _ = loss.max(dim=1)
                adv_losses += worst_loss.sum().item()
                deltas.append(delta.cpu().numpy())
                total += adv_imgs.size(0)
                total_worst += imgs.size(0)
    assert(not algorithm.classifier.training)
    if algorithm_state:
        algorithm.train()
    algorithm.unexport()
    adv_acc = 100. * adv_correct / total_worst
    adv_mean = 100. * correct / total
    adv_loss = adv_losses / total_worst
    accs = np.array(accs)
    losses = np.array(losses)
    return adv_acc, adv_mean, adv_loss, accs, losses, np.concatenate(deltas, axis=0)

def adv_accuracy_loss_delta_balanced(algorithm, loader, device, attack, max_batches = None, augs_per_batch=1, batched=True):
    adv_correct, correct, total, total_worst, adv_losses = 0, 0, 0, 0, 0
    losses, deltas, accs, worst_preds, all_labels, repeated_labels, all_preds = [], [], [], [], [], [], []
    algorithm_state = algorithm.training
    algorithm.eval()
    algorithm.export()
    with torch.no_grad():
        for batch_idx, (imgs, labels) in tqdm(enumerate(loader)):
            if max_batches is not None and batch_idx>max_batches-1:
                break 
            imgs, labels = imgs.to(device), labels.to(device)
            all_labels.append(labels.cpu().numpy())
            if not batched:
                attacked = attack(imgs, labels)
                if len(attacked) == 2:
                    adv_imgs, delta = attacked
                elif len(attacked) == 3:
                    adv_imgs, delta, labels = attacked
                output = algorithm.predict(adv_imgs)
                loss = algorithm.classifier.loss(output, labels, reduction='none')
                pred = output.argmax(dim=1)
            else:
                batch_preds, batch_deltas, batch_labels, batch_losses = [], [], [], []
                for _ in range(augs_per_batch):
                    attacked = attack(imgs, labels)
                    if len(attacked) == 2:
                        adv_imgs, delta = attacked
                    elif len(attacked) == 3:
                        adv_imgs, delta, labels = attacked
                    output = algorithm.predict(adv_imgs)
                    loss = algorithm.classifier.loss(output, labels, reduction='none')
                    pred = output.argmax(dim=1)
                    batch_losses.append(loss)
                    batch_preds.append(pred)
                    batch_deltas.append(delta)
                    batch_labels.append(labels)
                loss = torch.concat(batch_losses, dim=0)
                pred = torch.concat(batch_preds, dim=0)
                delta = torch.concat(batch_deltas, dim=0)
                labels = torch.concat(batch_labels, dim=0)
            all_preds.append(pred.cpu().numpy())
            repeated_labels.append(labels.cpu().numpy())
            pred = rearrange(pred, '(B S) -> B S', B=imgs.shape[0])
            eq = pred.eq(labels.view_as(pred))
            accs.append(eq.view_as(loss).cpu().numpy())
            worst, worst_idx = eq.min(dim = 1)
            adv_correct += worst.sum().item()
            correct += eq.sum().item()
            losses.append(loss.cpu().numpy())
            loss = rearrange(loss, '(B S) -> B S', B=imgs.shape[0])
            worst_loss, _ = loss.max(dim=1)
            adv_losses += worst_loss.sum().item()
            deltas.append(delta.cpu().numpy())
            total += adv_imgs.size(0)
            total_worst += imgs.size(0)
            worst_preds.append(worst.cpu().numpy())
    if algorithm_state:
        algorithm.train()
    algorithm.unexport()
    adv_acc = 100. * adv_correct / total_worst
    adv_mean = 100. * correct / total
    adv_loss = adv_losses / total_worst
    adv_acc_bal = 100. * balanced_accuracy_score(np.concatenate(all_labels, axis=0), np.concatenate(worst_preds, axis=0))
    adv_mean_bal = 100. * balanced_accuracy_score(np.concatenate(repeated_labels, axis=0), np.concatenate(all_preds, axis=0))
    return adv_acc, adv_mean, adv_acc_bal, adv_mean_bal, adv_loss, np.concatenate(accs, axis=0), np.concatenate(losses, axis=0), np.concatenate(deltas, axis=0)

def adv_accuracy_loss_delta_ensembleacc(algorithm, loader, device, attack):
    correct, ensemble_correct, total, total_ens = 0, 0, 0, 0
    losses, accs, deltas = [], [], []
    algorithm_state = algorithm.training
    algorithm.eval()
    algorithm.export()
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            if FFCV_AVAILABLE:
                with autocast():
                    attacked = attack(imgs, labels)
            else:
                attacked = attack(imgs, labels)
            old_labels = labels
            if len(attacked) == 2:
                adv_imgs, delta = attacked
            elif len(attacked) == 3:
                adv_imgs, delta, labels = attacked
            if FFCV_AVAILABLE:
                with autocast():
                    output = algorithm.predict(adv_imgs)
            else:
                output = algorithm.predict(adv_imgs)
            loss = algorithm.classifier.loss(output, labels, reduction='none')
            pred = output.argmax(dim=1, keepdim=True)
            if len(attacked) == 3:
                # get the models prediction for each transform
                ensemble_preds = torch.zeros_like(output)
                ensemble_preds[torch.arange(ensemble_preds.shape[0]), pred.squeeze()] = 1
                ensemble_preds = rearrange(ensemble_preds, '(B S) C -> B S C', B=imgs.shape[0], C=output.shape[1])
            # Average over transforms (S)
                ensemble_preds = ensemble_preds.mean(dim=1)
            # predict using average
                ensemble_preds = ensemble_preds.argmax(dim=1, keepdim=True)
            else:
                ensemble_preds = pred
            losses.append(loss.cpu().numpy())
            deltas.append(delta.cpu().numpy())
            corr = pred.eq(labels.view_as(pred))
            accs.append(corr.cpu().numpy())
            ensemble_correct += ensemble_preds.eq(old_labels.view_as(ensemble_preds)).sum().item() 
            correct += corr.sum().item()
            total += adv_imgs.size(0)
            total_ens += imgs.size(0)
    if algorithm_state:
        algorithm.train()
    algorithm.unexport()
    acc = 100. * correct / total
    ensemble_acc = 100. * ensemble_correct / total_ens
    return acc, np.concatenate(accs, axis=0), np.concatenate(losses, axis=0), np.concatenate(deltas, axis=0), ensemble_acc

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
