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
    algorithm.train()
    algorithm.unexport()

    return 100. * correct / total

@torch.no_grad()
def accuracy_mean_overall(algorithm, loader, device):
    correct, total = 0, 0
    true = []
    preds = []
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
    algorithm.train()
    algorithm.unexport()
    true = np.concatenate(true)
    preds = np.concatenate(preds)
    mean = balanced_accuracy_score(true, preds)
    return 100. * correct / total, 100. * mean

def adv_accuracy(algorithm, loader, device, attack):
    correct, total = 0, 0

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
    algorithm.train()
    algorithm.unexport()

    return 100. * correct / total

def adv_accuracy_loss_delta(algorithm, loader, device, attack):
    correct, total = 0, 0
    losses, deltas = [], []

    algorithm.eval()
    algorithm.export()
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            if FFCV_AVAILABLE:
                with autocast():
                    attacked = attack(imgs, labels)
                    if len(attacked) == 2:
                        adv_imgs, delta = attacked
                    elif len(attacked) == 3:
                        adv_imgs, delta, labels = attacked
                    output = algorithm.predict(adv_imgs)
            else:
                attacked = attack(imgs, labels)
                if len(attacked) == 2:
                    adv_imgs, delta = attacked
                elif len(attacked) == 3:
                    adv_imgs, delta, labels = attacked
                output = algorithm.predict(adv_imgs)
                loss = algorithm.classifier.loss(output, labels, reduction='none')
                pred = output.argmax(dim=1, keepdim=True)
            losses.append(loss.cpu().numpy())
            deltas.append(delta.cpu().numpy())
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += adv_imgs.size(0)
    algorithm.train()
    algorithm.unexport()
    acc = 100. * correct / total
    return acc, np.concatenate(losses, axis=0), np.concatenate(deltas, axis=0)

def adv_accuracy_loss_delta_ensembleacc(algorithm, loader, device, attack):
    correct, ensemble_correct, total, total_ens = 0, 0, 0, 0
    losses, accs, deltas = [], [], []

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
    algorithm.train()
    algorithm.unexport()
    acc = 100. * correct / total
    ensemble_acc = 100. * ensemble_correct / total_ens
    return acc, np.concatenate(accs, axis=0), np.concatenate(losses, axis=0), np.concatenate(deltas, axis=0), ensemble_acc

def adv_accuracy_loss_delta_ensembleacc_overall(algorithm, loader, device, attack):
    correct, ensemble_correct, total, total_ens = 0, 0, 0, 0
    losses, accs, deltas, true, preds, ens_labels, ens_preds = [], [], [], [], [], [], []

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
            true.append(labels.cpu().numpy())
            ens_labels.append(old_labels.cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
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
            ens_preds.append(ensemble_preds.cpu().numpy()) 
            correct += corr.sum().item()
            total += adv_imgs.size(0)
            total_ens += imgs.size(0)
            break
            
    algorithm.train()
    algorithm.unexport()
    acc = 100. * correct / total
    ensemble_acc = 100. * ensemble_correct / total_ens
    true = np.concatenate(true)
    preds = np.concatenate(preds)
    ens_preds = np.concatenate(ens_preds)
    ens_labels = np.concatenate(ens_labels)
    ens_mean = balanced_accuracy_score(ens_labels, ens_preds)
    mean = balanced_accuracy_score(true, preds)
    return acc, np.concatenate(accs, axis=0), np.concatenate(losses, axis=0), np.concatenate(deltas, axis=0), ensemble_acc, ens_mean, mean

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

def adv_accuracy_loss_delta_ensembleacc(algorithm, loader, device, attack):
    correct, ensemble_correct, total, total_ens = 0, 0, 0, 0
    losses, accs, deltas = [], [], []

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
    algorithm.train()
    algorithm.unexport()
    acc = 100. * correct / total
    ensemble_acc = 100. * ensemble_correct / total_ens
    return acc, np.concatenate(accs, axis=0), np.concatenate(losses, axis=0), np.concatenate(deltas, axis=0), ensemble_acc
