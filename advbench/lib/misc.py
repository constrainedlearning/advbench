import hashlib
import sys
from functools import wraps
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
try:
    import ffcv
    FFCV_AVAILABLE=True
except ImportError:
    FFCV_AVAILABLE=False

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
        with autocast():
            output = algorithm.predict(imgs)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.size(0)
    algorithm.train()
    algorithm.unexport()

    return 100. * correct / total

def adv_accuracy(algorithm, loader, device, attack):
    correct, total = 0, 0

    algorithm.eval()
    algorithm.export()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs, _ = attack(imgs, labels)

        with torch.no_grad():
            with autocast():
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
            with autocast():
                attacked = attack(imgs, labels)
                if len(attacked) == 2:
                    adv_imgs, delta = attacked
                elif len(attacked) == 3:
                    adv_imgs, delta, labels = attacked
                output = algorithm.predict(adv_imgs)
                loss = F.cross_entropy(output, labels, reduction='none')
                pred = output.argmax(dim=1, keepdim=True)
        losses.append(loss.cpu().numpy())
        deltas.append(delta.cpu().numpy())

        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += adv_imgs.size(0)
    algorithm.train()
    algorithm.unexport()
    acc = 100. * correct / total
    return acc, np.concatenate(losses, axis=0), np.concatenate(deltas, axis=0)

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