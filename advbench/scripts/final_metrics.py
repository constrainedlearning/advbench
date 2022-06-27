from pathlib import Path
import os
import wandb
import pandas as pd
import matplotlib.colors as mcolors
import os
import sys
import seaborn as sns
from tqdm import tqdm
sys.path.append('../')
import numpy as np
import pandas as pd
from advbench.datasets import MNIST, STL10
import argparse
from advbench.datasets import to_loaders
from advbench.algorithms import ERM, Augmentation, Adversarial_Worst_Of_K, Adversarial_PGD
from advbench.attacks import Fo_Adam
from advbench import hparams_registry
from advbench.lib import  misc
import torch
from advbench.lib.transformations import se_matrix, angle_to_rotation_matrix, se_transform
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import tarfile
from torch import nn
import shutil
from pathlib import Path
import os
import wandb
import pandas as pd
import matplotlib.colors as mcolors
import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd
from advbench.datasets import MNIST, STL10
from advbench.datasets import to_loaders
from advbench.algorithms import ERM, Augmentation, Adversarial_Worst_Of_K, Adversarial_PGD
from advbench.attacks import Fo_Adam
from advbench import hparams_registry
from advbench.lib import  misc
import torch
from advbench.lib.transformations import se_matrix, angle_to_rotation_matrix, se_transform
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn

import torch
import random
import numpy as np
import os
import json
import pandas as pd
import time
from humanfriendly import format_timespan


from advbench import datasets
from advbench import algorithms
from advbench import attacks
from advbench import hparams_registry
from advbench.lib import misc, meters, plotting, logging
from torch.cuda.amp import autocast
from torchsummary import summary

def load_weights(weight_path, dataset,hparams, device="cuda:1"):       
        algorithm = ERM(
        dataset.INPUT_SHAPE, 
        dataset.NUM_CLASSES,
        hparams,
        device).to(device)
        w_dict = torch.load(weight_path,map_location='cuda:1')
        #print(weight_dict.keys())
        #if db=="mnist":
        weight_dict = {}
        for s,v in w_dict.items():
            if "classifier" in s and "model" not in s:
                s = s.replace('classifier', 'classifier.model')
            if dataset == "MNIST":
                if "model.conv1" in s:
                    s = s.replace('model.conv1', 'model.convs.0')
                if "model.conv2" in s:
                    s = s.replace('model.conv2', 'model.convs.2')
            weight_dict[s] = v
        algorithm.load_state_dict(weight_dict)
        return algorithm

def compute_metrics(args):
    if args.download:
        api = wandb.Api(timeout=50)
        dsets = args.datasets#["CIFAR100", "MNIST", "STL10"]
        #results_dfs = {}
        entity = "hounie"
        all_runs_list = []
        for dataset in dsets:
            print("Fetching data for {}".format(dataset))
            project = f"DAug-{dataset}"
            base = f"../trained_weights/{project}"
            Path( base ).mkdir( parents=True, exist_ok=True )
            runs = api.runs(f"{entity}/{project}")
            print(f"{len(runs)} runs found")
            for run in runs:
                if run.state == "finished":
                    id = run.id
                    if True:
                        run = api.run(f"{entity}/{project}/{id}")
                        tmp_path = os.path.join(base, f"tmp")
                        weight_dir = os.path.join(base, run.name)
                        path = os.path.join(weight_dir, f"{run.id}.pkl")
                        Path( weight_dir ).mkdir( parents=True, exist_ok=True )
                        results = {**run.summary, **run.config, "weights": path, "id": run.id, "name":run.name}
                        try:
                            f = run.file("train-output/delta hist_ckpt.pkl").download(tmp_path, replace=True) 
                            os.rename(f.name, path)
                            all_runs_list.append(results)
                        except:
                            try:
                                run = api.run(f"{entity}/{project}/{id}")
                                f = run.file("train-output/loss_ckpt.pkl").download(tmp_path, replace=True)
                                os.rename(f.name, path)
                                all_runs_list.append(results)
                            except:
                                try:
                                    run = api.run(f"{entity}/{project}/{id}")
                                    f = run.file("train-output/acceptance rate_ckpt.pkl").download(tmp_path, replace=True)
                                    os.rename(f.name, path)
                                    all_runs_list.append(results)
                                except:
                                    print(f"{run.name} {run.id} failed")
                                    pass
                #break
            df = pd.DataFrame(all_runs_list)
            df.to_csv(f"./results_{dataset}.csv")
    for dataset in args.datasets:
        df = pd.read_csv(f"./results_{dataset}.csv")
        test_keys = []
        train_keys = []
        for c in df.columns:
                if "loss" not in c and "acc" not in c:
                    if c.startswith("test"):
                        test_keys.append(c)
                        #print(c)
                    else:
                        train_keys.append(c)
                    df = pd.read_csv(f"./results_{dataset}.csv")
                        #df = pd.read_csv(f"./results_{dataset}.csv")
        test_keys = []
        train_keys = []
        for c in df.columns:
                if "loss" not in c and "acc" not in c:
                    if c.startswith("test"):
                        test_keys.append(c)
                        #print(c)
                    else:
                        train_keys.append(c)

        data_dir = '../advbench/data'
        device = "cuda:1"
        for exp_id in tqdm(df["id"]):
            if True:
                api = wandb.Api(timeout=50)
                exp = df[df["id"]==exp_id]
                dataset =  exp["dataset"].values[0]
                algorithm = "ERM"
                algo = exp["algorithm"].values[0]
                perturbation = exp["perturbation"].values[0]
                name = exp["name"].values[0]
                model = exp["model"].values[0]
                augment = exp["augment"].values[0]
                if perturbation == "SE" and "rot" not in model and "Laplacian_DALE" in name and "range" not in name and augment:
                    project = f"DAug-{dataset}"
                    t_hparams = exp[test_keys].to_dict(orient='index')[exp.index.values[0]]
                    test_hparams = {}
                    for k, v in t_hparams.items():
                        test_hparams[k.replace("test_","")] = v

                    hparams = exp[train_keys].to_dict(orient='index')[exp.index.values[0]]
                    if model == "wrn-16-8":
                        hparams['model'] = "wrn-16-8-stl"
                    else:
                        hparams['model'] = model
                    hparams['epsilon'] = torch.tensor([hparams[f'epsilon_{i}'] for i in ("rot","tx","ty")]).to(device)
                    test_hparams['epsilon'] = hparams['epsilon']
                    hparams['batched'] = False
                    test_hparams['batched'] = False
                    aug = hparams["augment"]
                    dataset = vars(datasets)[dataset](data_dir, augmentation = aug)
                    train_ldr, val_ldr, test_ldr = datasets.to_loaders(dataset, hparams, device=device)
                    kw_args = {"perturbation": perturbation}
                    if dataset =="MNIST" and not isinstance(hparams["n_layers"], int):
                        hparams.pop("n_layers")
                    algorithm = load_weights(exp["weights"].values[0], dataset, hparams, device="cuda:1")

                    test_attacks = {
                        a: vars(attacks)[a](algorithm.classifier, test_hparams, device, perturbation=perturbation) for a in args.attacks}
                    #test_attacks = {
                        #"Uniform": attacks.Rand_Aug(algorithm.classifier, test_hparams, device, perturbation="SE"),
                        #"Gaussian": attacks.Rand_Aug(algorithm.classifier, test_hparams, device, perturbation="SE"),
                        #"Laplace": attacks.Rand_Aug(algorithm.classifier, test_hparams, device, perturbation="SE"),}
                    #try:
                        #train_clean_acc, train_clean_loss = misc.accuracy_loss(algorithm, val_ldr, device)
                    test_clean_acc, test_clean_loss = misc.accuracy_loss(algorithm, test_ldr, device)
                    #except:
                    #    print("failed to compute clean, wtf")
                    #try:
                        #run = api.run(f"hounie/{project}/{exp_id}")
                    wandb.init(project=project, resume=exp_id)
                    #wandb.log({'train_clean_acc_nb': train_clean_acc, 'train_clean_loss_nb': train_clean_loss})
                    wandb.log({'final test_clean_acc': test_clean_acc, 'final test_clean_loss': test_clean_loss})
                    #wandb.config.update({"model": "wrn-16-8-stl"})
                    # compute save and log adversarial accuracies on validation/test sets
                    for attack_name, attack in test_attacks.items():
                            test_adv_acc, test_adv_acc_mean, adv_loss, accs, loss, deltas = misc.adv_accuracy_loss_delta(algorithm, test_ldr, device, attack, augs_per_batch=args.n_aug)
                            train_adv_acc, train_adv_acc_mean, train_adv_loss, train_accs, train_loss, train_deltas = misc.adv_accuracy_loss_delta(algorithm, val_ldr, device, attack, augs_per_batch=args.n_aug)
                            print(f"Logging {attack_name}...")
                            wandb.log({'final test_acc_adv_'+attack_name: test_adv_acc, 'final test_acc_adv_mean_'+attack_name: test_adv_acc_mean, 'final test_loss_adv_'+attack_name: adv_loss,
                            'final test_loss_adv_mean_'+attack_name: loss.mean()})
                            wandb.log({'final train_acc_adv_'+attack_name: train_adv_acc,'final train_loss_adv_'+attack_name: train_adv_loss,
                            'final train_loss_adv_mean_'+attack_name: loss.mean()})
                    wandb.finish(quiet=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial robustness evaluation')
    parser.add_argument('--n_aug', type=int, default=20)
    parser.add_argument('--datasets', type=str, nargs='+', default=['STL10'])
    parser.add_argument('--attacks', type=str, nargs='+', default=['Gaussian_aug', 'Beta_aug' ])
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--algos', type=str, nargs='+', default=["Laplacian_DALE_PD_Reverse"])

    args = parser.parse_args()
    compute_metrics(args)
    