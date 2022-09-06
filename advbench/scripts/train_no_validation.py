import argparse
from itertools import combinations_with_replacement
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
try:
    import wandb
    wandb_log=True
except ImportError:
    wandb_log=False

PD_ALGORITHMS = [
    'Gaussian_DALE',
    'Laplacian_DALE',
    'Gaussian_DALE_PD',
    'Gaussian_DALE_PD_Reverse',
    'Laplacian_DALE_PD_Reverse',
    'MH_DALE_PD_Reverse',
    'KL_DALE_PD',
]

def main(args, hparams, test_hparams):
    device = args.device
    print(f"Using {device}")
    hparams['model'] = args.model
    if args.perturbation=='SE':
        hparams['epsilon'] = torch.tensor([hparams[f'epsilon_{i}'] for i in ("rot","tx","ty")]).to(device)
        test_hparams['epsilon'] = torch.tensor([test_hparams[f'epsilon_{tfm}'] for tfm in ("rot","tx","ty")]).to(device)
    elif args.perturbation=='Translation':
        hparams['epsilon'] = torch.tensor([hparams[f'epsilon_{i}'] for i in ("tx","ty")]).to(device)
        test_hparams['epsilon'] = torch.tensor([test_hparams[f'epsilon_{tfm}'] for tfm in ("tx","ty")]).to(device)
    aug = args.augment
    if args.auto_augment:
        dataset = vars(datasets)[args.dataset](args.data_dir, augmentation= aug, auto_augment=True)
    elif args.auto_augment_wo_translations:
        dataset = vars(datasets)[args.dataset](args.data_dir, augmentation= aug, auto_augment=True, exclude_translations=True)
    else:
        dataset = vars(datasets)[args.dataset](args.data_dir, augmentation= aug)
    if args.epochs>0:
        dataset.N_EPOCHS = args.epochs
    train_ldr, val_ldr, test_ldr = datasets.to_loaders(dataset, hparams, device=device)
    kw_args = {"perturbation": args.perturbation}
    if args.algorithm in PD_ALGORITHMS: 
        if args.algorithm.endswith("Reverse"):
            kw_args["init"] = 0.0
        else:
            kw_args["init"] = 1.0
    algorithm = vars(algorithms)[args.algorithm](
        dataset.INPUT_SHAPE, 
        dataset.NUM_CLASSES,
        hparams,
        device,
        **kw_args).to(device)
    adjust_lr = None if dataset.HAS_LR_SCHEDULE is False else dataset.adjust_lr
    try:
        summary(algorithm.classifier, input_size=dataset.INPUT_SHAPE, device=device)
    except:
        print("Model summary failed, currently does not support devices other than cpu or cuda.")

    test_attacks = {
        a: vars(attacks)[a](algorithm.classifier, test_hparams, device, perturbation=args.perturbation) for a in args.test_attacks if 'Beta_aug' not in a}
    
    if 'Beta_aug'in args.test_attacks:
        for alpha, beta in zip(args.alpha, args.beta):
            test_hparams['beta_attack_alpha'] = alpha
            test_hparams['beta_attack_beta'] =  beta
            test_attacks[f'Beta_aug_{alpha}_{beta}'] =  vars(attacks)['Beta_aug'](algorithm.classifier, test_hparams, device, perturbation=args.perturbation)
    
    columns = ['Epoch', 'Accuracy', 'Eval-Method', 'Split', 'Train-Alg', 'Dataset', 'Trial-Seed', 'Output-Dir']
    results_df = pd.DataFrame(columns=columns)
    def add_results_row(data):
        defaults = [args.algorithm, args.dataset, args.trial_seed, args.output_dir]
        results_df.loc[len(results_df)] = data + defaults
    if wandb_log:
        name = f"{args.flags}{args.perturbation} {args.algorithm} {args.model} {args.seed}"
        wandb.init(project=f"{args.project}-{args.dataset}", name=name)
        wandb.config.update(args)
        wandb.config.update(hparams)
        wandb.config.update({"test_"+key:val for key, val in test_hparams.items()})
        train_eval, test_eval = [], []
        if args.perturbation =="SE":
            translations = list(combinations_with_replacement(dataset.TRANSLATIONS, r=2))
            for tx, ty in translations:
                eval_dict = {"tx": tx, "ty": ty}
                eval_dict["grid"] = logging.AngleGrid(algorithm, train_ldr, max_perturbations=dataset.ANGLE_GSIZE, tx=tx, ty=ty)
                train_eval.append(eval_dict.copy())
                eval_dict["grid"] = logging.AngleGrid(algorithm, test_ldr, max_perturbations=dataset.ANGLE_GSIZE, tx=tx, ty=ty)
                test_eval.append(eval_dict.copy())

    total_time = 0
    step = 0
    for epoch in range(0, dataset.N_EPOCHS):
        if adjust_lr is not None:
            adjust_lr(algorithm.optimizer, epoch, hparams)
        if wandb_log:
            wandb.log({'lr': algorithm.optimizer.param_groups[0]['lr'], 'epoch': epoch, 'step':step})
        timer = meters.TimeMeter()
        epoch_start = time.time()
        for batch_idx, (imgs, labels) in enumerate(train_ldr):
            step+=imgs.shape[0]
            timer.batch_start()
            imgs, labels = imgs.to(device), labels.to(device)
            algorithm.step(imgs, labels)

            if batch_idx % dataset.LOG_INTERVAL == 0:
                print(f'Train epoch {epoch}/{dataset.N_EPOCHS} ', end='')
                print(f'({100. * batch_idx / len(train_ldr):.0f}%)]\t', end='')
                for name, meter in algorithm.meters.items():
                    if meter.print:
                        print(f'{name}: {meter.val:.3f} (avg. {meter.avg:.3f})\t', end='')
                        if wandb_log:
                            wandb.log({name+"_avg": meter.avg, 'epoch': epoch, 'step':step})
                print(f'Time: {timer.batch_time.val:.3f} (avg. {timer.batch_time.avg:.3f})')
            timer.batch_end()
        # save clean accuracies on validation/test sets
        if dataset.TEST_INTERVAL is None or epoch % dataset.TEST_INTERVAL == 0 or epoch == dataset.N_EPOCHS-1:
            test_clean_acc = misc.accuracy(algorithm, test_ldr, device)
            if wandb_log:
                wandb.log({'test_clean_acc': test_clean_acc, 'epoch': epoch, 'step':step})
            add_results_row([epoch, test_clean_acc, 'ERM', 'Test'])
            train_clean_acc = misc.accuracy(algorithm, val_ldr, device)
            if wandb_log:
                wandb.log({'train_clean_acc': train_clean_acc, 'epoch': epoch, 'step':step})
            add_results_row([epoch, train_clean_acc, 'ERM', 'Train'])

        if (epoch % dataset.ATTACK_INTERVAL == 0 and epoch>0) or epoch == dataset.N_EPOCHS-1:
            # Save model
            name = f"{args.flags}{args.perturbation} {args.algorithm} {args.model} {args.seed}"
            model_filepath = os.path.join(args.output_dir, f'{name}_ckpt.pkl')
            torch.save(algorithm.state_dict(), model_filepath)
            if wandb_log:
                # Push weights to wandb
                wandb.save(model_filepath, policy='now')
            # compute save and log adversarial accuracies on validation/test sets
            test_adv_accs = []
            for attack_name, attack in test_attacks.items():
                test_adv_acc, test_adv_acc_mean, adv_loss, accs, loss, deltas = misc.adv_accuracy_loss_delta(algorithm, test_ldr, device, attack, augs_per_batch=args.n_eval)
                add_results_row([epoch, test_adv_acc, attack_name, 'Test'])
                test_adv_accs.append(test_adv_acc)
                train_adv_acc, train_adv_acc_mean, train_adv_loss, train_accs, train_loss, train_deltas = misc.adv_accuracy_loss_delta(algorithm, val_ldr, device, attack, augs_per_batch=args.n_eval)
                add_results_row([epoch, train_adv_acc, attack_name, 'Train'])
                test_adv_accs.append(test_adv_acc)
                if wandb_log:
                    if args.n_eval>1:
                        attack_name = attack_name + "_Batch"
                    print(f"Logging {attack_name}...")
                    wandb.log({'test_acc_adv_'+attack_name: test_adv_acc,'mean_test_acc_adv_'+attack_name: test_adv_acc_mean, 'test_loss_adv_'+attack_name: adv_loss,
                    'test_loss_adv_mean_'+attack_name: loss.mean(), 'epoch': epoch, 'step':step})
                    wandb.log({'train_acc_adv_'+attack_name: train_adv_acc,'mean_train_acc_adv_'+attack_name: train_adv_acc_mean, 'train_loss_adv_'+attack_name: train_adv_loss,
                    'train_loss_adv_mean_'+attack_name: loss.mean(), 'epoch': epoch, 'step':step})
                    if args.perturbation!="Linf":
                        plotting.plot_perturbed_wandb(deltas, loss, name="test_loss_adv"+attack_name, wandb_args = {'epoch': epoch, 'step':step}, plot_mode="table")
                        plotting.plot_perturbed_wandb(deltas, accs, name="test_acc_adv"+attack_name, wandb_args = {'epoch': epoch, 'step':step}, plot_mode="table")
                        plotting.plot_perturbed_wandb(train_deltas, train_loss, name="train_loss_adv"+attack_name, wandb_args = {'epoch': epoch, 'step':step}, plot_mode="table")
                        plotting.plot_perturbed_wandb(train_deltas, train_accs, name="train_acc_adv"+attack_name, wandb_args = {'epoch': epoch, 'step':step}, plot_mode="table")
                    
                    if args.log_imgs:
                        imgs, labels = next(iter(test_ldr))
                        if algorithms.FFCV_AVAILABLE:
                            with autocast():
                                attacked = attack(imgs.to(device), labels.to(device))[0]
                        else:
                            attacked = attack(imgs.to(device), labels.to(device))[0]
                        for i in range(10):
                            if attacked.shape[0] > imgs.shape[0]:
                                og = imgs[0].to(device)
                            else:
                                og = imgs[i].to(device)
                            wb_pert = wandb.Image(torch.stack([attacked[i], og]), caption=f"Perturbed and original image {i} {attack_name}")
                            wandb.log({'Test image '+attack_name: wb_pert,'epoch': epoch, 'step':step})

        if args.perturbation =="SE" and wandb_log and ((epoch % dataset.LOSS_LANDSCAPE_INTERVAL == 0 and epoch>0) or epoch == dataset.N_EPOCHS-1):
        # log loss landscape
            print(f"plotting and logging loss landscape")
            for eval, split in zip([train_eval, test_eval], ['train', 'test']):
                for eval_dict in eval:    
                    deltas, loss, acc = eval_dict["grid"].eval_perturbed(single_img=False)
                    tx, ty = eval_dict["tx"], eval_dict["ty"]
                    plotting.plot_perturbed_wandb(deltas[:, 0], loss, name=f"{split} angle vs loss ({tx},{ty})", wandb_args = {'epoch': epoch, 'step':step, 'tx':tx, 'ty':ty})
                    plotting.plot_perturbed_wandb(deltas[:, 0], acc, name=f"{split} angle vs accuracy ({tx},{ty})", wandb_args = {'epoch': epoch, 'step':step, 'tx':tx, 'ty':ty})
                    plotting.plot_perturbed_wandb(deltas[:, 0], acc, name=f"{split} angle vs accuracy ({tx},{ty})", wandb_args = {'epoch': epoch, 'step':step, 'tx':tx, 'ty':ty})
        epoch_end = time.time()
        total_time += epoch_end - epoch_start

        # print results
        print(f'Epoch: {epoch+1}/{dataset.N_EPOCHS}\t', end='')
        print(f'Epoch time: {format_timespan(epoch_end - epoch_start)}\t', end='')
        print(f'Total time: {format_timespan(total_time)}\t', end='')
        print(f'Training alg: {args.algorithm}\t', end='')
        print(f'Dataset: {args.dataset}\t', end='')
        print(f'Path: {args.output_dir}')
        for name, meter in algorithm.meters.items():
            if meter.print:
                print(f'Avg. train {name}: {meter.avg:.3f}\t', end='')
        if dataset.TEST_INTERVAL is None or epoch % dataset.TEST_INTERVAL == 0:
            print(f'\nClean val. accuracy: {test_clean_acc:.3f}\t', end='')
        if (epoch % dataset.ATTACK_INTERVAL == 0 and epoch>0) or epoch == dataset.N_EPOCHS-1:
            for attack_name, acc in zip(test_attacks.keys(), test_adv_accs):
                print(f'{attack_name} val. accuracy: {acc:.3f}\t', end='')
        print('\n')

        # save results dataframe to file
        results_df.to_pickle(os.path.join(args.output_dir, 'results.pkl'))

        # reset all meters
        meters_df = algorithm.meters_to_df(epoch)
        meters_df.to_pickle(os.path.join(args.output_dir, 'meters.pkl'))
        algorithm.reset_meters()
    wandb.finish(quiet=True)
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial robustness evaluation')
    parser.add_argument('--data_dir', type=str, default='./advbench/data')
    parser.add_argument('--output_dir', type=str, default='train_output')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default='ERM', help='Algorithm to run')
    parser.add_argument('--perturbation', type=str, default='Linf', help=' Linf or Rotation')
    parser.add_argument('--test_attacks', type=str, nargs='+', default=['PGD_Linf'])
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for hyperparameters')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--n_eval', type=int, default=1, help='Number of transforms for evaluation')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use')
    parser.add_argument('--log_imgs', action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--auto_augment', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--auto_augment_wo_translations', action='store_true')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--eps', type=float, default=0.0, help="Constraint level")
    parser.add_argument('--flags', type=str,default='', help='add to exp name')
    parser.add_argument('--epochs', type=int,default=0, help='custom number of epochs, use defaults if 0')
    parser.add_argument('--max_rot', type=int,default=0, help='max angle in degrees')
    parser.add_argument('--max_trans', type=int,default=0, help='max translation in pixels')
    parser.add_argument('--penalty', type=float,default=1.0, help='Penalised regularisation coeff for adv loss')
    parser.add_argument('--beta', type=float, nargs='+', default=[0.0], help='Beta distribution beta coefficient')
    parser.add_argument('--alpha', type=float,nargs='+', default=[0.0], help='Beta distribution alpha coefficient')
    parser.add_argument('--project', type=str, default='DAug', help='wandb-project')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.dataset not in vars(datasets):
        raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.perturbation, args.dataset)
    else:
        seed = misc.seed_hash(args.hparams_seed, args.trial_seed)
        hparams = hparams_registry.random_hparams(args.algorithm, args.perturbation, args.dataset, seed)
    if args.eps > 0:
        hparams['l_dale_pd_inv_margin'] = args.eps

    if args.max_rot > 0:
        hparams['epsilon_rot'] = args.max_rot
    if args.max_trans > 0:
        hparams['epsilon_tx'] = args.max_trans
        hparams['epsilon_ty'] = args.max_trans

    if args.penalty > 0.0:
        hparams['adv_penalty'] = args.penalty
    if args.alpha[0] or args.beta[0]:
        for i, (alpha, beta) in enumerate(zip(args.alpha, args.beta)):
            hparams[f'beta_attack_alpha_{i}'] = args.alpha
            hparams[f'beta_attack_beta_{i}'] = args.beta
            if i>0 or "Beta_aug" not in args.test_attacks:
                args.test_attacks.append(f'Beta_aug')

    hparams['optimizer'] = args.optimizer
    hparams['label_smoothing'] = args.label_smoothing
    print ('Hparams:')
    for k, v in sorted(hparams.items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=2)

    test_hparams = hparams_registry.test_hparams(args.algorithm, args.perturbation, args.dataset)

    print('Test hparams:')
    for k, v in sorted(test_hparams.items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'test_hparams.json'), 'w') as f:
        json.dump(test_hparams, f, indent=2)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    wandb.setup()
    main(args, hparams, test_hparams)
    exit()
