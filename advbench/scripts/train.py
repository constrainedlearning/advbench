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

from torch.optim.lr_scheduler import CosineAnnealingLR
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
    'MCMC_DALE_PD_Reverse',
    'KL_DALE_PD',
    'NUTS_DALE',
]

def main(args, hparams, test_hparams):
    #if args.dataset=="IMNET" or args.dataset=="MNIST":
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device
    print(f"Using {device}")
    hparams['model'] = args.model
    if args.perturbation=='SE':
        hparams['epsilon'] = torch.tensor([hparams[f'epsilon_{i}'] for i in ("rot","tx","ty")]).to(device)
        test_hparams['epsilon'] = torch.tensor([test_hparams[f'epsilon_{tfm}'] for tfm in ("rot","tx","ty")]).to(device)
    elif args.perturbation =='Translation':
        hparams['epsilon'] = torch.tensor([hparams[f'epsilon_{i}'] for i in ("tx","ty")]).to(device)
        test_hparams['epsilon'] = torch.tensor([test_hparams[f'epsilon_{tfm}'] for tfm in ("tx","ty")]).to(device)
    elif args.perturbation =='PointcloudTranslation':
        hparams['epsilon'] = torch.tensor([hparams['epsilon_tx'] for i in range(3)] + [hparams['epsilon_ty'] for i in range(3)]).to(device)
        test_hparams['epsilon'] = torch.tensor([test_hparams['epsilon_tx'] for i in range(3)] + [test_hparams['epsilon_ty'] for i in range(3)]).to(device)
    aug = not args.no_augmentation
    if args.auto_augment:
        dataset = vars(datasets)[args.dataset](args.data_dir, augmentation= aug, auto_augment=True)
    elif args.auto_augment_wo_translations:
        dataset = vars(datasets)[args.dataset](args.data_dir, augmentation= aug, auto_augment=True, exclude_translations=True)
    else:
        dataset = vars(datasets)[args.dataset](args.data_dir, augmentation= aug)
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
    if args.dataset in ["modelnet40", "scanobjectnn"]:
        adjust_lr = CosineAnnealingLR(algorithm.optimizer, dataset.N_EPOCHS, eta_min=dataset.MIN_LR, last_epoch=dataset.START_EPOCH - 1)
    else:
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
        name = f"{args.flags}{args.perturbation} {args.algorithm} {args.trial_seed} {args.seed}"
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
    best_acc = 0
    for epoch in range(0, dataset.N_EPOCHS):
        if wandb_log:
            wandb.log({'lr': adjust_lr.get_last_lr(), 'epoch': epoch, 'step':step})
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
        adjust_lr.step()
        
        val_clean_acc, val_clean_mean_acc = misc.accuracy_mean_overall(algorithm, val_ldr, device)
        if val_clean_acc>= best_acc:
            torch.save(algorithm.state_dict(), os.path.join(args.output_dir, f'{name}_best_ckpt.pkl'))            
        if wandb_log:
            wandb.log({'val_clean_oacc': val_clean_acc, 'val_clean_macc': val_clean_mean_acc, 'epoch': epoch, 'step':step})
        
        add_results_row([epoch, val_clean_acc, 'ERM', 'Val'])
        if epoch == dataset.N_EPOCHS-1:
            algorithm.load_state_dict(torch.load(os.path.join(args.output_dir, f'{name}_best_ckpt.pkl')))
            train_clean_acc, train_clean_mean_acc, train_clean_loss = misc.accuracy_mean_overall_loss(algorithm, train_ldr, device, max_batches=100)
            val_clean_acc, val_clean_mean_acc, val_clean_loss = misc.accuracy_mean_overall_loss(algorithm, val_ldr, device)
            test_clean_acc, test_clean_mean_acc, test_clean_loss = misc.accuracy_mean_overall_loss(algorithm, test_ldr, device)
            wandb.log({'train_clean_loss': train_clean_loss,'train_clean_acc': train_clean_acc, 'train_clean_acc_bal': train_clean_mean_acc, 'epoch': epoch, 'step':step})
            wandb.log({'best_val_clean_loss': val_clean_loss, 'best_val_clean_acc': val_clean_acc, 'best_val_clean_acc_bal': val_clean_mean_acc, 'epoch': epoch, 'step':step})
            wandb.log({'test_clean_loss': test_clean_loss,'test_clean_acc': test_clean_acc, 'test_clean_acc_bal': test_clean_mean_acc, 'epoch': epoch, 'step':step})
            #test_clean_acc_voting, test_clean_mean_acc_voting = voting(algorithm, test_ldr, device)
            #wandb.log({'test_clean_oacc_voting': test_clean_acc_voting, 'test_clean_macc_voting': test_clean_mean_acc_voting, 'epoch': epoch, 'step':step})
        if (epoch % dataset.ATTACK_INTERVAL == 0 and epoch>0) or epoch == dataset.N_EPOCHS-1:
            # Save model
            name = f"{args.flags}{args.perturbation} {args.algorithm} {args.model} {args.seed}"
            model_filepath = os.path.join(args.output_dir, f'{name}_ckpt.pkl')
            torch.save(algorithm.state_dict(), model_filepath)
            # Push it to wandb
            if wandb_log:
                wandb.save(model_filepath)
            # compute save and log adversarial accuracies on validation/test sets
            train_ldr_small = datasets.change_batch_size(train_ldr, 10)
            test_adv_accs = []
            for attack_name, attack in test_attacks.items():
                if args.n_eval>1:
                    attack_name = attack_name + "_Batch"
                print(attack_name)
                test_adv_acc, test_adv_acc_mean, test_adv_acc_bal, test_adv_acc_mean_bal, adv_loss, accs, loss, deltas = misc.adv_accuracy_loss_delta_balanced(algorithm, test_ldr, device, attack, augs_per_batch=args.n_eval)
                print("Test Adversarial Accuracy:", test_adv_acc)
                print("Test Balanced Adversarial Accuracy:", test_adv_acc_bal)
                train_adv_acc, train_adv_acc_mean, train_adv_acc_bal, train_adv_acc_mean_bal, adv_loss, train_accs, train_loss, train_deltas = misc.adv_accuracy_loss_delta_balanced(algorithm, train_ldr_small, device, attack, augs_per_batch=args.n_eval)
                print("Train Adversarial Accuracy:", test_adv_acc)
                print("Train Balanced Adversarial Accuracy:", test_adv_acc_bal)
                add_results_row([epoch, test_adv_acc, attack_name, 'Test'])
                add_results_row([epoch, train_adv_acc, attack_name, 'Train'])
                test_adv_accs.append(test_adv_acc)
                if wandb_log and args.perturbation!="Linf":
                    print(f"plotting and logging {attack_name}")
                    wandb.log({'test_adv_acc_'+attack_name: test_adv_acc,'test_adv_acc_mean_'+attack_name: test_adv_acc_mean, 'test_loss_adv_'+attack_name: loss.mean(),
                     'test_adv_acc_bal_'+attack_name: test_adv_acc_bal, 'test_adv_acc_mean_bal_'+attack_name: test_adv_acc_mean_bal, 'epoch': epoch, 'step':step})
                    wandb.log({'train_adv_acc_'+attack_name: train_adv_acc,'train_adv_acc_mean_'+attack_name: train_adv_acc_mean, 'train_loss_adv_'+attack_name: loss.mean(),
                     'train_adv_acc_bal_'+attack_name: train_adv_acc_bal, 'train_adv_acc_mean_bal_'+attack_name: train_adv_acc_mean_bal, 'epoch': epoch, 'step':step})
                    plotting.plot_perturbed_wandb(deltas, loss, name="test_loss_adv"+attack_name, wandb_args = {'epoch': epoch, 'step':step}, plot_mode="scatter")
                    plotting.plot_perturbed_wandb(deltas, accs, name="test_acc_adv"+attack_name, wandb_args = {'epoch': epoch, 'step':step}, plot_mode="scatter")
                    
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
        print(f'\nClean val. accuracy: {val_clean_acc:.3f}\t', end='')
        if (epoch % dataset.ATTACK_INTERVAL == 0 and epoch>0) or epoch == dataset.N_EPOCHS-1:
            for attack_name, acc in zip(test_attacks.keys(), test_adv_accs):
                print(f'{attack_name} val. accuracy: {acc:.3f}\t', end='')
        print('\n')

        # reset all meters
        meters_df = algorithm.meters_to_df(epoch)
        meters_df.to_pickle(os.path.join(args.output_dir, 'meters.pkl'))
        algorithm.reset_meters()

    

    wandb.finish()
    print("Finished")

    # save results dataframe to file
    results_df.to_pickle(os.path.join(args.output_dir, 'results.pkl'))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

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
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use')
    parser.add_argument('--log_imgs', action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.2)
    parser.add_argument('--auto_augment', action='store_true')
    parser.add_argument('--auto_augment_wo_translations', action='store_true')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--no_augmentation', action='store_false')
    parser.add_argument('--eps', type=float, default=0.0, help="Constraint level")
    parser.add_argument('--flags', type=str,default='', help='add to exp name')
    parser.add_argument('--n_eval', type=int, default=1, help='Number of transforms for evaluation')
    parser.add_argument('--penalty', type=float,default=0.0, help='Penalised regularisation coeff for adv loss')
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

    hparams['optimizer'] = args.optimizer
    hparams['label_smoothing'] = args.label_smoothing
    if args.eps > 0:
        hparams['l_dale_pd_inv_margin'] = args.eps
    if args.penalty > 0.0:
        hparams['adv_penalty'] = args.penalty
    if args.alpha[0] or args.beta[0]:
        for i, (alpha, beta) in enumerate(zip(args.alpha, args.beta)):
            hparams[f'beta_attack_alpha_{i}'] = args.alpha
            hparams[f'beta_attack_beta_{i}'] = args.beta
            if i>0 or "Beta_aug" not in args.test_attacks:
                args.test_attacks.append(f'Beta_aug')
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
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args, hparams, test_hparams)
