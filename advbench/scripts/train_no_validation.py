import argparse
from itertools import combinations_with_replacement
import torch
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    hparams['model'] = args.model
    if args.perturbation=='SE':
        hparams['epsilon'] = torch.tensor([hparams[f'epsilon_{i}'] for i in ("rot","tx","ty")]).to(device)
        test_hparams['epsilon'] = torch.tensor([test_hparams[f'epsilon_{tfm}'] for tfm in ("rot","tx","ty")]).to(device)
    dataset = vars(datasets)[args.dataset](args.data_dir)
    train_ldr, _, test_ldr = datasets.to_loaders(dataset, hparams)
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

    adjust_lr_dual = None if dataset.HAS_LR_SCHEDULE_DUAL is False or not (args.algorithm in PD_ALGORITHMS) else dataset.adjust_lr_dual

    test_attacks = {
        a: vars(attacks)[a](algorithm.classifier, test_hparams, device, perturbation=args.perturbation) for a in args.test_attacks}
    
    columns = ['Epoch', 'Accuracy', 'Eval-Method', 'Split', 'Train-Alg', 'Dataset', 'Trial-Seed', 'Output-Dir']
    results_df = pd.DataFrame(columns=columns)
    def add_results_row(data):
        defaults = [args.algorithm, args.dataset, args.trial_seed, args.output_dir]
        results_df.loc[len(results_df)] = data + defaults
    if wandb_log:
        name = f"{args.perturbation} {args.algorithm} {args.test_attacks} {args.trial_seed} {args.seed}"
        wandb.init(project=f"adversarial-constrained-{args.dataset}", name=name)
        wandb.config.update(args)
        wandb.config.update(hparams)
        train_eval, test_eval = [], []
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
        if adjust_lr_dual is not None:
            adjust_lr_dual(algorithm.pd_optimizer, epoch)
        if wandb_log:
            wandb.log({'lr': hparams['learning_rate'], 'epoch': epoch, 'step':step})
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
            break

        # save clean accuracies on validation/test sets
        test_clean_acc = misc.accuracy(algorithm, test_ldr, device)
        if wandb_log:
            wandb.log({'test_clean_acc': test_clean_acc, 'epoch': epoch, 'step':step})

        add_results_row([epoch, test_clean_acc, 'ERM', 'Test'])
        if epoch % dataset.ATTACK_INTERVAL == 0 and epoch > 0:
            # compute save and log adversarial accuracies on validation/test sets
            test_adv_accs = []
            for attack_name, attack in test_attacks.items():
                print(attack_name)
                test_adv_acc, loss, deltas = misc.adv_accuracy_loss_delta(algorithm, test_ldr, device, attack)
                print("Test Adv Acc:", test_adv_acc)
                add_results_row([epoch, test_adv_acc, attack_name, 'Test'])
                test_adv_accs.append(test_adv_acc)
                if wandb_log and args.perturbation!="Linf":
                    print(f"plotting and logging {attack_name}")
                    wandb.log({'test_acc_adv_'+attack_name: test_adv_acc, 'test_loss_adv_'+attack_name: loss.mean(), 'epoch': epoch, 'step':step})
                    plotting.plot_perturbed_wandb(deltas, loss, name="test_loss_adv"+attack_name, wandb_args = {'epoch': epoch, 'step':step}, plot_mode="scatter")
                
        if wandb_log and epoch % dataset.LOSS_LANDSCAPE_INTERVAL == 0 and epoch > 1:
        # log loss landscape
            print(f"plotting and logging loss landscape")
            for eval, split in zip([train_eval, test_eval], ['train', 'test']):
                for eval_dict in eval:    
                    deltas, loss = eval_dict["grid"].eval_perturbed(single_img=False)
                    tx, ty = eval_dict["tx"], eval_dict["ty"]
                    plotting.plot_perturbed_wandb(deltas[:, 0], loss, name=f"{split} angle vs loss ({tx},{ty})", wandb_args = {'epoch': epoch, 'step':step, 'tx':tx, 'ty':ty})
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
        print(f'\nClean val. accuracy: {test_clean_acc:.3f}\t', end='')
        if epoch % dataset.ATTACK_INTERVAL == 0 and epoch>0:
            for attack_name, acc in zip(test_attacks.keys(), test_adv_accs):
                print(f'{attack_name} val. accuracy: {acc:.3f}\t', end='')
        print('\n')

        # save results dataframe to file
        results_df.to_pickle(os.path.join(args.output_dir, 'results.pkl'))

        # reset all meters
        meters_df = algorithm.meters_to_df(epoch)
        meters_df.to_pickle(os.path.join(args.output_dir, 'meters.pkl'))
        algorithm.reset_meters()

    torch.save(
        {'model': algorithm.state_dict()}, 
        os.path.join(args.output_dir, f'ckpt.pkl'))

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

    main(args, hparams, test_hparams)