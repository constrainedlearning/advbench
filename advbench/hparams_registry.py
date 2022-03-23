import numpy as np

from advbench.lib import misc
from advbench import datasets

def default_hparams(algorithm, perturbation, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, perturbation, dataset, 0).items()}

def random_hparams(algorithm, perturbation, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, perturbation, dataset, seed).items()}

def _hparams(algorithm: str, perturbation:str, dataset: str, random_seed: int):
    """Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""

        assert(name not in hparams)
        random_state = np.random.RandomState(misc.seed_hash(random_seed, name))
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    if dataset == 'IMNET':
        _hparam('batch_size', 64, lambda r: int(2 ** r.uniform(3, 8)))
    else:
        _hparam('batch_size', 128, lambda r: int(2 ** r.uniform(3, 8)))
    _hparam('augmentation_prob', 0.5, lambda r: 0.5)
    _hparam('perturbation_batch_size', 10, lambda r: 10)
    _hparam('mcmc_dale_scale', 0.05, lambda r: 0.05)
    _hparam('mcmc_dale_n_steps', 5, lambda r: 5)
    _hparam('mcmc_proposal', 'Laplace', lambda r: 'Laplace')
    _hparam('gaussian_attack_std', 1, lambda r: 1 )
    _hparam('laplacian_attack_std', 1, lambda r: 1 )

    if dataset == 'IMNET':
        _hparam('label_smoothing', 0.1, lambda r: 0.1 )
    else:
        _hparam('label_smoothing', 0.0, lambda r: 0.0)


    # optimization
    if dataset == 'MNIST':
        _hparam('learning_rate', 0.01, lambda r: 10 ** r.uniform(-1.5, -0.5))
        _hparam('lr_decay_start', 15, lambda r: 15)
        _hparam('lr_decay_factor', 0.8, lambda r: r.uniform(0.1, 0.3))
        _hparam('lr_decay_epoch', 1, lambda r: 1)
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
        _hparam('learning_rate', 0.1, lambda r: 10 ** r.uniform(-2, -1))
    if dataset == 'CIFAR100':
        _hparam('lr_decay_start', 0, lambda r: 0)
        _hparam('lr_decay_factor', 0.2, lambda r: r.uniform(0.1, 0.3))
        _hparam('lr_decay_epoch', 60, lambda r: 60)
    if dataset == 'IMNET':
        _hparam('learning_rate', 0.01, lambda r: 10 ** r.uniform(-1.5, -0.5))
        _hparam('lr_decay_start', 0, lambda r: 0)
        _hparam('lr_decay_factor', 0.8, lambda r: r.uniform(0.1, 0.3))
        _hparam('lr_decay_epoch', 1, lambda r: 1)
    _hparam('sgd_momentum', 0.9, lambda r: r.uniform(0.8, 0.95))
    _hparam('weight_decay', 5e-4, lambda r: 10 ** r.uniform(-6, -3))
    if perturbation == 'Linf':
        if dataset == 'MNIST':
            _hparam('epsilon', 0.3, lambda r: 0.3)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('epsilon', 0.031, lambda r: 0.031)
    elif perturbation == 'Rotation':
        if dataset == 'MNIST':
            _hparam('epsilon', 30, lambda r: 30)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('epsilon', 30, lambda r: 30)

    # Algorithm specific
    if perturbation == 'Linf':
        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 7, lambda r: 7)
            _hparam('pgd_step_size', 0.1, lambda r: 0.1)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('pgd_n_steps', 10, lambda r: 10)
            _hparam('pgd_step_size', 0.007, lambda r: 0.007)

        ##### TRADES #####
        if dataset == 'MNIST':
            _hparam('trades_n_steps', 7, lambda r: 7)
            _hparam('trades_step_size', 0.1, lambda r: r.uniform(0.01, 0.1))
            _hparam('trades_beta', 1.0, lambda r: r.uniform(0.1, 10.0))
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('trades_n_steps', 10, lambda r: 15)
            _hparam('trades_step_size', 2/255., lambda r: r.uniform(0.01, 0.1))
            _hparam('trades_beta', 6.0, lambda r: r.uniform(0.1, 10.0))

        ##### MART #####
        if dataset == 'MNIST':
            _hparam('mart_beta', 5.0, lambda r: r.uniform(0.1, 10.0))
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('mart_beta', 5.0, lambda r: r.uniform(0.1, 10.0))

        ##### Gaussian DALE #####
        if dataset == 'MNIST':
            _hparam('g_dale_n_steps', 15, lambda r: 15)
            _hparam('g_dale_step_size', 10, lambda r: 10)
            _hparam('g_dale_noise_coeff', 0.001, lambda r: 10 ** r.uniform(-6.0, -2.0))
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('g_dale_n_steps', 10, lambda r: 10)
            _hparam('g_dale_step_size', 0.007, lambda r: 0.007)
            _hparam('g_dale_noise_coeff', 0, lambda r: 0)
        _hparam('g_dale_nu', 0.1, lambda r: 0.1)
        _hparam('g_dale_eta', 0.007, lambda r: 0.007)

        # DALE (Laplacian-HMC)
        if dataset == 'MNIST':
            _hparam('l_dale_n_steps', 7, lambda r: 7)
            _hparam('l_dale_step_size', 0.1, lambda r: 0.1)
            _hparam('l_dale_noise_coeff', 0.001, lambda r: 10 ** r.uniform(-6.0, -2.0))
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('l_dale_n_steps', 10, lambda r: 10)
            _hparam('l_dale_step_size', 0.007, lambda r: 0.007)
            _hparam('l_dale_noise_coeff', 1e-2, lambda r: 1e-2)
        _hparam('l_dale_nu', 0.1, lambda r: 0.1)
        _hparam('l_dale_eta', 0.007, lambda r: 0.007)

        # DALE-PD (Gaussian-HMC)
        _hparam('g_dale_pd_step_size', 0.001, lambda r: 0.001)
        _hparam('g_dale_pd_eta', 0.001, lambda r: 0.001)
        _hparam('g_dale_pd_margin', 0.1, lambda r: 0.1)

        # DALE NUTS
        if dataset == 'MNIST':
            _hparam('n_dale_n_steps', 10, lambda r: 10)
            _hparam('n_dale_step_size', 0.1, lambda r: 0.1)
            _hparam('n_burn', 3, lambda r: 3)

    elif perturbation == 'Rotation':
        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 20, lambda r: 20)
            _hparam('pgd_step_size', 1, lambda r: 1)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('pgd_n_steps', 20, lambda r: 20)
            _hparam('pgd_step_size', 10, lambda r: 10)

        ##### TRADES #####
        if dataset == 'MNIST':
            _hparam('trades_n_steps', 15, lambda r: 15)
            _hparam('trades_step_size', 2, lambda r: r.uniform(0.2, 2))
            _hparam('trades_beta', 1.0, lambda r: r.uniform(0.1, 10.0))
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('trades_n_steps', 10, lambda r: 15)
            _hparam('trades_step_size', 2/255., lambda r: r.uniform(0.01, 0.1))
            _hparam('trades_beta', 6.0, lambda r: r.uniform(0.1, 10.0))

        ##### MART #####
        if dataset == 'MNIST':
            _hparam('mart_beta', 5.0, lambda r: r.uniform(0.1, 10.0))
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('mart_beta', 5.0, lambda r: r.uniform(0.1, 10.0))

        ##### Gaussian DALE #####
        if dataset == 'MNIST':
            _hparam('g_dale_n_steps', 20, lambda r: 1)
            _hparam('g_dale_step_size', 10, lambda r: 10)
            _hparam('g_dale_noise_coeff', 1, lambda r: 10 ** r.uniform(-1.0, 1.0))
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('g_dale_n_steps', 10, lambda r: 10)
            _hparam('g_dale_step_size', 0.007, lambda r: 0.007)
            _hparam('g_dale_noise_coeff', 0, lambda r: 0)
        _hparam('g_dale_nu', 0.1, lambda r: 0.1)
        _hparam('g_dale_eta', 0.1, lambda r: 0.1)

        # DALE (Laplacian-HMC)
        if dataset == 'MNIST':
            _hparam('l_dale_n_steps', 15, lambda r: 15)
            _hparam('l_dale_step_size', 2, lambda r: 2)
            _hparam('l_dale_noise_coeff', 1, lambda r: 10 ** r.uniform(-1.0, -1.0))
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('l_dale_n_steps', 10, lambda r: 10)
            _hparam('l_dale_step_size', 0.007, lambda r: 0.007)
            _hparam('l_dale_noise_coeff', 1e-2, lambda r: 1e-2)
        _hparam('l_dale_nu', 0.1, lambda r: 0.1)
        _hparam('l_dale_eta', 0.1, lambda r: 0.1)

        # Discrete Dale
        if dataset == 'MNIST':
            _hparam('l_dale_n_steps', 15, lambda r: 15)
            _hparam('l_dale_step_size', 2, lambda r: 2)
            _hparam('l_dale_noise_coeff', 1, lambda r: 10 ** r.uniform(-1.0, -1.0))
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('l_dale_n_steps', 10, lambda r: 10)
            _hparam('l_dale_step_size', 0.007, lambda r: 0.007)
            _hparam('l_dale_noise_coeff', 1e-2, lambda r: 1e-2)

        # DALE-PD (Gaussian-HMC)
        _hparam('g_dale_pd_step_size', 2, lambda r: 2)
        _hparam('g_dale_pd_eta', 0.01, lambda r: 0.01)
        _hparam('g_dale_pd_margin', 1.45, lambda r: 1.45)

        # DALE-PD-INV (Gaussian-HMC)
        _hparam('g_dale_pd_inv_step_size', 2, lambda r: 2)
        _hparam('g_dale_pd_inv_eta', 0.01, lambda r: 0.01)
        _hparam('g_dale_pd_inv_margin', 0.1469, lambda r: 0.1469)



       # Worst of K
        _hparam('worst_of_k_steps', 10, lambda r: 10)

        # DALE NUTS
        _hparam('n_dale_n_steps', 15, lambda r: 15)
        _hparam('n_dale_step_size', 2, lambda r: 2)
        _hparam('n_burn', 3, lambda r: 3)
       
        # Grid Search
        _hparam('grid_size', 10, lambda r: 10)

    elif perturbation=='SE':
        ##### Worst of K ######
        _hparam('worst_of_k_steps', 10, lambda r:10)

        _hparam('epsilon_rot', 30, lambda r:30)
        _hparam('epsilon_tx', 3, lambda r:3)
        _hparam('epsilon_ty', 3, lambda r:3)
        ##### PGD #####
        _hparam('pgd_n_steps', 30, lambda r: 30)
        _hparam('pgd_step_size', 0.1, lambda r: 0.1)

        ##### TRADES #####
        _hparam('trades_n_steps', 15, lambda r: 15)
        _hparam('trades_step_size', 2, lambda r: r.uniform(0.2, 2))
        _hparam('trades_beta', 1.0, lambda r: r.uniform(0.1, 10.0))

        ##### MART #####
        _hparam('mart_beta', 5.0, lambda r: r.uniform(0.1, 10.0))

        ##### Gaussian DALE #####
        _hparam('g_dale_n_steps', 30, lambda r: 30)
        _hparam('g_dale_step_size', 1, lambda r: 1)
        _hparam('g_dale_noise_coeff', 1, lambda r: 10 ** r.uniform(-1.0, 1.0))
        _hparam('g_dale_nu', 0.1, lambda r: 0.1)
        _hparam('g_dale_eta', 0.0001, lambda r: 0.0001)

        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 30, lambda r: 30)
        _hparam('l_dale_step_size', 0.05, lambda r: 10 ** r.uniform(-2.0, -0.5))
        _hparam('l_dale_noise_coeff', 0.02,lambda r: 10 ** r.uniform(-3.0, -1.5))
        _hparam('l_dale_nu', 0.1, lambda r: 0.1)
        _hparam('l_dale_eta', 0.001, lambda r: 0.001)

        # DALE-PD (Gaussian-HMC)
        _hparam('g_dale_pd_step_size', 2, lambda r: 2)
        _hparam('g_dale_pd_eta', 0.0001, lambda r: 0.0001)
        _hparam('g_dale_pd_margin', 0.16, lambda r: 0.16)

        # DALE-PD-INV (Gaussian-HMC)
        _hparam('g_dale_pd_inv_step_size', 0.1, lambda r: 10 ** r.uniform(-2.0, -0.5))
        _hparam('g_dale_pd_inv_eta', 0.0001, lambda r: 0.0001)
        _hparam('g_dale_pd_inv_margin', 0.2, lambda r: 0.2)

        # DALE-PD-INV (Laplacian-HMC)
        if dataset == 'MNIST':
            _hparam('l_dale_pd_inv_step_size', 0.05, lambda r: 0.05)
            _hparam('l_dale_pd_inv_eta', 0.0008, lambda r: 0.0008)
            _hparam('l_dale_pd_inv_margin', 0.14, lambda r: 0.14)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('l_dale_pd_inv_step_size', 1, lambda r: 1)
            _hparam('l_dale_pd_inv_eta', 0.00005, lambda r: 0.00005)
            _hparam('l_dale_pd_inv_margin', 0.3, lambda r: 0.3)
        
        # Discrete DALE-PD-INV
        _hparam('d_num_translations', 3, lambda r: 3)
        _hparam('d_num_rotations', 20, lambda r: 20)
        if dataset == 'MNIST':
            _hparam('d_dale_pd_inv_step_size', 0.05, lambda r: 0.05)
            _hparam('d_dale_pd_inv_eta', 0.0008, lambda r: 0.0008)
            _hparam('d_dale_pd_inv_margin', 0.14, lambda r: 0.14)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('d_dale_pd_inv_step_size', 1, lambda r: 1)
            _hparam('d_dale_pd_inv_eta', 0.00005, lambda r: 0.00005)
            _hparam('d_dale_pd_inv_margin', 0.15, lambda r: 0.15)

        # DALE NUTS
        _hparam('n_dale_n_steps', 15, lambda r: 15)
        _hparam('n_dale_step_size', 2, lambda r: 2)
        _hparam('n_burn', 3, lambda r: 3)
       
        # Grid Search
        _hparam('grid_size', 120, lambda r: 120)
    elif perturbation=='Translation':
        ##### Worst of K ######
        _hparam('worst_of_k_steps', 10, lambda r:10)
        _hparam('epsilon_tx', 3, lambda r:4)
        _hparam('epsilon_ty', 3, lambda r:4)
        ##### PGD #####
        _hparam('pgd_n_steps', 20, lambda r: 20)
        _hparam('pgd_step_size', 0.1, lambda r: 0.1)

        ##### TRADES #####
        _hparam('trades_n_steps', 15, lambda r: 15)
        _hparam('trades_step_size', 2, lambda r: r.uniform(0.2, 2))
        _hparam('trades_beta', 1.0, lambda r: r.uniform(0.1, 10.0))

        ##### MART #####
        _hparam('mart_beta', 5.0, lambda r: r.uniform(0.1, 10.0))

        ##### Gaussian DALE #####
        _hparam('g_dale_n_steps', 10, lambda r: 10)
        _hparam('g_dale_step_size', 1, lambda r: 1)
        _hparam('g_dale_noise_coeff', 1, lambda r: 10 ** r.uniform(-1.0, 1.0))
        _hparam('g_dale_nu', 0.1, lambda r: 0.1)
        _hparam('g_dale_eta', 0.0001, lambda r: 0.0001)

        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 10, lambda r: 10)
        _hparam('l_dale_step_size', 0.4, lambda r: 10 ** r.uniform(-2.0, -0.5))
        _hparam('l_dale_noise_coeff', 0.02,lambda r: 10 ** r.uniform(-3.0, -1.5))
        _hparam('l_dale_nu', 0.1, lambda r: 0.1)
        _hparam('l_dale_eta', 0.001, lambda r: 0.001)

        # DALE-PD (Gaussian-HMC)
        _hparam('g_dale_pd_step_size', 2, lambda r: 2)
        _hparam('g_dale_pd_eta', 0.0001, lambda r: 0.0001)
        _hparam('g_dale_pd_margin', 0.16, lambda r: 0.16)

        # DALE-PD-INV (Gaussian-HMC)
        _hparam('g_dale_pd_inv_step_size', 0.4, lambda r: 10 ** r.uniform(-2.0, -0.5))
        _hparam('g_dale_pd_inv_eta', 0.0001, lambda r: 0.0001)
        _hparam('g_dale_pd_inv_margin', 0.2, lambda r: 0.2)

        # DALE-PD-INV (Laplacian-HMC)
        if dataset == 'MNIST':
            _hparam('l_dale_pd_inv_step_size', 0.4, lambda r: 0.4)
            _hparam('l_dale_pd_inv_eta', 0.0008, lambda r: 0.0008)
            _hparam('l_dale_pd_inv_margin', 0.03, lambda r: 0.03)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('l_dale_pd_inv_step_size', 0.4, lambda r: 0.4)
            _hparam('l_dale_pd_inv_eta', 0.00005, lambda r: 0.00005)
            _hparam('l_dale_pd_inv_margin', 0.08, lambda r: 0.08)
        
        # Discrete DALE-PD-INV
        _hparam('d_num_translations', 3, lambda r: 3)
        _hparam('d_num_rotations', 20, lambda r: 20)
        if dataset == 'MNIST':
            _hparam('d_dale_pd_inv_step_size', 0.05, lambda r: 0.05)
            _hparam('d_dale_pd_inv_eta', 0.0008, lambda r: 0.0008)
            _hparam('d_dale_pd_inv_margin', 0.14, lambda r: 0.14)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('d_dale_pd_inv_step_size', 0.2, lambda r: 0.2)
            _hparam('d_dale_pd_inv_eta', 0.00005, lambda r: 0.00005)
            _hparam('d_dale_pd_inv_margin', 0.08, lambda r: 0.08)

        # DALE NUTS
        _hparam('n_dale_n_steps', 15, lambda r: 15)
        _hparam('n_dale_step_size', 2, lambda r: 2)
        _hparam('n_burn', 3, lambda r: 3)
       
        # Grid Search
        _hparam('grid_size', 120, lambda r: 120)

    elif perturbation=='CPAB':
        _hparam('tesselation', 20, lambda r:20)
        _hparam('epsilon', 1.2, lambda r: 1.2)

        ##### Worst of K ######
        _hparam('worst_of_k_steps', 10, lambda r:10)
        
        ##### PGD #####
        _hparam('pgd_n_steps', 10, lambda r: 10)
        _hparam('pgd_step_size', 1, lambda r: 1)

        ##### TRADES #####
        _hparam('trades_n_steps', 15, lambda r: 15)
        _hparam('trades_step_size', 2, lambda r: r.uniform(0.2, 2))
        _hparam('trades_beta', 1.0, lambda r: r.uniform(0.1, 10.0))

        ##### MART #####
        _hparam('mart_beta', 5.0, lambda r: r.uniform(0.1, 10.0))

        ##### Gaussian DALE #####
        _hparam('g_dale_n_steps', 10, lambda r: 10)
        _hparam('g_dale_step_size', 1, lambda r: 1)
        _hparam('g_dale_noise_coeff', 1, lambda r: 10 ** r.uniform(-1.0, 1.0))
        _hparam('g_dale_nu', 0.1, lambda r: 0.1)
        _hparam('g_dale_eta', 0.0001, lambda r: 0.0001)

        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 10, lambda r: 10)
        _hparam('l_dale_step_size', 0.05, lambda r: 10 ** r.uniform(-2.0, -0.5))
        _hparam('l_dale_noise_coeff', 0.02,lambda r: 10 ** r.uniform(-3.0, -1.5))
        _hparam('l_dale_nu', 0.1, lambda r: 0.1)
        _hparam('l_dale_eta', 0.001, lambda r: 0.001)

        # DALE-PD (Gaussian-HMC)
        _hparam('g_dale_pd_step_size', 2, lambda r: 2)
        _hparam('g_dale_pd_eta', 0.0001, lambda r: 0.0001)
        _hparam('g_dale_pd_margin', 0.01, lambda r: 0.01)

        # DALE-PD-INV (Gaussian-HMC)
        _hparam('g_dale_pd_inv_step_size', 0.1, lambda r: 10 ** r.uniform(-2.0, -0.5))
        _hparam('g_dale_pd_inv_eta', 0.0001, lambda r: 0.0001)
        _hparam('g_dale_pd_inv_margin', 0.01, lambda r: 0.01)

        # DALE-PD-INV (Laplacian-HMC)
        if dataset == 'MNIST':
            _hparam('l_dale_pd_inv_step_size', 0.05, lambda r: 0.05)
            _hparam('l_dale_pd_inv_eta', 0.0008, lambda r: 0.0008)
            _hparam('l_dale_pd_inv_margin',  0.01, lambda r: 0.01)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('l_dale_pd_inv_step_size', 0.1, lambda r: 0.1)
            _hparam('l_dale_pd_inv_eta', 0.00005, lambda r: 0.00005)
            _hparam('l_dale_pd_inv_margin',  0.01, lambda r: 0.01)
        
        # Discrete DALE-PD-INV
        _hparam('d_num_translations', 3, lambda r: 3)
        _hparam('d_num_rotations', 20, lambda r: 20)
        if dataset == 'MNIST':
            _hparam('d_dale_pd_inv_step_size', 0.05, lambda r: 0.05)
            _hparam('d_dale_pd_inv_eta', 0.0008, lambda r: 0.0008)
            _hparam('d_dale_pd_inv_margin', 0.01, lambda r: 0.01)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('d_dale_pd_inv_step_size', 1, lambda r: 1)
            _hparam('d_dale_pd_inv_eta', 0.00005, lambda r: 0.00005)
            _hparam('d_dale_pd_inv_margin', 0.01, lambda r: 0.01)

        # DALE NUTS
        _hparam('n_dale_n_steps', 15, lambda r: 15)
        _hparam('n_dale_step_size', 2, lambda r: 2)
        _hparam('n_burn', 3, lambda r: 3)
       
        # Grid Search
        _hparam('grid_size', 120, lambda r: 120)

    elif perturbation=='Crop' or perturbation=='Crop_and_Flip':
        _hparam('epsilon', 4, lambda r: 4)

        ##### Worst of K ######
        _hparam('worst_of_k_steps', 10, lambda r:10)

        
        ##### PGD #####
        _hparam('pgd_n_steps', 5, lambda r: 5)
        _hparam('pgd_step_size', 0.2, lambda r: 0.2)

        ##### TRADES #####
        _hparam('trades_n_steps', 15, lambda r: 15)
        _hparam('trades_step_size', 2, lambda r: r.uniform(0.2, 2))
        _hparam('trades_beta', 1.0, lambda r: r.uniform(0.1, 10.0))

        ##### MART #####
        _hparam('mart_beta', 5.0, lambda r: r.uniform(0.1, 10.0))

        ##### Gaussian DALE #####
        _hparam('g_dale_n_steps', 10, lambda r: 10)
        _hparam('g_dale_step_size', 1, lambda r: 1)
        _hparam('g_dale_noise_coeff', 1, lambda r: 10 ** r.uniform(-1.0, 1.0))
        _hparam('g_dale_nu', 0.1, lambda r: 0.1)
        _hparam('g_dale_eta', 0.0001, lambda r: 0.0001)

        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 10, lambda r: 10)
        _hparam('l_dale_step_size', 0.5, lambda r: 10 ** r.uniform(-2.0, -0.5))
        _hparam('l_dale_noise_coeff', 0.02,lambda r: 10 ** r.uniform(-3.0, -1.5))
        _hparam('l_dale_nu', 0.5, lambda r: 0.5)
        _hparam('l_dale_eta', 0.00001, lambda r: 0.00001)

        # DALE-PD (Gaussian-HMC)
        _hparam('g_dale_pd_step_size', 2, lambda r: 2)
        _hparam('g_dale_pd_eta', 0.0001, lambda r: 0.0001)
        _hparam('g_dale_pd_margin', 0.1, lambda r: 0.1)

        # DALE-PD-INV (Gaussian-HMC)
        _hparam('g_dale_pd_inv_step_size', 0.1, lambda r: 10 ** r.uniform(-2.0, -0.5))
        _hparam('g_dale_pd_inv_eta', 0.0001, lambda r: 0.0001)
        _hparam('g_dale_pd_inv_margin', 0.2, lambda r: 0.2)

        # DALE-PD-INV (Laplacian-HMC)
        if dataset == 'MNIST':
            _hparam('l_dale_pd_inv_step_size', 0.05, lambda r: 0.05)
            _hparam('l_dale_pd_inv_eta', 0.0008, lambda r: 0.0008)
            _hparam('l_dale_pd_inv_margin',  0.1, lambda r: 0.1)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('l_dale_pd_inv_step_size', 0.1, lambda r: 0.1)
            _hparam('l_dale_pd_inv_eta', 0.000001, lambda r: 0.000001)
            _hparam('l_dale_pd_inv_margin',  0.15, lambda r: 0.15)
        
        # Discrete DALE-PD-INV
        if dataset == 'MNIST':
            _hparam('d_dale_pd_inv_step_size', 0.05, lambda r: 0.05)
            _hparam('d_dale_pd_inv_eta', 0.0008, lambda r: 0.0008)
            _hparam('d_dale_pd_inv_margin', 0.01, lambda r: 0.01)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('d_dale_pd_inv_step_size', 1, lambda r: 1)
            _hparam('d_dale_pd_inv_eta', 0.00005, lambda r: 0.00005)
            _hparam('d_dale_pd_inv_margin', 0.3, lambda r: 0.3)

        # DALE NUTS
        _hparam('n_dale_n_steps', 15, lambda r: 15)
        _hparam('n_dale_step_size', 2, lambda r: 2)
        _hparam('n_burn', 3, lambda r: 3)
       
        # Grid Search
        _hparam('grid_size', 120, lambda r: 120)

    else:
        raise NotImplementedError

    return hparams

def test_hparams(algorithm: str, perturbation:str, dataset: str):

    hparams = {}

    def _hparam(name, default_val):
        """Define a hyperparameter for test adversaries."""

        assert(name not in hparams)
        hparams[name] = default_val
    if dataset=="MNIST":
        _hparam('perturbation_batch_size', 10)
    if dataset=="CIFAR10" or dataset=="CIFAR100":
        _hparam('perturbation_batch_size', 10)
    if dataset=="IMNET":
        _hparam('perturbation_batch_size', 20)
    _hparam('gaussian_attack_std', 0.5)
    _hparam('laplacian_attack_std', 0.5)

    if perturbation == 'Linf':
        if dataset == 'MNIST':
            _hparam('epsilon', 0.3)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('epsilon', 8/255.)
        
        ##### Worst of K ######
        _hparam('worst_of_k_steps', 10)

        ###### MCMC ###########
        _hparam('mcmc_dale_scale', 0.2)
        _hparam('mcmc_dale_n_steps', 10)
        _hparam('mcmc_proposal', 'Laplace')

        

        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 10)
            _hparam('pgd_step_size', 0.1)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('pgd_n_steps', 10)
            _hparam('pgd_step_size', 0.003)

        ##### TRADES #####
        if dataset == 'MNIST':
            _hparam('trades_n_steps', 10)
            _hparam('trades_step_size', 0.1)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('trades_n_steps', 20)
            _hparam('trades_step_size', 2/255.)
    elif perturbation=='Rotation':
        ##### Worst of K ######
        _hparam('worst_of_k_steps', 100)

        if dataset == 'MNIST':
            _hparam('epsilon', 30)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('epsilon', 20)

        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 10)
            _hparam('pgd_step_size', 2)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('pgd_n_steps', 10)
            _hparam('pgd_step_size', 2)

        ##### TRADES #####
        if dataset == 'MNIST':
            _hparam('trades_n_steps', 40)
            _hparam('trades_step_size', 5)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('trades_n_steps', 40)
            _hparam('trades_step_size', 5)
        
        # Grid Search
        _hparam('grid_size', 120)
    
    elif perturbation=='Translation':
        ##### Worst of K ######
        _hparam('worst_of_k_steps', 10)
        _hparam('epsilon_tx', 4)
        _hparam('epsilon_ty', 4)

        ###### MCMC ###########
        _hparam('mcmc_dale_scale', 0.2)
        _hparam('mcmc_dale_n_steps', 10)
        _hparam('mcmc_proposal', 'Laplace')
        

        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 30)
            _hparam('pgd_step_size', 0.5)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('pgd_n_steps', 20)
            _hparam('pgd_step_size', 0.2)

        ##### TRADES #####
        if dataset == 'MNIST':
            _hparam('trades_n_steps', 40)
            _hparam('trades_step_size', 5)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('trades_n_steps', 40)
            _hparam('trades_step_size', 5)
        
        # Grid Search
        _hparam('grid_size', 120)
        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 20)
        _hparam('l_dale_step_size', 0.2)
        _hparam('l_dale_noise_coeff', 0.2)
        _hparam('l_dale_nu', 0.1)

    elif perturbation=='SE':
        ##### Worst of K ######
        _hparam('worst_of_k_steps', 10)

        _hparam('epsilon_rot', 30)
        _hparam('epsilon_tx', 3)
        _hparam('epsilon_ty', 3)

        ###### MCMC ###########
        _hparam('mcmc_dale_scale', 0.5)
        _hparam('mcmc_dale_n_steps', 40)
        _hparam('mcmc_proposal', 'Laplace')
        

        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 40)
            _hparam('pgd_step_size', 0.5)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('pgd_n_steps', 10)
            _hparam('pgd_step_size', 0.5)

        ##### TRADES #####
        if dataset == 'MNIST':
            _hparam('trades_n_steps', 40)
            _hparam('trades_step_size', 5)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('trades_n_steps', 40)
            _hparam('trades_step_size', 5)
        
        # Grid Search
        _hparam('grid_size', 120)
        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 40)
        _hparam('l_dale_step_size', 0.5)
        _hparam('l_dale_noise_coeff', 0.05)
        _hparam('l_dale_nu', 0.1)
    
    elif perturbation == 'CPAB':
        _hparam('tesselation', 10)
        _hparam('epsilon', 1)

        ##### Worst of K ######
        _hparam('worst_of_k_steps', 10)

        _hparam('epsilon_rot', 30)
        _hparam('epsilon_tx', 3)
        _hparam('epsilon_ty', 3)

        ###### MCMC ###########
        _hparam('mcmc_dale_scale', 1)
        _hparam('mcmc_dale_n_steps', 10)
        _hparam('mcmc_proposal', 'Laplace')
        

        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 5)
            _hparam('pgd_step_size', 0.1)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('pgd_n_steps', 10)
            _hparam('pgd_step_size', 0.5)

        ##### TRADES #####
        if dataset == 'MNIST':
            _hparam('trades_n_steps', 40)
            _hparam('trades_step_size', 5)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('trades_n_steps', 40)
            _hparam('trades_step_size', 5)
        
        # Grid Search
        _hparam('grid_size', 120)
        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 8)
        _hparam('l_dale_step_size', 1)
        _hparam('l_dale_noise_coeff', 0.2)
        _hparam('l_dale_nu', 0.1)
    

    elif perturbation == 'Crop' or perturbation == 'Crop_and_Flip':
        _hparam('epsilon', 4)

        ##### Worst of K ######
        _hparam('worst_of_k_steps', 10)

        ###### MCMC ###########
        _hparam('mcmc_dale_scale', 0.01)
        _hparam('mcmc_dale_n_steps', 5)
        _hparam('mcmc_proposal', 'Laplace')
        

        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 5)
            _hparam('pgd_step_size', 0.1)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('pgd_n_steps', 10)
            _hparam('pgd_step_size', 0.5)

        ##### TRADES #####
        if dataset == 'MNIST':
            _hparam('trades_n_steps', 40)
            _hparam('trades_step_size', 5)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            _hparam('trades_n_steps', 40)
            _hparam('trades_step_size', 5)
        
        # Grid Search
        _hparam('grid_size', 120)
        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 15)
        _hparam('l_dale_step_size', 0.05)
        _hparam('l_dale_noise_coeff', 0.05)
        _hparam('l_dale_nu', 0.5)
    else:
        raise NotImplementedError
        
    return hparams
