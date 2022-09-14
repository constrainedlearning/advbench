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
    if dataset in ['modelnet40', 'scanobjectnn']:
        _hparam('batch_size', 32, lambda r: int(2 ** r.uniform(3, 6)))
        _hparam('label_smoothing', 0.2, lambda r: 0.2)
    else:
        _hparam('batch_size', 128, lambda r: int(2 ** r.uniform(3, 8)))
        _hparam('label_smoothing', 0.0, lambda r: 0.0)
    _hparam('augmentation_prob', 1, lambda r: 1)
    _hparam('perturbation_batch_size', 10, lambda r: 10)
    _hparam('mh_dale_scale', 0.05, lambda r: 0.05)
    _hparam('mh_proposal', 'Laplace', lambda r: 'Laplace')
    _hparam('gaussian_attack_std', 1, lambda r: 1 )
    _hparam('laplacian_attack_std', 1, lambda r: 1 )
    _hparam('adv_penalty', 0, lambda r: 0)
    # Beta Aug
    _hparam('beta_attack_alpha', 0.5, lambda r: 0.5)
    _hparam('beta_attack_beta', 0.25, lambda r: 0.25)
    
    # optimization
    if dataset == 'MNIST':
        _hparam('learning_rate', 0.01, lambda r: 10 ** r.uniform(-1.5, -0.5))
        _hparam('lr_decay_start', 15, lambda r: 15)
        _hparam('lr_decay_factor', 0.8, lambda r: r.uniform(0.1, 0.3))
        _hparam('lr_decay_epoch', 1, lambda r: 1)
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'STL10':
        _hparam('learning_rate', 0.1, lambda r: 10 ** r.uniform(-2, -1))
    if dataset == 'CIFAR100' or dataset == 'STL10':
        _hparam('lr_decay_start', 0, lambda r: 0)
        _hparam('lr_decay_factor', 0.2, lambda r: r.uniform(0.1, 0.3))
        _hparam('lr_decay_epoch', 60, lambda r: 60)
    if dataset == 'IMNET':
        _hparam('learning_rate', 0.01, lambda r: 10 ** r.uniform(-1.5, -0.5))
        _hparam('lr_decay_start', 0, lambda r: 0)
        _hparam('lr_decay_factor', 0.8, lambda r: r.uniform(0.1, 0.3))
        _hparam('lr_decay_epoch', 1, lambda r: 1)
    if dataset in ['modelnet40', 'scanobjectnn']:
        _hparam('learning_rate', 0.1, lambda r: 10 ** r.uniform(-1.5, -0.5))
        _hparam('weight_decay', 1e-4, lambda r: 10 ** r.uniform(-6, -3))
        _hparam('clip_grad', 1, lambda r: 1)
    else:
        _hparam('weight_decay', 5e-4, lambda r: 10 ** r.uniform(-6, -3))
    _hparam('sgd_momentum', 0.9, lambda r: r.uniform(0.8, 0.95))

    # Wether to batch parrallelizable attacks, bigger mem footprint but faster
    _hparam('batched', 0, lambda r: 0)
    
    if perturbation=='SE':
        ##### Worst of K ######
        _hparam('worst_of_k_steps', 10, lambda r:10)
        _hparam('fo_sgd_step_size', 1, lambda r:1)
        _hparam('fo_sgd_momentum', 0.1, lambda r:0.1)
        _hparam('fo_adam_step_size', 0.1, lambda r:0.1)
        _hparam('fo_n_steps', 10, lambda r:10)
        _hparam('fo_restarts', 1, lambda r:1)
        _hparam('pgd_n_steps', 10, lambda r: 10)
        _hparam('pgd_step_size', 0.1, lambda r: 0.1)
        # MH DALE
        _hparam('mh_dale_n_steps', 30, lambda r:30)

        _hparam('epsilon_rot', 30, lambda r:30)
        if dataset == 'STL10':
            _hparam('epsilon_tx', 10, lambda r:10)
            _hparam('epsilon_ty', 10, lambda r:10)
        else:
            _hparam('epsilon_tx', 3, lambda r:3)
            _hparam('epsilon_ty', 3, lambda r:3)

        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 10, lambda r: 10)
        _hparam('l_dale_step_size', 0.1, lambda r: 10 ** r.uniform(-2.0, -0.5))
        _hparam('l_dale_noise_coeff', 0.01,lambda r: 10 ** r.uniform(-3.0, -1.5))
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
            _hparam('l_dale_pd_inv_margin', 0.07, lambda r: 0.07)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'STL10':
            _hparam('l_dale_pd_inv_step_size', 1, lambda r: 1)
            _hparam('l_dale_pd_inv_eta', 0.0001, lambda r: 0.0001)
            _hparam('l_dale_pd_inv_margin', 0.3, lambda r: 0.3)
        
        # Discrete DALE-PD-INV
        _hparam('d_num_translations', 3, lambda r: 3)
        _hparam('d_num_rotations', 20, lambda r: 20)
        if dataset == 'MNIST':
            _hparam('d_dale_pd_inv_step_size', 0.05, lambda r: 0.05)
            _hparam('d_dale_pd_inv_eta', 0.0008, lambda r: 0.0008)
            _hparam('d_dale_pd_inv_margin', 0.14, lambda r: 0.14)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'STL10':
            _hparam('d_dale_pd_inv_step_size', 1, lambda r: 1)
            _hparam('d_dale_pd_inv_eta', 0.00005, lambda r: 0.00005)
            _hparam('d_dale_pd_inv_margin', 0.15, lambda r: 0.15)
        # Grid Search
        _hparam('grid_size', 120, lambda r: 120)
        

    elif dataset in ['modelnet40', 'scanobjectnn']:
        _hparam('fo_sgd_momentum', 0.1, lambda r:0.1)
        _hparam('fo_adam_step_size', 0.1, lambda r:0.1)
        _hparam('fo_n_steps', 7, lambda r:7)
        _hparam('fo_restarts', 1, lambda r:1)
        _hparam('pgd_n_steps', 7, lambda r: 7)
        _hparam('pgd_step_size', 0.01, lambda r: 0.01)
        _hparam('worst_of_k_steps', 10, lambda r:10)
        if perturbation == "PointcloudTranslation":
            _hparam('epsilon_tx', 0.25, lambda r:0.25)
            _hparam('epsilon_ty', 0.2, lambda r:0.2)
        elif perturbation == "PointcloudJitter":
            _hparam('epsilon', 0.05, lambda r:0.05)
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
        if perturbation == "PointcloudTranslation":
            _hparam('l_dale_pd_inv_step_size', 0.1, lambda r: 0.1)
            _hparam('l_dale_pd_inv_eta', 0.0002, lambda r: 0.0002)
            _hparam('l_dale_pd_inv_margin', 0.35, lambda r: 0.35)
        elif perturbation == "PointcloudJitter":
            _hparam('l_dale_pd_inv_step_size', 0.4, lambda r: 0.4)
            _hparam('l_dale_pd_inv_eta', 0.000025, lambda r: 0.000025)
            _hparam('l_dale_pd_inv_margin', 1.4, lambda r: 1.4)
        
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
    _hparam('perturbation_batch_size', 100)
    _hparam('gaussian_attack_std', 0.5)
    _hparam('laplacian_attack_std', 0.5)
    _hparam('fo_sgd_step_size', 10e2)
    _hparam('fo_sgd_momentum', 0.5)
    if perturbation == "PointcloudJitter":
        _hparam('fo_n_steps', 200)
        _hparam('fo_restarts', 1)
        _hparam('fo_adam_step_size', 0.1)
    else:
        _hparam('fo_n_steps', 30)
        _hparam('fo_restarts', 10)
        _hparam('fo_adam_step_size', 0.1)
    _hparam('grid_size', 100)
    _hparam('worst_of_k_steps', 120)
    _hparam('batched', 0)

    if perturbation=='Rotation':
        if dataset == 'MNIST':
            _hparam('epsilon', 30)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'STL10':
            _hparam('epsilon', 20)

        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 10)
            _hparam('pgd_step_size', 2)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'STL10':
            _hparam('pgd_n_steps', 10)
            _hparam('pgd_step_size', 2)
    
    elif perturbation=='Translation':
        _hparam('epsilon_tx', 4)
        _hparam('epsilon_ty', 4)

        ###### MH ###########
        _hparam('mh_dale_scale', 0.2)
        _hparam('mh_dale_n_steps', 30)
        _hparam('mh_proposal', 'Laplace')
        
        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 30)
            _hparam('pgd_step_size', 0.5)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'STL10':
            _hparam('pgd_n_steps', 20)
            _hparam('pgd_step_size', 0.2)
        elif dataset in ['modelnet40', 'scanobjectnn']:
            _hparam('pgd_n_steps', 200)
            _hparam('pgd_step_size', 0.05)
            
        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 20)
        _hparam('l_dale_step_size', 0.2)
        _hparam('l_dale_noise_coeff', 0.2)
        _hparam('l_dale_nu', 0.1)

    elif perturbation=='SE':
        ##### Worst of K ######

        _hparam('epsilon_rot', 30)
        if dataset == 'STL10':
            _hparam('epsilon_tx', 10)
            _hparam('epsilon_ty', 10)
        else:
            _hparam('epsilon_tx', 3)
            _hparam('epsilon_ty', 3)

        ###### MH ###########
        _hparam('mh_dale_scale', 0.5)
        _hparam('mh_dale_n_steps', 10)
        _hparam('mh_proposal', 'Laplace')
        

        ##### PGD #####
        if dataset == 'MNIST':
            _hparam('pgd_n_steps', 30)
            _hparam('pgd_step_size', 0.3)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'STL10':
            _hparam('pgd_n_steps', 10)
            _hparam('pgd_step_size', 0.5)
        
        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 40)
        _hparam('l_dale_step_size', 0.5)
        _hparam('l_dale_noise_coeff', 0.05)
        _hparam('l_dale_nu', 0.1)

    elif dataset in ['modelnet40', 'scanobjectnn']:
        ##### Worst of K ######
        if perturbation == "PointcloudTranslation":
            _hparam('epsilon_tx', 0.25)
            _hparam('epsilon_ty', 0.2)
        else:
            _hparam('epsilon', 0.05)
        

        ###### MCMC ###########
        if perturbation == "PointcloudTranslation":
            _hparam('mh_dale_scale', 0.2)
        else:
            _hparam('mh_dale_scale', 0.002)
        _hparam('mh_dale_n_steps', 10)
        _hparam('mh_proposal', 'Laplace')
        

        ##### PGD #####
        _hparam('pgd_n_steps', 200)
        if perturbation == "PointcloudTranslation":
            _hparam('pgd_step_size', 0.2)
        else:
            _hparam('pgd_step_size', 0.1)
        
        # DALE (Laplacian-HMC)
        _hparam('l_dale_n_steps', 10)
        if perturbation == "PointcloudTranslation":
            _hparam('l_dale_step_size', 0.1)
        else:
            _hparam('l_dale_step_size', 0.001)
        _hparam('l_dale_pd_inv_margin', 0.35)
        if perturbation == "PointcloudTranslation":
            _hparam('l_dale_noise_coeff', 0.2)
        else:
            _hparam('l_dale_noise_coeff', 0.0002)
        _hparam('l_dale_pd_inv_eta', 0.0002)
        _hparam('l_dale_nu', 0.1)
    else:
        raise NotImplementedError
        
    return hparams
