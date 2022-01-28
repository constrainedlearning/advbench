for algo in Augmentation MCMC_DALE_PD_Reverse PGD_DALE_PD_Reverse Laplacian_DALE_PD_Reverse PGD_DALE_PD_Reverse Gaussian_DALE_PD_Reverse Worst_of_K
    do
        python -m advbench.scripts.train_no_validation --dataset CIFAR --algorithm $algo --output_dir train-output --test_attacks Worst_Of_K PGD_Linf Grid_Search MCMC LMC_Laplacian_Linf --perturbation SE
    done