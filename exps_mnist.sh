for algo in MCMC_DALE_PD_Reverse Worst_DALE_PD_Reverse
    do
        python -m advbench.scripts.train_no_validation --dataset MNIST --algorithm $algo --output_dir train-output --test_attacks Worst_Of_K PGD_Linf Grid_Search MCMC LMC_Laplacian_Linf --perturbation SE
    done