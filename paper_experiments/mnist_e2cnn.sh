for seed in 0
do
    for algo in ERM Worst_of_K PGD #MCMC_DALE_PD_Reverse Laplacian_DALE_PD_Reverse Augmentation
        do
            python -m advbench.scripts.train_no_validation --dataset MNIST --algorithm $algo --output_dir train-output --test_attacks Gaussian_Batch Laplacian_Batch LMC_Laplacian_Linf MCMC Grid_Batch PGD_Linf Rand_Aug_Batch Worst_Of_K --perturbation SE --model CnSteerableCNN-exp --log_imgs --seed $seed
        done
done