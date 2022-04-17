pushd advbench
for seed in 0
do
for algo in ERM Augmentation Laplacian_DALE_PD_Reverse Adversarial_Worst_of_K Adversarial_Adam PGD MCMC_DALE_PD_Reverse
    do
        python -m advbench.scripts.train_no_validation --dataset CIFAR --algorithm $algo --output_dir train-output --test_attacks Gaussian_Batch Laplacian_Batch LMC_Laplacian_Linf MCMC Grid_Batch PGD_Linf Rand_Aug_Batch Worst_Of_K --perturbation SE --model wrn-28-10 --log_imgs --seed $seed
    done
done
popd