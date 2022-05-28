for seed in 0 1 2 3
do
for algo in Adversarial_Worst_Of_K
    do
        python -m advbench.scripts.train_no_validation --dataset CIFAR100 --algorithm $algo --output_dir train-output --test_attacks Rand_Aug_Batch Fo_PGD Fo_Adam Gaussian_Batch Laplacian_Batch --perturbation SE --model wrn-28-10 --seed $seed
    done
done