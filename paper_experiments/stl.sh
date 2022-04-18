for seed in 0
do
    for perturbations in 'SE' 'Translation'
    do
        for algo in ERM Augmentation Laplacian_DALE_PD_Reverse MH_DALE_PD_Reverse Adversarial_Adam Adversarial_Worst_of_K Adversarial_PGD Adam_DALE_PD_Reverse
        do
            python -m advbench.scripts.train_no_validation --dataset MNIST --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Worst_Of_K Gaussian_Batch Laplacian_Batch --perturbation SE --model wrn16_8 --seed $seed --device cuda:1 --aug
        done
        for algo in ERM Augmentation Laplacian_DALE_PD_Reverse MH_DALE_PD_Reverse Adversarial_Adam Adversarial_Worst_of_K Adversarial_PGD Adam_DALE_PD_Reverse
        do
            python -m advbench.scripts.train_no_validation --dataset MNIST --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Worst_Of_K Gaussian_Batch Laplacian_Batch --perturbation SE --model wrn16_8 --seed $seed --device cuda:1
        done
    done
done