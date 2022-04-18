for seed in 0
do
    for perturbations in 'SE' 'Translation'
    do
        for algo in ERM Augmentation MH_DALE_PD_Reverse Laplacian_DALE_PD_Reverse Adversarial_Adam Adversarial_Worst_Of_K Adversarial_PGD PGD_DALE_PD_Reverse Adam_DALE_PD_Reverse
        do
            python -m advbench.scripts.train_no_validation --dataset STL10 --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation SE --model wrn-16-8 --seed $seed --augment
        done
        for algo in ERM Augmentation Laplacian_DALE_PD_Reverse MH_DALE_PD_Reverse Adversarial_Adam Adversarial_Worst_Of_K Adversarial_PGD Adam_DALE_PD_Reverse
        do
            python -m advbench.scripts.train_no_validation --dataset STL10 --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation SE --model wrn-16-8 --seed $seed
        done
    done
done