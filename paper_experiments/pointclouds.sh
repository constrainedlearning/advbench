for seed in 0
do
    for perturbations in 'PointcloudJitter'
    do
        for algo in  ERM Augmentation Laplacian_DALE_PD_Reverse MH_DALE_PD_Reverse Adversarial_Worst_of_K Adversarial_PGD Adversarial_Adam PGD_DALE_PD_Reverse Adam_DALE_PD_Reverse
        do
            python -m advbench.scripts.train --dataset modelnet40 --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation $perturbations --model DGCNN --seed $seed
        done
    done
done