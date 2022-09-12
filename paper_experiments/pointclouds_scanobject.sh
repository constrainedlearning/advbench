for seed in 1 2 3 4
do
    for perturbations in 'PointcloudJitter'
    do
        for algo in  ERM
        do
            python -m advbench.scripts.train --dataset scanobjectnn --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation $perturbations --model DGCNN --seed $seed --project ood
        done
    done
done
