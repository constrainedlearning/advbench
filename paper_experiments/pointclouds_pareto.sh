for seed in 0
do
    for perturbations in 'PointcloudJitter'
    do
        for algo in 'Laplacian_DALE_PD_Reverse'
        do
            for eps in 0.7 2.8
            do
                python -m advbench.scripts.train --dataset modelnet40 --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation $perturbations --model DGCNN --seed $seed --eps $eps --flags pareto
            done
        done
    done
done