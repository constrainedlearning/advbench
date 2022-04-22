for seed in 0 1 2 3 4 5 6 7 8 9
do
    for algo in Laplacian_DALE_PD_Reverse
    do
        for eps in  0.025 0.05 0.1 0.2
        do
            python -m advbench.scripts.train --dataset modelnet40 --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation PointcloudJitter --model wrn-16-8 --seed $seed --eps $eps --flags pareto --device cuda:1
        done
    done
done