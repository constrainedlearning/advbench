for seed in 0 1
do
    for algo in Laplacian_DALE_PD_Reverse
    do
        for eps in 0.015 0.035 0.07 0.14 0.3 0.6
        do
            python -m advbench.scripts.train_no_validation --dataset STL10 --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation SE --model wrn-16-8-stl --seed $seed --eps $eps --flags pareto --device cuda:1
        done
    done
done
