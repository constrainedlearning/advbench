for seed in 0 1 2 3
do
    for algo in Laplacian_DALE_PD_Reverse
    do
        for eps in 0.035 0.14 0.7
        do
            python -m advbench.scripts.train_no_validation --dataset CIFAR100 --algorithm $algo --output_dir train-output --test_attacks Rand_Aug Fo_Adam Fo_PGD Gaussian_aug --perturbation SE --model wrn-28-10 --seed $seed --eps $eps --flags pareto --device cuda:1
        done
    done
done
