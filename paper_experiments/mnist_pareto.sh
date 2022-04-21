for seed in 0 1 2 3 4 5 6 7 8 9
do
    for algo in Laplacian_DALE_PD_Reverse
    do
        for eps in  0.0075 0.015 0.035 0.14 0.28 0.56
        do
            python -m advbench.scripts.train_no_validation --dataset MNIST --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation SE --model MNISTnet --seed $seed --eps $eps --flags pareto --epochs 200
        done
    done
done