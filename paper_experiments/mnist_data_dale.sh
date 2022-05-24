for seed in 0 1 2 3 4
do
    for algo in Laplacian_DALE_PD_Reverse
    do
        for eps in  0.14 0.28 0.56 0.035
        do
            for ndata in 0.2 0.4 0.6 0.8 1
            do
                python -m advbench.scripts.train_samples --dataset MNIST --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation SE --model MNISTnet --seed $seed --eps $eps --fracsamples $ndata --device cuda:1
            done
        done
    done
done