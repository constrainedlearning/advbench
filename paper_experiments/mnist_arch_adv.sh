for seed in 0 1 2 3 4
do
    for algo in Laplacian_DALE_PD_Reverse
    do
        for eps in  0.14
        do
            for nlayers in 8 4
            do
                python -m advbench.scripts.train_nlayers --dataset MNIST --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation SE --model MNISTnet --seed $seed --eps $eps --nlayers $nlayers --flag layers --device cuda:1
            done
        done
    done
done