for seed in 0 1 2 3 4
do
    for algo in Adversarial_Smoothed
    do
        for penalty in  1 1.4 0.6 0.2
        do
            for ndata in 0.2 0.4 0.6 0.8 1
            do
                python -m advbench.scripts.train_samples --dataset MNIST --algorithm $algo --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation SE --model MNISTnet --seed $seed --flags adv --fracsamples $ndata --device cuda:1 --adv_penalty $penalty
            done
        done
    done
done