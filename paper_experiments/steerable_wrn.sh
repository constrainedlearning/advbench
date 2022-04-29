for seed in 0 1 2 3 4
do
    for model in "wrn-28-7-rot-d8" "wrn-28-7-rot" "wrn-28-10-rot"
    do
        python -m advbench.scripts.train_no_validation --dataset CIFAR100 --algorithm Augmentation --output_dir train-output --test_attacks Fo_PGD Fo_Adam Rand_Aug_Batch Gaussian_Batch Laplacian_Batch --perturbation SE --model $model --seed $seed --device cuda:1
    done
done