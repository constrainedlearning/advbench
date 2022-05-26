for seed in 0
do
    python -m advbench.scripts.train_no_validation --dataset CIFAR100 --algorithm Augmentation --test_attacks Rand_Aug --output_dir train-output --perturbation TAUG --model wrn-28-10 --seed $seed
done