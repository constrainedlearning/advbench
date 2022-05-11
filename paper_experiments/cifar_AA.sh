for seed in 0 1 2
do
    python -m advbench.scripts.train_no_validation --dataset CIFAR100 --algorithm Laplacian_DALE_PD_Reverse --output_dir train-output --test_attacks Rand_Aug_Batch --perturbation Translation --model wrn-28-10 --seed $seed --auto_augment_wo_translations
    python -m advbench.scripts.train_no_validation --dataset CIFAR100 --algorithm ERM --output_dir train-output --test_attacks Rand_Aug_Batch --perturbation Translation --model wrn-28-10 --seed $seed --auto_augment
done