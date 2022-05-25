for seed in 0
do
    python -m advbench.scripts.train_no_validation --dataset CIFAR100 --algorithm Worst_DALE_PD_Reverse --test_attacks Worst_Of_K --output_dir train-output --perturbation TAUG --model wrn-28-10 --seed $seed --device cuda:1
done