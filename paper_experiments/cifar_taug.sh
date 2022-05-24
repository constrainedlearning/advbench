for seed in 0 1 2 3 4
do
    python -m advbench.scripts.train_no_validation --dataset CIFAR100 --algorithm Worst_DALE_PD_Reverse --output_dir train-output --perturbation TAUG --model wrn-28-10 --seed $seed --epsilon 0.8 --augment
done