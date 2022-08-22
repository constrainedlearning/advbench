for seed in 0 1 2 3
do
    for algo in Adversarial_Smoothed
    do
        for penalty in 0.25 0.5 1.0 2.0
        do
            python -m advbench.scripts.train_no_validation --penalty $penalty --dataset CIFAR100 --algorithm $algo --output_dir train-output --test_attacks Beta_aug Rand_Aug_Batch Fo_SGD --beta 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.5 1.0 1.0 1.0 1.0 2.0 2.0 2.0 2.0 --alpha 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 --perturbation SE --model wrn-16-8 --seed $seed --flags final --project ood --device cuda:1
        done
    done
done
