for seed in 0 1 2 3
do
    for algo in Beta_PD_Reverse
    do
        for eps in 0.015 0.035 0.07 0.14 0.3 0.6
        do
            python -m advbench.scripts.train_no_validation --dataset STL10 --algorithm $algo --output_dir train-output --test_attacks Beta_aug Rand_Aug_Batch Fo_SGD --beta 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.5 1.0 1.0 1.0 1.0 2.0 2.0 2.0 2.0 --alpha 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 --perturbation SE --model wrn-16-8-stl --seed $seed --eps $eps --flags final --project ood --device cuda:1
        done
    done
done

for seed in 0 1 2 3 4
do
    for algo in ERM
    do
            python -m advbench.scripts.train_no_validation --dataset STL10 --algorithm $algo --output_dir train-output --test_attacks Beta_aug Rand_Aug_Batch Fo_SGD --beta 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.5 1.0 1.0 1.0 1.0 2.0 2.0 2.0 2.0 --alpha 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 --perturbation SE --model wrn-16-8-stl --seed $seed --flags final --project ood --device cuda:1
    done
done
