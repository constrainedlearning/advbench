for seed in 0 1 2 3 4
do
    for perturbations in 'PointcloudJitter'
    do
        for algo in 'Adversarial_Smoothed'
        do
            for penalty in 0.25 0.5 1.0 2.0
            do
                python -m advbench.scripts.train --dataset modelnet40 --algorithm $algo --output_dir train-output --test_attacks Beta_aug Rand_Aug_Batch Fo_SGD --beta 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.5 1.0 1.0 1.0 1.0 2.0 2.0 2.0 2.0 --alpha 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 --perturbation $perturbations --model DGCNN --seed $seed --penalty $penalty --flags final --device cuda:1 --project ood
            done
        done
    done
done