for N in 10 100 500 1000
do
    python search.py --data_loc './datasets/cifar10'            --n_runs 20 --n_samples $N --api_loc 'datasets/NAS-Bench-201-v1_0-e61699.pth' --hardware_aware --latency_evaluate_method predictor
    python search.py --trainval --data_loc './datasets/cifar10' --n_runs 20 --n_samples $N --api_loc 'datasets/NAS-Bench-201-v1_0-e61699.pth' --hardware_aware --latency_evaluate_method predictor
done