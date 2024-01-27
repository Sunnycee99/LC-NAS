import numpy as np
import argparse
import os
import random
import pandas as pd
from collections import OrderedDict

import tabulate
parser = argparse.ArgumentParser(description='Produce tables')
parser.add_argument('--data_loc', default='./datasets/cifar/', type=str, help='dataset folder')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--GPU', default='0', type=str)

parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')

parser.add_argument('--n_runs', default=20, type=int)

parser.add_argument("--hardware_aware", action='store_true')
parser.add_argument("--latency_evaluate_method", default="predictor", type=str, help="choose latency evaluate method from predictor and lookup table")
parser.add_argument("--max_latency", default=500, type=int)

args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

from statistics import mean, median, stdev as std

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

df = []

datasets = OrderedDict()

datasets['CIFAR-10 (val)'] = ('cifar10-valid', 'x-valid', True)
datasets['CIFAR-10 (test)'] = ('cifar10', 'ori-test', False)



dataset_top1s = OrderedDict()
latencys = OrderedDict()

for n_samples in [10, 100, 500, 1000]:
    method = f"Ours (N={n_samples})"

    time = 0.

    for dataset, params in datasets.items():
        top1s = []

        dset =  params[0]
        acc_type = 'accs' if 'test' in params[1] else 'val_accs'
        if args.hardware_aware is True:
            filename = f"{args.save_loc}/{dset}_{args.n_runs}_{n_samples}_{args.seed}_{args.hardware_aware}_{args.latency_evaluate_method}_{args.max_latency}.t7"
        else:
            filename = f"{args.save_loc}/{dset}_{args.n_runs}_{n_samples}_{args.seed}_{args.hardware_aware}.t7"
        full_scores = torch.load(filename)
        if dataset == 'CIFAR-10 (test)':
            time_mean = mean(full_scores['times'])
            time_std = std(full_scores['times'])
            time = f"{time_mean:.2f} +- {time_std:.2f}"

        accs = []
        for n in range(args.n_runs):
            acc = full_scores[acc_type][n]
            accs.append(acc)
        dataset_top1s[dataset] = accs
        latencys[dataset+" pred_latency"] = full_scores["pred_latency"]
        latencys[dataset+" real_latency"] = full_scores["real_latency"]


    cifar10_val  = f"{mean(dataset_top1s['CIFAR-10 (val)']):.2f} +- {std(dataset_top1s['CIFAR-10 (val)']):.2f}"
    cifar10_test = f"{mean(dataset_top1s['CIFAR-10 (test)']):.2f} +- {std(dataset_top1s['CIFAR-10 (test)']):.2f}"
    cifar10_pred_latency = f"{mean(latencys['CIFAR-10 (test) pred_latency']):.2f} +- {std(latencys['CIFAR-10 (test) pred_latency']):.2f}"
    cifar10_real_latency = f"{mean(latencys['CIFAR-10 (test) real_latency']):.2f} +- {std(latencys['CIFAR-10 (test) real_latency']):.2f}"

    df.append([method, time, cifar10_val, cifar10_test, cifar10_pred_latency, cifar10_real_latency])

df = pd.DataFrame(df, columns=['Method', 'Search time (s)','CIFAR-10 (val)','CIFAR-10 (test)','Pred_latency','Real_latency'])
print("Prediction method: {}\nLatency constraint: {}".format(args.latency_evaluate_method, args.max_latency))
print(tabulate.tabulate(df.values,df.columns, tablefmt="pipe"))