import os
import time
import argparse
import random
import numpy as np
import math
import pandas as pd

from tqdm import trange
from statistics import mean
from scipy import stats
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from MLP import MLP
import data_process
import flop_benchmark

parser = argparse.ArgumentParser(description='EPE-NAS')
parser.add_argument('--data_loc', default='./datasets/cifar', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='./datasets/NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--evaluate_size', default=256, type=int)
# parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=10, type=int)
parser.add_argument('--n_runs', default=20, type=int)
parser.add_argument("--hardware_aware", action='store_true', help="whether to use hardware-aware")
parser.add_argument("--latency_evaluate_method", default="predictor", type=str, help="choose latency evaluate method from predictor and lookup table")
parser.add_argument("--max_latency", default=500, type=int, help="maximum latency")

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.optim as optim

from models import get_cell_based_tiny_net
import pickle

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

from datasets import get_datasets
from config_utils import load_config
from nas_201_api import NASBench201API as API


def get_batch_jacobian(net, x, target, to, device, args=None):
    net.zero_grad()

    x.requires_grad_(True)

    _, y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()#, grad


def eval_score_perclass(jacob, labels=None, n_classes=10):
    k = 1e-5
    #n_classes = len(np.unique(labels))
    per_class={}
    for i, label in enumerate(labels[0]):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label],jacob[i]))
        else:
            per_class[label] = jacob[i]

    ind_corr_matrix_score = {}
    for c in per_class.keys():
        s = 0
        try:
            corrs = np.corrcoef(per_class[c])

            s = np.sum(np.log(abs(corrs)+k))#/len(corrs)
            if n_classes > 100:
                s /= len(corrs)
        except: # defensive programming
            continue

        ind_corr_matrix_score[c] = s

    # per class-corr matrix A and B
    score = 0
    ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()
    if n_classes <= 100:

        for c in ind_corr_matrix_score_keys:
            # B)
            score += np.absolute(ind_corr_matrix_score[c])
    else: 
        for c in ind_corr_matrix_score_keys:
            # A)
            for cj in ind_corr_matrix_score_keys:
                score += np.absolute(ind_corr_matrix_score[c]-ind_corr_matrix_score[cj])

        # should divide by number of classes seen
        score /= len(ind_corr_matrix_score_keys)

    return score


# NAS-WOT method, for comparison
def eval_score(jacob, labels=None, n_classes=10):

    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    score = -np.sum(np.log(v + k) + 1./(v + k))
    return score

# MLP latency predictor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

max_latency = args.max_latency
model = MLP(input_size=32).to(device)
model.load_state_dict(torch.load("MLP_latency_predictor.pth"))
model.eval()

# LUT
if args.dataset == "cifar10" or args.dataset == "cifar100":
    lookup_table = pd.read_excel("./LUT/{}-LUT.xls".format("cifar10"), index_col="arch_str")
elif args.dataset == "ImageNet16-120":
    lookup_table = pd.read_excel("./LUT/{}-LUT.xls".format("ImageNet16-120"), index_col="arch_str")

api = API(args.api_loc)

os.makedirs(args.save_loc, exist_ok=True)

train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_loc, cutout=0)


if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'

else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

if args.trainval:
    cifar_split = load_config('config_utils/cifar-split.txt', None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               num_workers=0, pin_memory=True, sampler= torch.utils.data.sampler.SubsetRandomSampler(train_split))

else:
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True)

times     = []
chosen    = []
acc       = []
val_acc   = []
topscores = []
pred_latencys = []
real_latencys = []

dset = args.dataset if not args.trainval else 'cifar10-valid'

order_fn = np.nanargmax

runs = trange(args.n_runs, desc='')

# hardware-aware
if(args.hardware_aware == True):
    if(args.latency_evaluate_method == "predictor"):
        for N in runs:
            start = time.time()
            indices = np.random.randint(0,15625,args.n_samples)
            scores = []
            #accs      = []
            for arch in indices:
                config = api.get_net_config(arch, args.dataset)
                network = get_cell_based_tiny_net(config)  # create the network from configuration
                network = network.to(device)
                arch_str = config["arch_str"]

                if args.dataset == "cifar10" or args.dataset == "cifar100":
                    dataset_input_size = 1 * 3 * 32 * 32
                    flops = flop_benchmark.get_model_flops(network, (1, 3, 32, 32))
                elif args.dataset == "ImageNet16-120":
                    dataset_input_size = 1 * 3 * 16 * 16
                    flops = flop_benchmark.get_model_flops(network, (1, 3, 16, 16))
                
                features = data_process.get_features_by_arch_str(arch_str, dataset_input_size, flops).to(device)
                latency = model(features)
                if(latency > max_latency):
                    scores.append(0)
                    continue
                data_iterator = iter(train_loader)
                x, target = next(data_iterator)
                x, target = x.to(device), target.to(device)

                
                jacobs = []
                targets = []
                grads = []
                iterations = np.int32(np.ceil(args.evaluate_size/args.batch_size))
                
                for i in range(iterations):
                    jacobs_batch, target = get_batch_jacobian(network, x, target, None, None)
                    jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())
                    targets.append(target.cpu().numpy())
                jacobs = np.concatenate(jacobs, axis=0)
                if(jacobs.shape[0]>args.evaluate_size):
                    jacobs = jacobs[0:args.evaluate_size, :]

                try:
                    s = eval_score_perclass(jacobs, targets)

                except Exception as e:
                    print(e)
                    s = np.nan
                scores.append(s)


            #print(f'max test acc:{np.max(accs)}')

            best_arch = indices[order_fn(scores)]
            info      = api.query_by_index(best_arch)
            topscores.append(scores[order_fn(scores)])
            chosen.append(best_arch)
            acc.append(info.get_metrics(dset, acc_type)['accuracy'])

            if not args.dataset == 'cifar10' or args.trainval:
                val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

            times.append(time.time()-start)
            runs.set_description(f"acc: {mean(acc if not args.trainval else val_acc):.2f}%")
            
            print(times)

            config = api.get_net_config(best_arch, args.dataset)
            network = get_cell_based_tiny_net(config)  # create the network from configuration
            network = network.to(device)
            arch_str = config["arch_str"]
            network
            if args.dataset == "cifar10" or args.dataset == "cifar100":
                dataset_input_size = 1 * 3 * 32 * 32
                flops = flop_benchmark.get_model_flops(network, (1, 3, 32, 32))
            elif args.dataset == "ImageNet16-120":
                dataset_input_size = 1 * 3 * 16 * 16
                flops = flop_benchmark.get_model_flops(network, (1, 3, 16, 16))
            features = data_process.get_features_by_arch_str(arch_str, dataset_input_size, flops).to(device)
            pred_latency = model(features)
            real_latency = lookup_table.loc[arch_str, "nezhaD1"]
            pred_latencys.append(pred_latency.item())
            real_latencys.append(real_latency.item())
    elif args.latency_evaluate_method == "lookup_table":
        for N in runs:
            start = time.time()
            indices = np.random.randint(0,15625,args.n_samples)
            scores = []
            #accs      = []
            for arch in indices:
                config = api.get_net_config(arch, args.dataset)
                network = get_cell_based_tiny_net(config)  # create the network from configuration
                network = network.to(device)
                arch_str = config["arch_str"]

                latency = lookup_table.loc[arch_str, "nezhaD1"]
                if(latency > max_latency):
                    scores.append(0)
                    continue
                data_iterator = iter(train_loader)
                x, target = next(data_iterator)
                x, target = x.to(device), target.to(device)
                
                jacobs = []
                targets = []
                grads = []
                iterations = np.int32(np.ceil(args.evaluate_size/args.batch_size))
                
                for i in range(iterations):
                    jacobs_batch, target = get_batch_jacobian(network, x, target, None, None)
                    jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())
                    targets.append(target.cpu().numpy())
                jacobs = np.concatenate(jacobs, axis=0)
                if(jacobs.shape[0]>args.evaluate_size):
                    jacobs = jacobs[0:args.evaluate_size, :]

                try:
                    s = eval_score_perclass(jacobs, targets)

                except Exception as e:
                    print(e)
                    s = np.nan
                scores.append(s)


            best_arch = indices[order_fn(scores)]
            info      = api.query_by_index(best_arch)
            topscores.append(scores[order_fn(scores)])
            chosen.append(best_arch)
            acc.append(info.get_metrics(dset, acc_type)['accuracy'])

            if not args.dataset == 'cifar10' or args.trainval:
                val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

            times.append(time.time()-start)
            runs.set_description(f"acc: {mean(acc if not args.trainval else val_acc):.2f}%")
            
            print(times)
            
            config = api.get_net_config(best_arch, args.dataset)
            arch_str = config["arch_str"]
            pred_latency = lookup_table.loc[arch_str, "nezhaD1"]
            real_latency = pred_latency
            pred_latencys.append(pred_latency)
            real_latencys.append(real_latency)
# no hardware-aware
else:
    for N in runs:
        start = time.time()
        indices = np.random.randint(0,15625,args.n_samples)
        scores = []
        for arch in indices:
            data_iterator = iter(train_loader)
            x, target = next(data_iterator)
            x, target = x.to(device), target.to(device)

            config = api.get_net_config(arch, args.dataset)

            network = get_cell_based_tiny_net(config)  # create the network from configuration
            network = network.to(device)

            jacobs = []
            targets = []
            grads = []
            iterations = np.int32(np.ceil(args.evaluate_size/args.batch_size))
            
            for i in range(iterations):
                jacobs_batch, target = get_batch_jacobian(network, x, target, None, None)
                jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())
                targets.append(target.cpu().numpy())
            jacobs = np.concatenate(jacobs, axis=0)
            if(jacobs.shape[0]>args.evaluate_size):
                jacobs = jacobs[0:args.evaluate_size, :]

            try:
                s = eval_score_perclass(jacobs, targets)

            except Exception as e:
                print(e)
                s = np.nan
            scores.append(s)


        best_arch = indices[order_fn(scores)]
        info      = api.query_by_index(best_arch)
        topscores.append(scores[order_fn(scores)])
        chosen.append(best_arch)
        acc.append(info.get_metrics(dset, acc_type)['accuracy'])

        if not args.dataset == 'cifar10' or args.trainval:
            val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

        times.append(time.time()-start)
        runs.set_description(f"acc: {mean(acc if not args.trainval else val_acc):.2f}%")
        
        print(times)

        config = api.get_net_config(best_arch, args.dataset)
        arch_str = config["arch_str"]
        pred_latency = 0
        real_latency = lookup_table.loc[arch_str, "nezhaD1"]
        pred_latencys.append(pred_latency)
        real_latencys.append(real_latency)
print(f'mean time: {np.mean(times)}')

print(val_acc)
print(f"Final mean test accuracy: {np.mean(acc)}")
if len(val_acc) > 1:
    print(f"Final mean validation accuracy: {np.mean(val_acc)}")

print(f"Final predict latency: {pred_latencys}")
print(f"Final real latency: {real_latencys}")

state = {'accs': acc,
         'val_accs': val_acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         'pred_latency': pred_latencys,
         'real_latency': real_latencys,
         }

dset = args.dataset if not args.trainval else 'cifar10-valid'
if args.hardware_aware is True:
    fname = f"{args.save_loc}/{dset}_{args.n_runs}_{args.n_samples}_{args.seed}_{args.hardware_aware}_{args.latency_evaluate_method}_{args.max_latency}.t7"
else:
    fname = f"{args.save_loc}/{dset}_{args.n_runs}_{args.n_samples}_{args.seed}_{args.hardware_aware}.t7"
torch.save(state, fname)
