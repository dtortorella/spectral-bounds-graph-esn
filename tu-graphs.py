import argparse
import math
import statistics

import numpy as np
import torch
from scipy.io import savemat
from torch.nn.functional import one_hot
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import degree
from tqdm import tqdm

from experiment_tools import holdout_split_stratified
from graphesn import StaticGraphReservoir, initializer, Readout
from graphesn.util import graph_spectral_norm


def batch_split(x, y, batch_size):
    if batch_size is None:
        return x, y
    else:
        return zip(torch.split(x, batch_size), torch.split(y, batch_size))


def make_batches(dataset, batch_size, device, random_feats):
    if batch_size is None:
        batch_size = len(dataset)
    batches = []
    for i in range(math.ceil(len(dataset) / batch_size)):
        data = Batch.from_data_list(dataset[batch_size*i:batch_size*(i+1)]).to(device)
        if random_feats:
            data.x = torch.rand_like(data.x)
        elif data.x is None:
            data.x = torch.ones(data.num_nodes, 1).to(device)
        batches.append(data)
    return batches


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name')
parser.add_argument('--root', help='root directory for dataset', default='/tmp')
parser.add_argument('--device', help='device for torch computations', default='cpu')
parser.add_argument('--units', help='reservoir units per layer', type=int, nargs='+', default=[64])
parser.add_argument('--init', help='random recurrent initializer (uniform, normal, ring)', type=str, default='uniform')
parser.add_argument('--iterations', help='max iterations', type=int, default=100)
parser.add_argument('--epsilon', help='convergence threshold', type=float, default=1e-8)
parser.add_argument('--use', help='rho or sigma', default='rho')
parser.add_argument('--rho', help='rho for recurrent matrix initialization', type=float, nargs='+', default=[0.9])
parser.add_argument('--sigma', help='sigma for recurrent matrix initialization', type=float, nargs='+', default=[0.9])
parser.add_argument('--scale', help='scale for input matrix initialization', type=float, nargs='+', default=[1.0])
parser.add_argument('--ld', help='readout lambda', type=float, nargs='+', default=[1e-3])
parser.add_argument('--trials', help='number of trials', type=int, default=25)
parser.add_argument('--bias', help='whether bias term is present', action='store_true')
parser.add_argument('--perturb', help='initial state perturbation scale', type=float, default=None)
parser.add_argument('--no-feat', help='remove node features', action='store_true')
parser.add_argument('--batch-size', help='batch size for regression fit', type=int, default=None)
args = parser.parse_args()

dataset = TUDataset(root=args.root, name=args.dataset.upper())
device = torch.device(args.device)
data = make_batches(dataset, args.batch_size, device, args.no_feat)

alphas = [graph_spectral_norm(g.edge_index) for g in dataset]
alpha = statistics.mean(alphas)
print(f'graph alpha = {float(alpha):.2f} ± {float(statistics.stdev(alphas)):.2f}')

K_max = [degree(g.edge_index[0]).max().item() for g in dataset]
K_mean = [degree(g.edge_index[0]).mean().item() for g in dataset]
print(f'k_max = {statistics.mean(K_max):.2f} ± {statistics.stdev(K_max):.2f}, k_mean = {statistics.mean(K_mean):.2f} ± {statistics.stdev(K_mean):.2f}')

train_acc = torch.zeros(len(args.units), len(getattr(args, args.use)), len(args.scale), len(args.ld), args.trials)
test_acc = torch.zeros_like(train_acc)

labels = [int(g.y.item()) for g in dataset]
train_set, test_set = holdout_split_stratified(labels, .2)
y = torch.tensor(labels).to(device)
Y = one_hot(y).float()

with tqdm(total=train_acc.numel()) as progress:
    for unit_index, unit in enumerate(args.units):
        reservoir = StaticGraphReservoir(num_layers=1, in_features=max(dataset.num_features, 1), hidden_features=unit,
                                         max_iterations=args.iterations, epsilon=args.epsilon, bias=args.bias,
                                         pooling=global_add_pool)
        readout = Readout(num_features=reservoir.out_features, num_targets=dataset.num_classes)
        for rho_index, rho in enumerate(getattr(args, args.use)):
            for scale_index, scale in enumerate(args.scale):
                for trial_index in range(args.trials):
                    reservoir.initialize_parameters(recurrent=initializer(args.init, **{args.use: rho / alpha}),
                                                    input=initializer('uniform', scale=scale),
                                                    bias=initializer('uniform', scale=0.1))
                    reservoir.to(device)
                    x = torch.cat([reservoir(edge_index=batch.edge_index,
                                             input=batch.x,
                                             initial_state=torch.empty(batch.num_nodes, unit).uniform_(-args.perturb, args.perturb).to(device) if args.perturb else None,
                                             batch=batch.batch) for batch in data], dim=0)
                    for ld_index, ld in enumerate(args.ld):
                        readout.fit(batch_split(x[train_set], Y[train_set], args.batch_size), ld)
                        y_match = (readout(x).argmax(dim=-1) == y)
                        train_acc[unit_index, rho_index, scale_index, ld_index, trial_index] = y_match[train_set].float().mean()
                        test_acc[unit_index, rho_index, scale_index, ld_index, trial_index] = y_match[test_set].float().mean()
                        progress.update(1)

savemat(f'{args.dataset}.mat', mdict={
    'train_acc': train_acc.cpu().numpy(), 'test_acc': test_acc.cpu().numpy(), 'alpha': np.array(alphas),
    'units': np.array(args.units), args.use: np.array(getattr(args, args.use)), 'scale': np.array(args.scale), 'ld': np.array(args.ld)
})
