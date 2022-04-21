from graph import *
from utils import *
from brain import *
import argparse
import random
import numpy as np
import torch

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
parser.add_argument('--dataset', type=str, default='syn')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--node_emb_size', type=int, default=32)
parser.add_argument('--hidden_emb_size', type=int, default=32)
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--k', type=int, default=3)

# parser.add_argument('--learn_method', type=str, default='sup')
# parser.add_argument('--unsup_loss', type=str, default='normal')
# parser.add_argument('--max_vali_f1', type=float, default=0)
# parser.add_argument('--name', type=str, default='debug')
# parser.add_argument('--config', type=str, default='./src/experiments.conf')
args = parser.parse_args()



if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    graph = Graph(args.dataset)

    train(graph, args)

