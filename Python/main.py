from graph import *
from model import *
from utils import *
from brain import *
import argparse
import random
import numpy as np
import torch

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
parser.add_argument('--dataset', type=str, default='syn')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--node_emb_size', type=int, default=5)
parser.add_argument('--hidden_emb_size', type=int, default=10)
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gcn', action='store_true')

# parser.add_argument('--learn_method', type=str, default='sup')
# parser.add_argument('--unsup_loss', type=str, default='normal')
# parser.add_argument('--max_vali_f1', type=float, default=0)
# parser.add_argument('--name', type=str, default='debug')
# parser.add_argument('--config', type=str, default='./src/experiments.conf')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    graph = Graph(args.dataset)

    graph_model = GraphSage(
        n_nodes=graph.n_nodes,
        num_layers=args.num_layers,
        in_size=args.node_emb_size,
        out_size=args.hidden_emb_size,
        adj_lists=graph.adj_lists,
        device=device,
        gcn=args.gcn,
        agg_func=args.agg_func)
    graph_model.to(device)

    brain = Brain(
        state_dim=args.hidden_emb_size,
        action_dim=graph.n_nodes,
        device=device,
        hidden_dim=20,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=80,
        eps_clip=0.2
    )

    train(graph, graph_model, brain, args.batch_size)

