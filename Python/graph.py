from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch


class Graph(object):
    def __init__(self, name):
        super(Graph, self).__init__()
        self.name = name

        print("Loading {}.edges ...".format(self.name))
        self.n_nodes = 0
        self.n_edges = 0
        self.node_map = {}
        self.reverse_node_map = {}
        self.adj_lists = defaultdict(set)
        with open("./data/" + self.name + ".edges", "r") as f:
            for line in f.readlines():
                u, v = line.strip().split()
                u, v = int(u), int(v)
                if u not in self.node_map:
                    self.node_map[u] = self.n_nodes
                    self.reverse_node_map[self.n_nodes] = u
                    self.n_nodes += 1
                if v not in self.node_map:
                    self.node_map[v] = self.n_nodes
                    self.reverse_node_map[self.n_nodes] = v
                    self.n_nodes += 1
                self.adj_lists[self.node_map[u]].add(self.node_map[v])
                self.adj_lists[self.node_map[v]].add(self.node_map[u])
                self.n_edges += 1
        print("Finish loading {}.edges, |V| = {}, |E| = {}".format(self.name, self.n_nodes, self.n_edges))
