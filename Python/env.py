import numpy as np
import os
import re
from heap import *


class Env(object):
    def __init__(self, graph, k, dim=10):
        self.n_nodes = graph.n_nodes
        self.reverse_node_map = graph.reverse_node_map
        self.adj_lists = graph.adj_lists
        self.name = graph.name
        self.dim = dim
        
        self.k = k
        self.pattern = re.compile(r'calls[\t| ]*(\d+),')
        self.base_cmd = "./k-clique -r " + str(self.name) + " -k " + str(self.k) + " -o degeneracy"
        self.learn_cmd = "./k-clique -r " + str(self.name) + " -k " + str(self.k) + " -o learned"

        self.order = None
        self.state = None
        self.done = None
        self.max_deg = None
        self.heap = None
        self.indices = None
        self.deg = None
        self.cur_max = None
        self.cur_deg = None
        self.color_dict = None
        self.reset()

    def reset(self):
        self.order = []
        self.done = False
        self.heap = MaxHeap(self.n_nodes)
        
        for i in range(self.n_nodes):
            deg = len(self.adj_lists[i])
            self.heap.insert(elem=[i, deg])

        self.max_deg = self.heap.max_elem()[1]
        self.top_k = self.heap.top_k(self.dim)
        self.indices = np.array([elem[0] for elem in self.top_k])
        self.deg = np.array([elem[1] for elem in self.top_k])
        self.state = self.deg / self.max_deg

        self.color_dict = dict()
        self.cur_deg = [0] * self.n_nodes
        self.cur_max = 0
        return self.state

    def calls(self, cmd):
        res = self.pattern.findall(os.popen(cmd).read())
        assert len(res) == 1
        return int(res[0])

    def step(self, action):
        tgt_node = self.indices[action]
        self.order.append(tgt_node)

        for i in range(self.dim):
            elem = [self.indices[i], -self.max_deg] if action == i else [self.indices[i], self.deg[i]]
            self.heap.insert(elem)

        for node in self.adj_lists[tgt_node]:
            self.heap.update(node)

        used_color = set()
        for node in self.adj_lists[tgt_node]:
            if node in self.color_dict:
                used_color.add(self.color_dict[node])

        color = 1
        while color in used_color:
            color += 1
        self.color_dict[tgt_node] = color
        
        cur_max = self.cur_max
        for node in self.adj_lists[tgt_node]:
            if node not in self.color_dict:
                continue
            elif self.color_dict[node] > color:
                self.cur_deg[node] += 1
                cur_max = self.cur_deg[node] if self.cur_deg[node] > cur_max else cur_max
            elif self.color_dict[node] < color:
                cur_max = self.cur_deg[tgt_node] if self.cur_deg[tgt_node] > cur_max else cur_max
            else:
                print("Error: two neighbors have same color")
        reward = self.cur_max - cur_max
        self.cur_max = cur_max

        self.top_k = self.heap.top_k(self.dim)
        self.indices = np.array([elem[0] for elem in self.top_k])
        self.deg = np.array([elem[1] for elem in self.top_k])
        self.state = self.deg / self.max_deg
        self.done = self.state[0] < 0

        if self.done:
            with open("./data/" + self.name + ".order", "w") as f:
                for node_index in self.order:
                    f.write(str(self.reverse_node_map[node_index]) + "\n")
            reward += self.calculate_calls()
        
        return reward, self.state, self.done

    def calculate_calls(self):
        return self.calls(self.base_cmd) - self.calls(self.learn_cmd)
