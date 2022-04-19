import numpy as np
import os
import re


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
        self.deg = None
        self.indices = None
        self.reset()

    def reset(self):
        self.order = []
        self.done = False
        self.deg = np.array([len(self.adj_lists[i]) for i in range(self.n_nodes)])
        self.indices = np.argsort(self.deg)[::-1][:self.dim]
        self.state = self.deg[self.indices]
        return self.state

    def calls(self, cmd):
        res = self.pattern.findall(os.popen(cmd).read())
        assert len(res) == 1
        return int(res[0])

    def step(self, action):
        tgt_node = self.indices[action]
        self.order.append(tgt_node)
        self.deg[tgt_node] = -1

        for inf_node in self.adj_lists[tgt_node]:
            if self.deg[inf_node] == -1:
                continue
            assert self.deg[inf_node] > 0
            self.deg[inf_node] -= 1

        self.indices = np.argsort(self.deg)[::-1][:self.dim]
        self.state = self.deg[self.indices]
        self.done = self.state[0] == -1
        reward = 0

        if self.done:
            with open("./data/" + self.name + ".order", "w") as f:
                for node_index in self.order:
                    f.write(str(self.reverse_node_map[node_index]) + "\n")
            reward += self.calculate_calls()
        
        return reward, self.state, self.done

    def calculate_calls(self):
        return self.calls(self.base_cmd) - self.calls(self.learn_cmd)
