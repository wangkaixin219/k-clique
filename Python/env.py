import numpy as np
import os
import re


class Env(object):
    def __init__(self, graph):
        self.n_nodes = graph.n_nodes
        self.reverse_node_map = graph.reverse_node_map
        self.name = graph.name
        self.pattern = re.compile(r'calls[\t| ]*(\d+),')

        self.base_cmd = "./k-clique -r " + str(self.name) + " -k 3 -o degeneracy"
        self.learn_cmd = "./k-clique -r " + str(self.name) + " -k 3 -o learned"
        self.reset()


    def reset(self):
        self.order = []
        self.act_index = list(range(self.n_nodes))
        self.mask = np.ones((self.n_nodes,), dtype=int)
        self.done = False

    def calls(self, cmd):
        res = self.pattern.findall(os.popen(cmd).read())
        assert len(res) == 1
        return int(res[0])

    def step(self, action):
        self.order.append(action)
        self.act_index.remove(action)
        self.mask[action] = 0
        self.done = not np.any(self.mask)

        if self.done:
            with open("./data/" + self.name + ".order", "w") as f:
                for node_index in self.order:
                    f.write(str(self.reverse_node_map[node_index]) + "\n")


    def calculate_reward(self):
        return self.calls(self.base_cmd) -  self.calls(self.learn_cmd)


if __name__ == '__main__':
    pass
