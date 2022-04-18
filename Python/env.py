import numpy as np
import os
import re


class Env(object):
    def __init__(self, graph, k):
        self.n_nodes = graph.n_nodes
        self.reverse_node_map = graph.reverse_node_map
        self.adj_lists = graph.adj_lists
        self.name = graph.name
        
        self.k = k
        self.pattern = re.compile(r'calls[\t| ]*(\d+),')
        self.base_cmd = "./k-clique -r " + str(self.name) + " -k " + str(self.k) + " -o degeneracy"
        self.learn_cmd = "./k-clique -r " + str(self.name) + " -k " + str(self.k) + " -o learned"

        self.order = None
        self.act_index = None
        self.state = None
        self.done = None
        self.color_dict = None
        self.max_color = None
        self.reset()

    def reset(self):
        self.order = []
        self.act_index = list(range(self.n_nodes))
        self.state = [1] * self.n_nodes
        self.done = False
        self.color_dict = dict()
        self.max_color = 1
        return self.state

    def calls(self, cmd):
        res = self.pattern.findall(os.popen(cmd).read())
        assert len(res) == 1
        return int(res[0])

    def step(self, action):
        self.order.append(action)
        self.act_index.remove(action)
        self.state[action] = 0

        used_color = set()
        for node in self.adj_lists[action]:
            if node not in self.color_dict:
                continue
            else:
                used_color.add(self.color_dict[node])

        color = 1
        while color in used_color:
            color += 1
        self.color_dict[action] = color
        reward = 1 if color < self.k else 0
        self.done = not np.any(self.state)

        if self.done:
            with open("./data/" + self.name + ".order", "w") as f:
                for node_index in self.order:
                    f.write(str(self.reverse_node_map[node_index]) + "\n")
        
        return reward, self.state, self.done

    def calculate_calls(self):
        return self.calls(self.base_cmd) - self.calls(self.learn_cmd)

