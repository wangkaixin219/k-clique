import sys
import os
import torch
import random
import math
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

from env import *

def train(graph, graph_model, brain, batch_size):
    all_nodes = np.arange(graph.n_nodes)
    env = Env(graph)
    epoch = 1
    n_epoch = 50

    while epoch <= n_epoch:
        env.reset()

        while not env.done:
            act_nodes = all_nodes[env.act_index]
            batches = math.ceil(len(act_nodes) / batch_size)
            embeddings = []

            for index in range(batches):
                batch_nodes = act_nodes[index*batch_size:(index+1)*batch_size]
                batch_embeddings = graph_model(batch_nodes)
                embeddings.append(batch_embeddings)

            state = torch.mean(torch.stack(embeddings, dim=0), dim=0)
            action = brain.select_action(state, env.mask)
            env.step(action)

        reward = env.calculate_reward()
        print("Epoch {}: reward {}".format(epoch, reward))
        brain.buffer.fill(reward)
        brain.update()

        epoch += 1




'''
models = [graph_model]
params = []
for model in models:
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

optimizer = torch.optim.Adam(params, lr=0.01)
optimizer.zero_grad()
for model in models:
    model.zero_grad()
    
    optimizer.zero_grad()
for model in models:
    model.zero_grad()
'''