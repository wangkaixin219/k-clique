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

def train(brain, env):
    epoch, n_epoch = 1, 10000

    while epoch <= n_epoch:
        state = env.reset()
        brain.calm_down()

        while not env.done:
            action = brain.select_action(state)
            state = env.step(action)

        reward = env.calculate_reward()
        print("******************** Epoch [{}/{}] ********************".format(epoch, n_epoch))
        print("Reward = {}".format(reward))
        brain.buffer.fill(reward)
        brain.update()

        epoch += 1

        print("*******************************************************\n")



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
