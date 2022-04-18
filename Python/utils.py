import sys
import os
import torch
import random
import math
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from brain import *
from env import *

def train(graph, args):
    brain = Brain(graph, num_layers=2, emb_size=128, hidden_size=128, batch_size=16, gcn=False, agg_func="MEAN", lr=1e-3, gamma=0.99, K_epochs=10, eps_clip=0.2)
    env = Env(graph, args.k)
    
    epoch, n_epoch = 1, 10000
    best = 0
    prob = 1
    disc = 0.9

    while epoch <= n_epoch:
        state = env.reset()

        if epoch >= 10:
            for param in brain.policy.graph_sage.parameters():
                param.requires_grad = False
            for param in brain.policy_old.graph_sage.parameters():
                param.requires_grad = False

        brain.calm_down()
        rewards = 0
        while not env.done:
            action = brain.select_action(state, prob=prob)
            reward, state, done = env.step(action)
            brain.buffer.rewards.append(reward)
            brain.buffer.done.append(done)
            rewards += reward

        calls = env.calculate_calls()
        brain.buffer.rewards[-1] += calls
        if calls > best:
            best = calls
            brain.save('best.pt'.format(epoch))
            print("Save current best model at epoch {}".format(epoch))

        print("******************** Epoch [{}/{}] ********************".format(epoch, n_epoch))
        print("Rewards = {}, Calls = {}".format(rewards, calls))
        brain.update()
        print("*******************************************************\n")

        if epoch % 10 == 0:
            prob *= disc
            print("Explore prob: {}".format(prob))
            validate(graph, args)

        epoch += 1

def validate(graph, args):

    brain = Brain(graph, num_layers=2, emb_size=128, hidden_size=128, batch_size=16, gcn=False, agg_func="MEAN", lr=1e-3, gamma=0.99, K_epochs=1, eps_clip=0.2)
    env = Env(graph, args.k)
    
    brain.load('best.pt')
    state = env.reset()
    brain.calm_down()

    rewards = 0
    while not env.done:
        action = brain.select_action(state, prob=0)
        reward, state, done = env.step(action)
        rewards += reward

    calls = env.calculate_calls()

    print("*************** Validate ***************")
    print("Rewards = {}, Calls = {}".format(rewards, calls))
    print("****************************************\n")


