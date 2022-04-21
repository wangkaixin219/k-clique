import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import numpy as np
import random


if torch.cuda.is_available():
    device = torch.device("cuda")
    device_id = torch.cuda.current_device()
    print('DEVICE ', device_id, torch.cuda.get_device_name(device_id))
else:
    device = torch.device("cpu")
    print('DEVICE ', device)

'''
    States are defined as a set of masks. 
    For example, there are 5 nodes, and at the first round, state = [1, 1, 1, 1, 1]
    Graph embedding model takes masks and the node embeddings to generate intermediate embedding of the graph
    Actions are made based on the embedding of the graph; 
    After taking the action, we change the corresponding index of the state to 0
    Finally, the states should be all zeros; then we shall reset the environment and update the weights.
'''


class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.masks = []
        self.rewards = []
        self.logprobs = []
        self.done = []

    def size(self):
        return len(self.states)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.masks[:]
        del self.logprobs[:]
        del self.done[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim=10, action_dim=10, hidden_dim=32):
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, self.action_dim))
        self.critic = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def act(self, state, prob):
        score = self.actor(state)
        score = score.masked_fill(state < 0, -1e9)
        action_probs = F.softmax(score, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample() if random.random() < prob else torch.argmax(action_probs)
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, states, actions):
        scores = self.actor(states)
        scores = scores.masked_fill(states < 0, -1e9)
        action_probs = F.softmax(scores, dim=-1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        return action_logprobs, state_values, dist_entropy

    def forward(self):
        raise NotImplementedError


class Brain(object):
    def __init__(self, state_dim=10, action_dim=10, hidden_dim=32, batch_size=128, gamma=0.99, K_epochs=10, eps_clip=0.2):

        self.batch_size = batch_size
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': 1e-4},
                {'params': self.policy.critic.parameters(), 'lr': 1e-3},
        ])
        self.MseLoss = nn.MSELoss()

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = Memory()

    def select_action(self, state, prob=0.4):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state, prob)
        self.buffer.states.append(state.cpu())
        self.buffer.actions.append(action.cpu())
        self.buffer.logprobs.append(action_logprob.cpu())
        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.done)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        batches = math.ceil(self.buffer.size() / self.batch_size)

        for i in range(self.K_epochs):
            loss_epoch = 0
            for index in range(batches):
                old_states = states[index * self.batch_size: (index + 1) * self.batch_size].to(device)
                old_actions = actions[index * self.batch_size: (index + 1) * self.batch_size].to(device)
                old_rewards = rewards[index * self.batch_size: (index + 1) * self.batch_size].to(device)
                old_logprobs = logprobs[index * self.batch_size: (index + 1) * self.batch_size].to(device)

                logprobs_, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

                state_values = torch.squeeze(state_values)
                ratios = torch.exp(logprobs_ - old_logprobs.detach())

                advantages = old_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, old_rewards) - 0.01 * dist_entropy
                loss_epoch += loss.mean().item()

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            print("*** Update Round [{}/{}] Loss = {} ***".format(i + 1, self.K_epochs, loss_epoch))

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
