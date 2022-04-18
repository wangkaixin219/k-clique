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
        self.rewards = []
        self.logprobs = []
        self.done = []

    def size(self):
        return len(self.states)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.done[:]


class SageLayer(nn.Module):
    def __init__(self, emb_size, gcn=False):
        super(SageLayer, self).__init__()
        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(emb_size, emb_size if self.gcn else 2 * emb_size))

    def forward(self, self_feat, agg_feat):
        if not self.gcn:
            combined = torch.cat([self_feat, agg_feat], dim=1)
        else:
            combined = agg_feat
        return F.relu(self.weight.mm(combined.t())).t()


class GraphSage(nn.Module):
    def __init__(self, graph, num_layers=2, emb_size=32, gcn=False, agg_func="MEAN"):
        super(GraphSage, self).__init__()
        self.n_nodes = graph.n_nodes
        self.adj_lists = graph.adj_lists
        self.num_layers = num_layers
        self.gcn = gcn
        self.agg_func = agg_func
        self.emb_layer = nn.Embedding(self.n_nodes, emb_size)
        for index in range(1, self.num_layers+1):
            setattr(self, 'sage_layer'+str(index), SageLayer(emb_size, gcn=False))
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, node_batch):
        lower_layer_nodes = list(node_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]

        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers+1

        pre_hidden_embs = self.emb_layer(torch.LongTensor(range(self.n_nodes)).to(device))
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index-1]
            agg_feat = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer'+str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_neighs)
            pre_hidden_embs = sage_layer(self_feat=pre_hidden_embs[nb], agg_feat=agg_feat)

        return pre_hidden_embs

    @staticmethod
    def _nodes_map(nodes, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if num_sample is not None:
            samp_neighs = [set(random.sample(to_neigh, num_sample))
                           if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | {nodes[i]} for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [nodes[i] in samp_neighs[i] for i in range(len(samp_neighs))]
        assert False not in indicator
        if not self.gcn:
            samp_neighs = [(samp_neighs[i] - {nodes[i]}) for i in range(len(samp_neighs))]
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        row_indics = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        mask[row_indics, column_indices] = 1

        agg_feats = None
        if self.agg_func == "MEAN":
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh).to(embed_matrix.device)
            agg_feats = mask.mm(embed_matrix)

        elif self.agg_func == "MAX":
            indices = [x.nonzero() for x in mask == 1]
            agg_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indices]:
                if len(feat.size()) == 1:
                    agg_feats.append(feat.view(1, -1))
                else:
                    agg_feats.append(torch.max(feat, 0)[0].view(1, -1))
            agg_feats = torch.cat(agg_feats, 0)
        return agg_feats


class ActorCritic(nn.Module):
    def __init__(self, graph, num_layers=2, emb_size=32, hidden_size=32, batch_size=16, gcn=False, agg_func="MEAN"):
        super(ActorCritic, self).__init__()

        self.n_nodes = graph.n_nodes
        self.batch_size = batch_size
        self.embeddings = None

        self.graph_sage = GraphSage(graph, num_layers, emb_size, gcn, agg_func)
        self.actor = nn.Sequential(nn.Linear(emb_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, self.n_nodes))
        self.critic = nn.Sequential(nn.Linear(emb_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def _emb(self):
        all_nodes = np.arange(self.n_nodes)
        batches = math.ceil(len(all_nodes) / self.batch_size)
        for index in range(batches):
            batch_nodes = all_nodes[index * self.batch_size: (index+1) * self.batch_size]
            batch_embeddings = self.graph_sage(batch_nodes)
            self.embeddings = batch_embeddings if index == 0 else torch.cat((self.embeddings, batch_embeddings), dim=0)

    def act(self, state, prob):
        if self.embeddings is None:
            self._emb()

        index = np.where(state.cpu() == 1)[0]
        embedding = torch.mean(self.embeddings[index], dim=0).to(device)
        score = self.actor(embedding)
        score = score.masked_fill(state == 0, -1e9)

        action_probs = F.softmax(score, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample() if random.random() < prob else torch.argmax(action_probs)
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, states, actions):
        self._emb()

        indices = [np.where(state.cpu() == 1)[0] for state in states]
        embedding = [torch.mean(self.embeddings[index], dim=0) for index in indices]
        embedding = torch.stack(embedding).to(device)
        score = self.actor(embedding)
        score = score.masked_fill(states == 0, -1e9)

        action_probs = F.softmax(score, dim=-1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(embedding)

        return action_logprobs, state_values, dist_entropy

    def forward(self):
        raise NotImplementedError


class Brain(object):
    def __init__(self, graph, num_layers=2, emb_size=32, hidden_size=32, batch_size=16, gcn=False, agg_func="MEAN",
                 lr=1e-2, gamma=0.99, K_epochs=10, eps_clip=0.2):

        self.policy = ActorCritic(graph, num_layers, emb_size, hidden_size, batch_size, gcn, agg_func).to(device)
        self.policy_old = ActorCritic(graph, num_layers, emb_size, hidden_size, batch_size, gcn, agg_func).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
                {'params': self.policy.graph_sage.parameters(), 'lr': 0.01},
                {'params': self.policy.actor.parameters(), 'lr': 0.01},
                {'params': self.policy.critic.parameters(), 'lr': 0.01},
        ])
        self.MseLoss = nn.MSELoss()

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = Memory()

    def calm_down(self):
        self.policy.embeddings = None
        self.policy_old.embeddings = None

    def select_action(self, state, prob=0.4):
        with torch.no_grad():
            state = torch.LongTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state, prob)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
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
        print(rewards)

        states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        for i in range(self.K_epochs):
            rand_idx = random.sample(range(self.buffer.size()), 128)

            old_rewards = rewards[rand_idx]
            old_states = states[rand_idx]
            old_actions = actions[rand_idx]
            old_logprobs = logprobs[rand_idx]

            logprobs_, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs_ - old_logprobs.detach())

            advantages = old_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, old_rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            print("*** Update [{}/{}] Loss = {} ***".format(i+1, self.K_epochs, loss.mean().item()))

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
