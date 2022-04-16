import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class SageLayer(nn.Module):
    def __init__(self, in_size, out_size, gcn=False):
        super(SageLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.gcn = gcn

        self.weight = nn.Parameter(torch.FloatTensor(self.out_size, self.in_size if self.gcn else 2 * self.in_size))
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feat, agg_feat, neighs=None):
        if not self.gcn:
            combined = torch.cat([self_feat, agg_feat], dim=1)
        else:
            combined = agg_feat
        return F.relu(self.weight.mm(combined.t())).t()


class GraphSage(nn.Module):
    def __init__(self, n_nodes, num_layers, in_size, out_size, adj_lists, device, gcn=False, agg_func="MEAN"):
        super(GraphSage, self).__init__()
        self.n_nodes = n_nodes
        self.in_size = in_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func
        self.adj_lists = adj_lists

        self.emb_layer = nn.Embedding(n_nodes, in_size)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        for index in range(1, self.num_layers+1):
            layer_size = out_size if index != 1 else in_size
            setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=False))

    def forward(self, node_batch):
        lower_layer_nodes = list(node_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]

        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers+1

        pre_hidden_embs = self.emb_layer(torch.LongTensor(range(self.n_nodes)).to(self.device))
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index-1]
            agg_feat = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer'+str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
            pre_hidden_embs = sage_layer(self_feat=pre_hidden_embs[nb], agg_feat=agg_feat)

#        return torch.sum(pre_hidden_embs, dim=0)
        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if not num_sample is None:
            samp_neighs = [set(random.sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [nodes[i] in samp_neighs[i] for i in range(len(samp_neighs))]
        assert False not in indicator
        if not self.gcn:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        row_indics = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        mask[row_indics, column_indices] = 1

        if self.agg_func == "MEAN":
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh).to(embed_matrix.device)
            agg_feats = mask.mm(embed_matrix)

        elif self.agg_func == "MAX":
            indices = [x.nonzero() for x in mask==1]
            agg_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indices]:
                if len(feat.size()) == 1:
                    agg_feats.append(feat.view(1, -1))
                else:
                    agg_feats.append(torch.max(feat, 0)[0].view(1, -1))
            agg_feats = torch.cat(agg_feats, 0)

        return agg_feats
