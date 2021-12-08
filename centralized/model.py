import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_sum


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.query_proj = nn.Linear(hidden, hidden, bias=False)
        self.key_proj = nn.Linear(hidden, hidden, bias=False)
        self.score_proj = nn.Linear(2 * hidden, 1)

    def forward(self, query, key):
        query = self.query_proj(query)
        key = self.key_proj(key)
        return F.sigmoid(self.score_proj(torch.cat([query, key], dim=1)))


class AttentiveBP(nn.Module):
    def __init__(self, in_channels, out_channels, ass_to_sum_inp_dim, ass_to_sum_hid_dim, sum_to_ass_inp_dim,
                 sum_to_ass_hid_dim, num_heads=1):
        super().__init__()
        self.gru1 = nn.GRUCell(ass_to_sum_inp_dim, ass_to_sum_hid_dim)
        self.gru2 = nn.GRUCell(sum_to_ass_inp_dim, sum_to_ass_hid_dim)
        self.conv1 = GATConv(in_channels, 8, heads=4, edge_dim=sum_to_ass_hid_dim, concat=True)
        self.conv2 = GATConv(32, 8, 4, edge_dim=sum_to_ass_hid_dim, concat=True)
        self.conv3 = GATConv(32, 8, 4, edge_dim=sum_to_ass_hid_dim, concat=True)
        self.conv4 = GATConv(32, out_channels, 4, edge_dim=sum_to_ass_hid_dim, concat=False)
        self.attentions = nn.ModuleList()
        for _ in range(num_heads):
            self.attentions.append(Attention(out_channels))
        self.grads = dict()

    def forward(self, x, edge_index, ass_to_sum_prefix, sum_to_ass_prefix, local_costs, ass_to_sum_msg,
                ass_to_sum_hidden, sum_to_ass_msg, sum_to_ass_hidden,
                scatter_indexes, scatter_dom_size,
                neighbor_idx_info  # {x_i: {idx: []}}
                ):
        # update edge hidden
        hidden1 = self.gru1(ass_to_sum_msg, ass_to_sum_hidden)
        hidden1.register_hook(save_grad(self.grads, 'hidden1'))
        hidden2 = self.gru2(sum_to_ass_msg, sum_to_ass_hidden)
        hidden2.register_hook(save_grad(self.grads, 'hidden2'))

        # construct edge feature
        if ass_to_sum_prefix is not None:
            edge_attr1 = torch.cat([ass_to_sum_prefix, local_costs, hidden1], dim=1)
        else:
            edge_attr1 = torch.cat([local_costs, hidden1], dim=1)
        edge_attr1.register_hook(save_grad(self.grads, 'edge_attr1'))
        if sum_to_ass_prefix is not None:
            edge_attr2 = torch.cat([sum_to_ass_prefix, hidden2], dim=1)
        else:
            edge_attr2 = hidden2
        edge_attr2.register_hook(save_grad(self.grads, 'edge_attr2'))
        assert edge_attr1.shape[1] == edge_attr2.shape[1]
        assert edge_attr1.shape[0] + edge_attr2.shape[0] == edge_index.shape[1]
        edge_attr = torch.cat([edge_attr1, edge_attr2], dim=0)
        edge_attr.register_hook(save_grad(self.grads, 'edge_attr'))

        # graph conv
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x.register_hook(save_grad(self.grads, 'x'))

        # pooling
        pooling = scatter_sum(x, scatter_indexes, dim=0)[1:]  # each row is the sum of embeddings for a set of summarize node
        pooling = pooling / scatter_dom_size
        pooling.register_hook(save_grad(self.grads, 'pooling'))

        # attention score
        num_msg_directions = pooling.shape[0]  # should be 2 times of # of factors
        query = pooling.repeat_interleave(num_msg_directions, dim=0)
        query.register_hook(save_grad(self.grads, 'query'))
        key = pooling.repeat(num_msg_directions, 1)
        key.register_hook(save_grad(self.grads, 'key'))
        attention_scores = []
        for m in self.attentions:
            attention_scores.append(m(query, key))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores.register_hook(save_grad(self.grads, 'attention_score'))

        # attention weights
        attention_weights = dict()
        for x_i, target in neighbor_idx_info.items():
            attention_weights[x_i] = dict()
            self.grads[x_i] = dict()
            for idx in target.keys():
                src = target[idx]
                if type(src) is list:
                    src = torch.tensor(src, dtype=torch.long, device=x.device, requires_grad=False)
                src = src + idx * num_msg_directions
                scores = attention_scores[src]
                weights = F.softmax(scores, dim=0)
                attention_weights[x_i][idx] = weights * (len(target) - 1)
        return attention_weights, hidden1, hidden2
