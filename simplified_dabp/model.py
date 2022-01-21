import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_sum


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
    def __init__(self, in_channels, out_channels, num_heads=1, prefix_dim=3):
        super().__init__()
        self.gru1 = nn.GRUCell(1, in_channels - prefix_dim)
        self.gru2 = nn.GRUCell(1, in_channels - prefix_dim)
        self.conv1 = GATConv(in_channels, 8, heads=4, concat=True)
        self.conv2 = GATConv(32, 8, 4, concat=True)
        self.conv3 = GATConv(32, 8, 4, concat=True)
        self.conv4 = GATConv(32, out_channels, 4, concat=False)
        self.attentions = nn.ModuleList()
        for _ in range(num_heads):
            self.attentions.append(Attention(out_channels))

    def forward(self, x, edge_index, ass_to_sum_msg, ass_to_sum_hidden, sum_to_ass_msg,
                sum_to_ass_hidden, prefix, function_node_idxes, neighbor_idx_info  # {x_i: {idx: []}}
                ):
        # update edge hidden
        hidden1 = self.gru1(ass_to_sum_msg, ass_to_sum_hidden)
        hidden2 = self.gru2(sum_to_ass_msg, sum_to_ass_hidden)

        hidden = torch.cat([hidden1, hidden2], dim=0)
        hidden = torch.cat([prefix, hidden], dim=1)

        x = torch.cat([x, hidden], dim=0)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)

        pooling = x[function_node_idxes]

        num_msg_directions = pooling.shape[0]
        query = pooling.repeat_interleave(num_msg_directions, dim=0)
        key = pooling.repeat(num_msg_directions, 1)
        attention_scores = []
        for m in self.attentions:
            attention_scores.append(m(query, key))
        attention_scores = torch.cat(attention_scores, dim=1)

        attention_weights = dict()
        for x_i, target in neighbor_idx_info.items():
            attention_weights[x_i] = dict()
            for idx in target.keys():
                src = target[idx]
                if type(src) is list:
                    src = torch.tensor(src, dtype=torch.long, device=x.device, requires_grad=False)
                src = src + idx * num_msg_directions
                scores = attention_scores[src]
                weights = F.softmax(scores, dim=0)
                attention_weights[x_i][idx] = weights * (len(target) - 1)
        return attention_weights, hidden1, hidden2