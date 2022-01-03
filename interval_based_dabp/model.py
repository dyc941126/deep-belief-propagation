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
    def __init__(self, in_channels, out_channels, ass_to_sum_hid_dim, sum_to_ass_hid_dim, edge_dim, num_gru_layers,
                 ass_to_sum_inp_dim=1, sum_to_ass_inp_dim=1, num_heads=1):
        super().__init__()
        self.gru1 = nn.GRU(input_size=ass_to_sum_inp_dim, hidden_size=ass_to_sum_hid_dim, num_layers=num_gru_layers,
                           batch_first=True)
        self.gru2 = nn.GRU(input_size=sum_to_ass_inp_dim, hidden_size=sum_to_ass_hid_dim, num_layers=num_gru_layers,
                           batch_first=True)
        self.num_gru_layers = num_gru_layers
        self.gru1_proj = nn.Linear(num_gru_layers * ass_to_sum_hid_dim + 1, edge_dim)
        self.gru2_proj = nn.Linear(num_gru_layers * sum_to_ass_hid_dim, edge_dim)

        self.conv1 = GATConv(in_channels, 8, heads=4, edge_dim=edge_dim, concat=True)
        self.conv2 = GATConv(32, 8, 4, edge_dim=edge_dim, concat=True)
        self.conv3 = GATConv(32, 8, 4, edge_dim=edge_dim, concat=True)
        self.conv4 = GATConv(32, out_channels, 4, edge_dim=edge_dim, concat=False)
        self.attentions = nn.ModuleList()
        for _ in range(num_heads):
            self.attentions.append(Attention(out_channels))

    def forward(self, x, edge_index, local_costs, ass_to_sum_msg,
                ass_to_sum_hidden, sum_to_ass_msg, sum_to_ass_hidden,
                scatter_indexes, scatter_dom_size,
                neighbor_idx_info  # {x_i: {idx: []}}
                ):
        # update edge hidden
        _, hidden1 = self.gru1(ass_to_sum_msg, ass_to_sum_hidden)  # hidden: [num_gru_layers, edges, hidden_dim]
        flatted_hidden1 = hidden1.permute(1, 0, 2)  # [edges, num_gru_layers, hidden_dim]
        flatted_hidden1 = flatted_hidden1.reshape(-1, self.num_gru_layers * hidden1.shape[-1])  # [edges, num_gru_layers * hidden_dim]
        edge_attr1 = self.gru1_proj(torch.cat([local_costs, flatted_hidden1], dim=1))

        _, hidden2 = self.gru2(sum_to_ass_msg, sum_to_ass_hidden)
        flatted_hidden2 = hidden2.permute(1, 0, 2)
        flatted_hidden2 = flatted_hidden2.reshape(-1, self.num_gru_layers * hidden2.shape[-1])
        edge_attr2 = self.gru2_proj(flatted_hidden2)

        assert edge_attr1.shape[1] == edge_attr2.shape[1]
        assert edge_attr1.shape[0] + edge_attr2.shape[0] == edge_index.shape[1]
        edge_attr = torch.cat([edge_attr1, edge_attr2], dim=0)

        # graph conv
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.leaky_relu(x)

        # pooling
        pooling = scatter_sum(x, scatter_indexes, dim=0)[1:]  # each row is the sum of embeddings for a set of summarize node
        pooling = pooling / scatter_dom_size

        # attention score
        num_msg_directions = pooling.shape[0]  # should be 2 times of # of factors
        query = pooling.repeat_interleave(num_msg_directions, dim=0)
        key = pooling.repeat(num_msg_directions, 1)
        attention_scores = []
        for m in self.attentions:
            attention_scores.append(m(query, key))
        attention_scores = torch.cat(attention_scores, dim=1)

        # attention weights
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
