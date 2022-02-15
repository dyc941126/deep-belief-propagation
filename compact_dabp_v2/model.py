import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.query_proj = nn.Linear(hidden, hidden, bias=False)
        self.key_proj = nn.Linear(hidden, hidden, bias=False)
        self.score_proj = nn.Linear(2 * hidden, 1)

    def forward(self, query, key):
        query = self.query_proj(query)
        key = self.key_proj(key)
        return torch.sigmoid(self.score_proj(torch.cat([query, key], dim=1)))


class AttentiveBP(nn.Module):
    def __init__(self, in_channels, out_channels, dom_size, num_heads=1, prefix_dim=3, num_color=100):
        super().__init__()
        self.color_embed = nn.Embedding(num_color, in_channels - prefix_dim)
        self.gru_v2f = nn.GRUCell(dom_size, in_channels - prefix_dim)
        self.gru_f2v = nn.GRUCell(dom_size, in_channels - prefix_dim)
        self.conv1 = GATConv(in_channels, 8, heads=4, concat=True)
        self.conv2 = GATConv(32, 8, 4, concat=True)
        self.conv3 = GATConv(32, 8, 4, concat=True)
        self.conv4 = GATConv(32, out_channels, 4, concat=False)
        self.attentions = nn.ModuleList()
        for _ in range(num_heads):
            self.attentions.append(Attention(out_channels))

    def forward(self, edge_index, vn_colors, vn_prefix, fn_embed, v2f_msgs, v2f_hidden, f2v_msgs, f2v_hidden, msg_prefix, neighbor_idx_info  # {x_i: {idx: []}}
                ):
        x = self.color_embed(vn_colors)
        x = torch.cat([vn_prefix, x], dim=1)

        fn_start = x.shape[0]
        fn_end = fn_start + fn_embed.shape[0]

        x = torch.cat([x, fn_embed], dim=0)

        # update edge hidden
        v2f_hidden = self.gru_v2f(v2f_msgs, v2f_hidden)
        f2v_hidden = self.gru_f2v(f2v_msgs, f2v_hidden)

        hidden = torch.cat([v2f_hidden, f2v_hidden], dim=0)
        hidden = torch.cat([msg_prefix, hidden], dim=1)

        x = torch.cat([x, hidden], dim=0)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)

        pooling = x[fn_start:fn_end]

        num_factors = pooling.shape[0]
        query = pooling.repeat_interleave(num_factors, dim=0)
        key = pooling.repeat(num_factors, 1)
        attention_scores = []
        for m in self.attentions:
            attention_scores.append(m(query, key))
        attention_scores = torch.cat(attention_scores, dim=1)

        attention_weights = dict()
        for x_i, target in neighbor_idx_info.items():
            attention_weights[x_i] = dict()
            for idx in target.keys():
                src = target[idx]
                if len(src) == 1:
                    attention_weights[x_i][idx] = torch.zeros(1, attention_scores.shape[1], dtype=torch.float32, device=x.device)
                    continue
                out_i = src.index(idx)

                src = [ii for ii in src if ii != idx]
                assert len(src) == len(target[idx]) - 1

                if type(src) is list:
                    src = torch.tensor(src, dtype=torch.long, device=x.device, requires_grad=False)
                src = src + idx * num_factors
                scores = attention_scores[src]
                trg_score = attention_scores[idx + idx * num_factors]
                score_sum = torch.mean(scores, dim=0)
                total_weight = F.softmax(torch.stack([score_sum, trg_score], dim=0), dim=0)

                weights = F.softmax(scores, dim=0)
                weights = weights * total_weight[0].unsqueeze(0)
                weights = weights * (len(target) - 1)

                trg_weights = total_weight[1].unsqueeze(0)

                weights = torch.cat([weights[: out_i], trg_weights, weights[out_i:]], dim=0)
                attention_weights[x_i][idx] = weights
        return attention_weights, v2f_hidden, f2v_hidden