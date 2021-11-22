import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class AttentiveBPNet(nn.Module):
    def __init__(self, out_channels, heads, slop=.2):
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.att_lin = nn.Linear(out_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.leaky_relu = nn.LeakyReLU(slop)

    def forward(self, x, edge_index, node_idxes):

        x = self.att_lin(x).view(-1, self.heads, self.out_channels)
        res = []
        for idxes in node_idxes:
            value = torch.cat([x[l[0]] for l in idxes])
            key = torch.cat([x[l[1]].repeat(len(idxes), 1, 1) for l in idxes])
            value = value.repeat(len(idxes), 1, 1)
            inp = torch.cat([key, value], dim=-1)
            att_score = (inp * self.att).sum(-1)
            att_score = self.leaky_relu(att_score)
            avg_score = att_score.view(2 * len(idxes), -1, self.heads)
            avg_score = avg_score.mean(dim=1)
            avg_score = avg_score.view(len(idxes), len(idxes), -1)
            proba = torch.softmax(avg_score, dim=1)
            res.append(proba)


if __name__ == '__main__':
    # 2 variables, 3 domain, 4 channels, 5 heads
    model = AttentiveBPNet(4, 5)
    x = torch.rand((12, 4))
    node_idx = [
        [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]]
        ]
    ]
    model.forward(x, None, node_idx)