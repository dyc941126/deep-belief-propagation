import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveBP(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x, edge_index, ass_to_sum_prefix, local_costs, ass_to_sum_msg,
                ass_to_sum_hidden, sum_to_ass_prefix, sum_to_ass_msg, sum_to_ass_hidden):
        pass