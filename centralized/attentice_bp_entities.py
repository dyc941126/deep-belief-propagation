import random

import torch


def msg_to_tensor(msg, device):
    if type(msg) is list:
        msg = torch.tensor(msg, dtype=torch.float32, device=device)
    else:
        msg = msg.detach()
    return msg.squeeze()


class Feature:
    vn_id_embed_ub = 0.5

    def __init__(self, variable_nodes, function_nodes, assignment_node_prefix=None, summarize_node_prefix=None,
                 node_embed_dim=8, ass_to_sum_hidden=8, sum_to_ass_hidden=8, device='cpu',
                 ass_to_sum_prefix=None, sum_to_ass_prefix=None):
        self.vn_node_index = dict()  # {x_i: [START, END) of assignment nodes}
        self.fn_node_index = dict()  # {f_ij: {x_i: [START, END) of summarize nodes to x_j}}
        function_nodes = list(function_nodes)
        self.function_nodes = function_nodes
        if assignment_node_prefix is None:
            assignment_node_prefix = [1, 0]
        if summarize_node_prefix is None:
            summarize_node_prefix = [0, 1]
        assert summarize_node_prefix != assignment_node_prefix
        assert node_embed_dim > len(summarize_node_prefix) and node_embed_dim > len(assignment_node_prefix)
        vn_id_embed_len = node_embed_dim - len(assignment_node_prefix)
        x = []

        # build init node embedding for assignment nodes
        for vn in variable_nodes:
            start = len(x)
            vn_id_embed = [random.uniform(0, Feature.vn_id_embed_ub) for _ in range(vn_id_embed_len)]
            vn_embed = assignment_node_prefix + vn_id_embed
            for _ in range(vn.dom_size):
                x.append(vn_embed)
            end = len(x)
            self.vn_node_index[vn.name] = (start, end)

        # build init node embedding for summarize nodes
        sn_id_embed_len = node_embed_dim - len(summarize_node_prefix)
        sn_embed = summarize_node_prefix + [0 for _ in range(sn_id_embed_len)]
        for fn in function_nodes:
            i = fn.row_vn.name
            j = fn.col_vn.name
            start = len(x)
            for _ in range(fn.col_vn.dom_size):
                x.append(sn_embed)
            end = len(x)
            self.fn_node_index[fn.name] = {i: (start, end)}
            start = len(x)
            for _ in range(fn.row_vn.dom_size):
                x.append(sn_embed)
            end = len(x)
            self.fn_node_index[fn.name][j] = (start, end)
        self.x = torch.tensor(x, dtype=torch.float32, device=device)

        # build edge index
        edge_index = [[], []]
        src, dest = edge_index
        ass_to_sum_cnt = sum_to_ass_cnt = 0
        self.local_costs = []
        for fn in function_nodes:
            i = fn.row_vn.name
            j = fn.col_vn.name

            for val in range(fn.col_vn.dom_size):
                ass_to_sum_cnt += fn.row_vn.dom_size
                self.local_costs += [fn.matrix[k][val] for k in range(fn.row_vn.dom_size)]
                # x_i -> f_ij
                src += [idx for idx in range(*self.vn_node_index[i])]
                dest += [self.fn_node_index[fn.name][i][0] + val] * len(fn.row_vn.dom_size)

            for val in range(fn.row_vn.dom_size):
                ass_to_sum_cnt += fn.col_vn.dom_size
                self.local_costs += [fn.matrix[val][k] for k in range(fn.col_vn.dom_size)]
                # x_j -> f_ij
                src += [idx for idx in range(*self.vn_node_index[j])]
                dest += [self.fn_node_index[fn.name][j][0] + val] * len(fn.col_vn.dom_size)

        for fn in function_nodes:
            i = fn.row_vn.name
            j = fn.col_vn.name
            for val in range(fn.col_vn.dom_size):
                src.append(self.fn_node_index[fn.name][i][0] + val)
                dest.append(self.vn_node_index[j][0] + val)
                sum_to_ass_cnt += 1

            for val in range(fn.row_vn.dom_size):
                src.append(self.fn_node_index[fn.name][j][0] + val)
                dest.append(self.vn_node_index[i][0] + val)
                sum_to_ass_cnt += 1

        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        self.ass_to_sum_hidden = torch.zeros(ass_to_sum_cnt, ass_to_sum_hidden, device=device)
        self.sum_to_ass_hidden = torch.zeros(sum_to_ass_cnt, sum_to_ass_hidden, device=device)
        self.local_costs = torch.tensor(self.local_costs, dtype=torch.float32, device=device).unsqueeze(1)
        self.ass_to_sum_prefix = None
        if ass_to_sum_prefix is not None:
            self.ass_to_sum_prefix = torch.tensor(ass_to_sum_prefix, dtype=torch.float32, device=device).repeat(ass_to_sum_cnt, 1)
        if sum_to_ass_prefix is not None:
            self.sum_to_ass_prefix = torch.tensor(sum_to_ass_prefix, dtype=torch.float32, device=device).repeat(sum_to_ass_cnt, 1)
        self.device = device

    def update_message(self):
        ass_to_sum = []
        sum_to_ass = []
        for fn in self.function_nodes:
            i = fn.row_vn
            j = fn.col_vn
            # x_i -> f_ij
            msg = msg_to_tensor(fn.income_msg[i.name], self.device)
            msg = msg.repeat(j.dom_size)
            ass_to_sum.append(msg)
            # f_ij -> x_j
            msg = msg_to_tensor(j.income_msg[fn.name], self.device)
            sum_to_ass.append(msg)

            # x_j -> f_ij
            msg = msg_to_tensor(fn.income_msg[j.name], self.device)
            msg = msg.repeat(i.dom_size)
            ass_to_sum.append(msg)
            # f_ij -> x_i
            msg = msg_to_tensor(i.income_msg[fn.name], self.device)
            sum_to_ass.append(msg)
        return torch.cat(ass_to_sum).unsqueeze(1), torch.cat(sum_to_ass).unsqueeze(1)