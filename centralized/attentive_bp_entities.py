import random

import torch

from entities import VariableNode, FunctionNode


def msg_to_tensor(msg, device):
    if type(msg) is list:
        msg = torch.tensor(msg, dtype=torch.float32, device=device)
    else:
        msg = msg.detach()
    return msg.squeeze()


class FeatureConstructor:
    vn_id_embed_ub = 0.5

    def __init__(self, variable_nodes, function_nodes, assignment_node_prefix=None, summarize_node_prefix=None,
                 node_embed_dim=8, ass_to_sum_hidden=8, sum_to_ass_hidden=8, device='cpu',
                 ass_to_sum_prefix=None, sum_to_ass_prefix=None):
        self.vn_node_index = dict()  # {x_i: [START, END) of assignment nodes} in self.x
        self.fn_node_index = dict()  # {f_ij: {x_i: [START, END) of summarize nodes to x_j}} in self.x
        self.neighbor_idx_info = dict()  # {x_i: {f_ij.idx: [fik.idx]}}
        self.neighbor_idx_mapping = dict()  # {x_i: {f_ij.name: f_ij.idx}}
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
        scatter_indexes = []  # used for pooling: specifies the group each row of embedding matrix belongs to
        for vn in variable_nodes:
            self.neighbor_idx_info[vn.name] = dict()
            self.neighbor_idx_mapping[vn.name] = dict()
            start = len(x)
            vn_id_embed = [random.uniform(0, FeatureConstructor.vn_id_embed_ub) for _ in range(vn_id_embed_len)]
            vn_embed = assignment_node_prefix + vn_id_embed
            for _ in range(vn.dom_size):
                x.append(vn_embed)
                scatter_indexes.append(0)  # we do not care assignment nodes, just assign 0
            end = len(x)
            self.vn_node_index[vn.name] = (start, end)

        # build init node embedding for summarize nodes
        scatter_dom_size = [1]  # used for pooling
        idx = 1
        sn_id_embed_len = node_embed_dim - len(summarize_node_prefix)
        sn_embed = summarize_node_prefix + [0 for _ in range(sn_id_embed_len)]  # 0 for padding
        for fn in function_nodes:
            i = fn.row_vn.name
            j = fn.col_vn.name
            start = len(x)
            for _ in range(fn.col_vn.dom_size):
                x.append(sn_embed)
                scatter_indexes.append(idx)  # all f_ij -> x_j summarize nodes belong to the same group
                scatter_dom_size.append(fn.col_vn.dom_size)
            end = len(x)
            idx += 1  # increase group id
            self.fn_node_index[fn.name] = {i: (start, end)}
            start = len(x)
            for _ in range(fn.row_vn.dom_size):
                x.append(sn_embed)
                scatter_indexes.append(idx)  # all f_ij -> x_i summarize nodes belong to the same group
                scatter_dom_size.append(fn.row_vn.dom_size)
            end = len(x)
            idx += 1
            self.fn_node_index[fn.name][j] = (start, end)
        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.scatter_indexes = torch.tensor(scatter_indexes, torch.long, device=device)
        self.scatter_dom_size = torch.tensor(scatter_dom_size, device=device).unsqueeze(1)

        # build edge index
        edge_index = [[], []]
        src, dest = edge_index
        ass_to_sum_cnt = sum_to_ass_cnt = 0
        self.local_costs = []
        # assignment node -> summarize node
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
        # summarize node -> assignment node
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
        self.attentive_weights = dict()  # returned by neural network

        for vn in variable_nodes:
            in_idxes = []
            out_indexes = []
            for fn in vn.neighbors:
                idx = self.function_nodes.index(fn)
                idx *= 2
                if vn == fn.row_vn:
                    in_idxes.append(idx + 1)
                    out_indexes.append(idx)
                else:
                    in_idxes.append(idx)
                    out_indexes.append(idx + 1)
                self.neighbor_idx_mapping[vn.name][fn.name] = out_indexes[-1]
            for i in range(len(out_indexes)):
                idxes = list(in_idxes)
                idxes[i] = out_indexes[i]
                self.neighbor_idx_info[vn.name][out_indexes[i]] = idxes

        all_nodes = list(variable_nodes) + self.function_nodes
        for node in all_nodes:
            node.register_feature_extractor(self)

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


class AttentiveVariableNode(VariableNode):
    def __init__(self, name, dom_size):
        super().__init__(name, dom_size)
        self.feature_extractor = None
        self.ordered_neighbors = []
        self.device = ''
        self.distribution = None

    def register_feature_extractor(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.device = feature_extractor.device
        for n in self.feature_extractor.function_nodes:
            if n.name in self.neighbors:
                self.ordered_neighbors.append(n.name)

    def compute_msgs(self):
        all_income_messages = [self.incoming_msg[n].detach() for n in self.ordered_neighbors]
        for i, target in enumerate(self.ordered_neighbors):
            msgs = list(all_income_messages)
            msgs[i] = self.prev_sent[target] if target in self.prev_sent else torch.zeros(self.dom_size, dtype=torch.float32, device=self.device)
            msgs = torch.stack(msgs, dim=1)
            results = torch.mm(msgs, self.feature_extractor.attentive_weights[self.name][self.feature_extractor.neighbor_idx_mapping[self.name][target]])
            results = results.mean(dim=1)
            results = results - results.min().item()
            self.prev_sent[target] = results.detach()
            self.neighbors[target].incoming_msg[self.name] = results

    def compute_belief(self):
        all_income_messages = torch.stack([self.incoming_msg[n] for n in self.ordered_neighbors], dim=0)
        belief = all_income_messages.sum(dim=0)
        self.distribution = torch.softmax(belief, dim=0)

    def compute_local_loss(self):
        loss = 0
        i_dist = self.distribution.unsqueeze(0)
        for fn in self.neighbors.values():
            if self == fn.row_vn:
                j_dist = fn.col_vn.distribution.unsqueeze(1)
                expected_cost = torch.mm(i_dist, fn.data)
                expected_cost = torch.mm(expected_cost, j_dist)
                loss = loss + expected_cost
        return loss


class AttentiveFunctionNode(FunctionNode):
    def __init__(self, name, matirx, row_vn, col_vn):
        super().__init__(name, matirx, row_vn, col_vn)
        self.device = ''
        self.data = None

    def register_feature_extractor(self, feature_extractor):
        self.device = feature_extractor.device
        self.data = torch.tensor(self.matrix, dtype=torch.float32, device=self.device)

