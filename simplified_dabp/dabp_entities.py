import torch

from centralized.entities import VariableNode, FunctionNode
from centralized.parser import parse


class AttentiveVariableNode(VariableNode):
    def __init__(self, name, dom_size):
        super().__init__(name, dom_size)
        self.feature_extractor = None
        self.ordered_neighbors = []
        self.device = ''
        self.distribution = None
        self.grads = dict()

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
            msgs[i] = self.prev_sent[target] if target in self.prev_sent else torch.zeros(self.dom_size,
                                                                                          dtype=torch.float32,
                                                                                          device=self.device)
            msgs = torch.stack(msgs, dim=1)
            results = torch.mm(msgs, self.feature_extractor.attentive_weights[self.name][
                self.feature_extractor.neighbor_idx_mapping[self.name][target]])
            results = results.mean(dim=1)
            results = results - results.min().item()
            self.prev_sent[target] = results.detach()
            self.neighbors[target].incoming_msg[self.name] = results

    def compute_distribution(self):
        all_income_messages = torch.stack([self.incoming_msg[n] for n in self.ordered_neighbors], dim=0)
        belief = all_income_messages.sum(dim=0)
        self.distribution = torch.softmax(-belief, dim=0)

    def make_decision(self):
        self.val_idx = self.distribution.argmax().item()

    def compute_local_loss(self):
        normalized_dist = self.distribution + 1e-6
        entropy = -(normalized_dist * torch.log2(normalized_dist)).sum()
        loss = 0.0 * entropy
        i_dist = self.distribution.unsqueeze(0)
        for fn in self.neighbors.values():
            if self == fn.row_vn:
                j_dist = fn.col_vn.distribution.unsqueeze(1)
                expected_cost = torch.mm(i_dist, fn.data)
                expected_cost = torch.mm(expected_cost, j_dist)
                loss = loss + expected_cost.squeeze()
        return loss

    def reset(self):
        for n in self.incoming_msg.keys():
            if not type(self.incoming_msg[n]) is list:
                self.incoming_msg[n] = self.incoming_msg[n].detach().clone()


class AttentiveFunctionNode(FunctionNode):
    def __init__(self, name, matirx, row_vn, col_vn):
        super().__init__(name, matirx, row_vn, col_vn)
        self.device = ''
        self.data = None

    def reset(self):
        for n in self.incoming_msg.keys():
            if not type(self.incoming_msg[n]) is list:
                self.incoming_msg[n] = self.incoming_msg[n].detach().clone()

    def register_feature_extractor(self, feature_extractor):
        self.device = feature_extractor.device
        self.data = torch.tensor(self.matrix, dtype=torch.float32, device=self.device)
        self.incoming_msg[self.row_vn.name] = torch.zeros(self.row_vn.dom_size, device=self.device)
        self.incoming_msg[self.col_vn.name] = torch.zeros(self.col_vn.dom_size, device=self.device)

    def compute_msgs(self):
        neighbors = [self.row_vn, self.col_vn]
        for vn in neighbors:
            oppo = [v for v in neighbors if v != vn][0]
            msg = self.incoming_msg[oppo.name]
            if oppo == self.row_vn:
                msg = msg.unsqueeze(1)
                min_dim = 0
            else:
                msg = msg.unsqueeze(0)
                min_dim = 1
            data = self.data + msg
            data, _ = data.min(min_dim)
            vn.incoming_msg[self.name] = data


def _matrix_multiply(matrix, coefficient):
    res = []
    for row in matrix:
        res.append([x * coefficient for x in row])
    return res


class AttentiveFactorGraph:
    def __init__(self, pth, scale=100, splitting_ratio=-1):
        self.variable_nodes = dict()
        self.function_nodes = []
        all_vars, all_matrix = parse(pth, scale=scale)
        self._construct_nodes(all_vars, all_matrix, splitting_ratio)

    def _construct_nodes(self, all_vars, all_matrix, splitting_ratio):
        for v, dom in all_vars:
            self.variable_nodes[v] = AttentiveVariableNode(v, dom)
        for matrix, row, col in all_matrix:
            if 0 < splitting_ratio < 1:
                matrix1 = _matrix_multiply(matrix, splitting_ratio)
                matrix2 = _matrix_multiply(matrix, 1 - splitting_ratio)
                self.function_nodes.append(AttentiveFunctionNode(f'({row},{col})1', matrix1, self.variable_nodes[row],
                                                                 self.variable_nodes[col]))
                self.function_nodes.append(AttentiveFunctionNode(f'({row},{col})2', matrix2, self.variable_nodes[row],
                                                                 self.variable_nodes[col]))
            else:
                self.function_nodes.append(AttentiveFunctionNode(f'({row},{col})', matrix, self.variable_nodes[row],
                                                                 self.variable_nodes[col]))
        all_degree = sum([len(x.neighbors) for x in self.variable_nodes.values()])
        assert int(all_degree / 2) == len(self.function_nodes)

    def step(self, model, fe, training=False, first_it=True):
        for func in self.function_nodes:
            func.compute_msgs()
        v2f_msgs, f2v_msgs = fe.update_message()
        for variable in self.variable_nodes.values():
            variable.compute_distribution()
            variable.make_decision()
        loss = 0
        if training and not first_it:
            for variable in self.variable_nodes.values():
                loss = loss + variable.compute_local_loss()
        if training:
            fe.attentive_weights, fe.hidden1, fe.hidden2 = model(fe.x,
                                                                 fe.edge_index,
                                                                 v2f_msgs,
                                                                 fe.hidden1,
                                                                 f2v_msgs,
                                                                 fe.hidden2,
                                                                 fe.prefix,
                                                                 fe.function_indexes,
                                                                 fe.neighbor_idx_info)
        else:
            with torch.no_grad():
                fe.attentive_weights, fe.hidden1, fe.hidden2 = model(fe.x,
                                                                     fe.edge_index,
                                                                     v2f_msgs,
                                                                     fe.hidden1,
                                                                     f2v_msgs,
                                                                     fe.hidden2,
                                                                     fe.prefix,
                                                                     fe.function_indexes,
                                                                     fe.neighbor_idx_info)
        for variable in self.variable_nodes.values():
            variable.compute_msgs()
        cost = 0
        for func in self.function_nodes:
            cost += func.matrix[func.row_vn.val_idx][func.col_vn.val_idx]
        return cost, loss
