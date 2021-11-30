import torch

from entities import VariableNode, FunctionNode

# variable-assignment node, summarize node


class AttentiveVariableNode(VariableNode):
    def __init__(self, name, dom_size):
        super().__init__(name, dom_size)
        self.assignment_node_idxes = []
        self.cost_matrices = []
        self.enforced_vn = []
        self.belief_dist = torch.zeros(dom_size)
        self.all_fn_names = []
        self.weight = None
        self.name_embed = None
        self.attentive_node_idxes = []

    def post_initialization(self, assignment_node_idxes, name_embed):
        self.assignment_node_idxes = assignment_node_idxes
        assert len(assignment_node_idxes) == self.dom_size

        self.name_embed = torch.tensor([1, 0] + name_embed).repeat(self.dom_size, 1)

        self.all_fn_names = []
        for func in self.neighbors.values():
            self.all_fn_names.append(func.name)
            if func.row_vn == self:
                self.cost_matrices.append(func.row_matrix)
                self.enforced_vn.append(func.col_vn)
                in_idxes = func.c2r_summarize_node_idxes
                out_idxes = func.r2c_summarize_node_idxes
            else:
                in_idxes = func.r2c_summarize_node_idxes
                out_idxes = func.c2r_summarize_node_idxes
            self.attentive_node_idxes.append([in_idxes, out_idxes])
        self.cost_matrices = torch.stack(self.cost_matrices)

    def compute_belief_dist(self):
        all_income_msgs = torch.stack([x for x in self.incoming_msg.values()])
        self.belief_dist = torch.sum(all_income_msgs, dim=0)
        self.belief_dist = torch.softmax(-self.belief_dist, dim=0)

    def make_decision(self):
        self.val_idx = torch.argmax(self.belief_dist).item()

    def compute_expected_cost(self):
        belief_dist = self.belief_dist.unsqueeze(0)
        all_enforced_vn_bd = torch.stack([x.belief_dist.unsqueeze(-1) for x in self.enforced_vn], dim=0)
        cost = torch.matmul(belief_dist, self.cost_matrices)
        return torch.bmm(cost, all_enforced_vn_bd)

    def get_weight(self, weight):
        self.weight = weight

    def compute_msgs(self):
        all_income_msg = [self.incoming_msg[x] for x in self.all_fn_names]
        for idx, name in enumerate(self.all_fn_names):
            tensors = list(all_income_msg)
            tensors[idx] = self.prev_sent[name]
            tensors = torch.stack(tensors, dim=1)
            weight = self.weight[idx]
            weight = weight * len(self.neighbors)
            msg = torch.matmul(tensors, weight)
            msg = msg.mean(-1)
            norm = msg.detach().min()
            msg = msg - norm
            self.neighbors[name].incoming_msg[self.name] = msg
            self.neighbors[name].norms[self.name] = norm

    def fill_feat(self, x):
        x[self.assignment_node_idxes] = self.name_embed


class AttentiveFunctionNode(FunctionNode):
    def __init__(self, name, matirx, row_vn, col_vn, to_s_hidden_size, s_to_hidden_size, device='cpu'):
        super().__init__(name, matirx, row_vn, col_vn)
        self.row_matrix = torch.tensor(self.matrix)
        self.col_matrix = self.row_matrix.t()
        self.norms = {row_vn.name: None, col_vn.name: None}
        self.r2c_summarize_node_idxes = []
        self.c2r_summarize_node_idxes = []
        self.r2s_hidden = torch.zeros(self.row_vn.dom_size, self.col_vn.dom_size, to_s_hidden_size, device=device)
        self.s2c_hidden = torch.zeros(self.col_vn.dom_size, s_to_hidden_size, device=device)
        self.c2s_hidden = torch.zeros(self.col_vn.dom_size, self.row_vn.dom_size, to_s_hidden_size, device=device)
        self.s2r_hidden = torch.zeros(self.row_vn.dom_size, s_to_hidden_size, device=device)
        self.r2c_type = torch.tensor([0, 1], dtype=torch.float32, device=device).repeat(
            len(self.r2c_summarize_node_idxes), 1)
        self.c2r_type = torch.tensor([0, 1], dtype=torch.float32, device=device).repeat(
            len(self.c2r_summarize_node_idxes), 1)

    def post_initialization(self, r2c_summarize_node_idxes, c2r_summarize_node_idxes):
        assert len(r2c_summarize_node_idxes) == self.col_vn.dom_size
        assert len(c2r_summarize_node_idxes) == self.row_vn.dom_size

        self.r2c_summarize_node_idxes = r2c_summarize_node_idxes
        self.c2r_summarize_node_idxes = c2r_summarize_node_idxes

    def compute_msgs(self):
        msg = AttentiveFunctionNode._marginalize(self.row_matrix, self.incoming_msg[self.col_vn.name])
        self.row_vn.income_msg[self.name] = msg
        msg = AttentiveFunctionNode._marginalize(self.col_matrix, self.incoming_msg[self.row_vn.name])
        self.col_vn.income_msg[self.name] = msg

    @classmethod
    def _marginalize(cls, matrix, income_msg):
        income_msg = income_msg.unsqueeze(0)
        matrix = matrix + income_msg
        if AttentiveFunctionNode.op == min:
            msg, _ = matrix.min(-1)
        else:
            msg, _ = matrix.max(-1)
        return msg

    def fill_feat(self, x, edge_index, to_s_embed, to_s_hidden, to_s_idxes, s_to_embed, s_to_hidden, s_to_idxes):
        x[self.r2c_summarize_node_idxes] = torch.cat([self.r2c_type, self.col_vn.name_embed[:, 2:]], dim=1)
        x[self.c2r_summarize_node_idxes] = torch.cat([self.c2r_type, self.row_vn.name_embed[:, 2:]], dim=1)
        src, dest = edge_index
        for idx, node_idx in enumerate(self.r2c_summarize_node_idxes):
            # r to s
            for row_val in range(self.row_vn.dom_size):
                local_cost = self.row_matrix[row_val, idx].item()
                msg = self.incoming_msg[self.row_vn.name][row_val].item()
                norm = self.norms[self.row_vn.name]
                to_s_embed.append([local_cost, msg, norm])
                to_s_idxes.append(len(src))
                src.append(self.row_vn.assignment_node_idxes[row_val])
                dest.append(node_idx)
                to_s_hidden.append(self.r2s_hidden[row_val, idx, :])
            # s to c
            s_to_idxes.append(len(src))
            s_to_embed.append(self.col_vn.income_msg[self.name][idx].item())
            s_to_hidden.append(self.s2c_hidden[idx, :])
            src.append(node_idx)
            dest.append(self.col_vn.assignment_node_idxes[idx])

        for idx, node_idx in enumerate(self.c2r_summarize_node_idxes):
            # c to s
            for col_val in range(self.col_vn.dom_size):
                local_cost = self.col_matrix[col_val, idx].item()
                msg = self.incoming_msg[self.col_vn.name][col_val].name
                norm = self.norms[self.col_vn.name]
                to_s_embed.append([local_cost, msg, norm])
                to_s_hidden.append(self.c2s_hidden[col_val, idx, :])
                to_s_idxes.append(len(src))
                src.append(self.col_vn.assignment_node_idxes[col_val])
                dest.append(node_idx)
            # s to c
            s_to_idxes.append(len(src))
            s_to_embed.append(self.row_vn.income_msg[self.name][idx].item())
            s_to_hidden.append(self.s2r_hidden[idx, :])
            src.append(node_idx)
            dest.append(self.row_vn.assignment_node_idxes[idx])
