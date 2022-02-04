import random

import torch

ASSIGN_ID = [1, 0, 0]
MSG_ID = [0, 1, 0]
FUNCTION_ID = [0, 0, 1]

NODE_ID_EMBED_UB = 0


def msg_to_tensor(msg, device):
    if type(msg) is list:
        msg = torch.tensor(msg, dtype=torch.float32, device=device)
    else:
        msg = msg.detach()
    return msg.squeeze()


class FeatureExtractor:
    def __init__(self, variable_nodes, function_nodes, node_embed_dim=11, device='cpu'):
        vn_start_index = dict()
        function_nodes = list(function_nodes)
        function_indexes = []

        x = []
        for node in variable_nodes:
            node_id = [random.random() for _ in range(node_embed_dim - 3)]
            norm = sum(node_id)
            node_id = [i * NODE_ID_EMBED_UB / norm for i in node_id]
            vn_start_index[node.name] = len(x)
            for i in range(node.dom_size):
                x.append(ASSIGN_ID + node_id)
        padding = [0 for _ in range(node_embed_dim - 3)]
        for node in function_nodes:
            for _ in range(2):
                function_indexes.append(len(x))
                x.append(FUNCTION_ID + padding)

        edge_index = [[], []]
        src, dest = edge_index
        node_running_idx = len(x)

        hidden_cnt = 0
        # v->m->f
        for idx, node in enumerate(function_nodes):
            i = node.row_vn
            j = node.col_vn
            idx = idx * 2
            f_idx = function_indexes[idx]
            for val in range(i.dom_size):
                # v-> m
                src.append(vn_start_index[i.name] + val)
                dest.append(node_running_idx)

                # m -> f
                src.append(node_running_idx)
                dest.append(f_idx)
                node_running_idx += 1
                hidden_cnt += 1

            idx += 1
            f_idx = function_indexes[idx]
            for val in range(j.dom_size):
                # v-> m
                src.append(vn_start_index[j.name] + val)
                dest.append(node_running_idx)
                # m -> f
                src.append(node_running_idx)
                dest.append(f_idx)
                node_running_idx += 1
                hidden_cnt += 1

        # f->m->v
        for idx, node in enumerate(function_nodes):
            i = node.row_vn
            j = node.col_vn
            idx = idx * 2 + 1
            f_idx = function_indexes[idx]
            for val in range(i.dom_size):
                # f -> m
                src.append(f_idx)
                dest.append(node_running_idx)

                # m -> v
                src.append(node_running_idx)
                dest.append(vn_start_index[i.name] + val)
                node_running_idx += 1
            idx -= 1
            f_idx = function_indexes[idx]
            for val in range(j.dom_size):
                # f -> m
                src.append(f_idx)
                dest.append(node_running_idx)

                # m -> v
                src.append(node_running_idx)
                dest.append(vn_start_index[j.name] + val)
                node_running_idx += 1

        self.neighbor_idx_mapping = dict()
        self.neighbor_idx_info = dict()
        for vn in variable_nodes:
            in_idxes = []
            out_indexes = []
            self.neighbor_idx_mapping[vn.name] = dict()
            self.neighbor_idx_info[vn.name] = dict()
            for fn in vn.neighbors.values():
                idx = function_nodes.index(fn)
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

        self.hidden1 = torch.zeros(hidden_cnt, node_embed_dim - 3, dtype=torch.float32, device=device)
        self.hidden2 = torch.zeros(hidden_cnt, node_embed_dim - 3, dtype=torch.float32, device=device)
        self.prefix = torch.tensor(MSG_ID, dtype=torch.float32, device=device)
        self.prefix = self.prefix.repeat(2 * hidden_cnt, 1)
        self.function_indexes = function_indexes
        self.function_nodes = function_nodes
        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        self.device = device
        self.attentive_weights = dict()  # returned by neural network

        all_nodes = list(variable_nodes) + self.function_nodes
        for node in all_nodes:
            node.register_feature_extractor(self)

    def update_message(self):
        v2f = []
        f2v = []
        for fn in self.function_nodes:
            i = fn.row_vn
            j = fn.col_vn
            # x_i -> f_ij
            msg = msg_to_tensor(fn.incoming_msg[i.name], self.device)
            v2f.append(msg)
            # x_j -> f_ij
            msg = msg_to_tensor(fn.incoming_msg[j.name], self.device)
            v2f.append(msg)

            # f_ij -> x_i
            msg = msg_to_tensor(i.incoming_msg[fn.name], self.device)
            f2v.append(msg)
            # f_ij -> x_j
            msg = msg_to_tensor(j.incoming_msg[fn.name], self.device)
            f2v.append(msg)
        return torch.cat(v2f).unsqueeze(1), torch.cat(f2v).unsqueeze(1)
