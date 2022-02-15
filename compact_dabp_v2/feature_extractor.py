import torch

MAX_COLOR_NUM = 100
VAR_ID = [1, 0, 0, 0]
V2F_ID = [0, 1, 0, 0]
F2V_ID = [0, 0, 1, 0]
FUN_ID = [0, 0, 0, 1]


def label_variables(variable_nodes: dict):
    ordered_keys = sorted([x for x in variable_nodes.keys()])
    labels = dict()
    for key in ordered_keys:
        vn = variable_nodes[key]
        neighbor_vn_name = set()
        for fn in vn.neighbors.values():
            oppo = fn.row_vn.name if fn.row_vn != vn else fn.col_vn.name
            neighbor_vn_name.add(oppo)
        assert len(neighbor_vn_name) == len(vn.neighbors)
        all_colors = set()
        for nb in neighbor_vn_name:
            if nb in labels:
                all_colors.add(labels[nb])
        for color in range(MAX_COLOR_NUM):
            if color not in all_colors:
                break
        labels[key] = color
    return labels


def check_gc(variable_nodes: dict, labels: dict):
    for vn in variable_nodes.values():
        my_color = labels[vn.name]
        for fn in vn.neighbors.values():
            oppo = fn.row_vn if fn.row_vn != vn else fn.col_vn
            your_color = labels[oppo.name]
            assert my_color != your_color


def msg_to_tensor(msg, device):
    if type(msg) is list:
        msg = torch.tensor(msg, dtype=torch.float32, device=device)
    else:
        msg = msg.detach()
    return msg.squeeze()


class FeatureExtractor:
    def __init__(self, variable_nodes: dict, function_nodes, hidden_dim, device='cpu'):
        function_nodes = list(function_nodes)

        vn_colors = label_variables(variable_nodes)
        # check_gc(variable_nodes, vn_colors)

        self.vn_id_embed = []
        embedding_index = dict()
        for idx, vn_name in enumerate(variable_nodes.keys()):
            self.vn_id_embed.append(vn_colors[vn_name])
            embedding_index[vn_name] = idx

        self.fn_embed = []
        padding = [0 for _ in range(hidden_dim)]
        for fn in function_nodes:
            embedding_index[fn.name] = len(self.fn_embed) + len(variable_nodes)
            self.fn_embed.append(FUN_ID + padding)
        running_node_index = len(variable_nodes) + len(self.fn_embed)
        self.fn_embed = torch.tensor(self.fn_embed, dtype=torch.float32, device=device)

        self.edge_index = [[], []]
        src, dst = self.edge_index
        for fn in function_nodes:
            i = fn.row_vn.name
            j = fn.col_vn.name
            # i -> m -> f
            src.append(embedding_index[i])
            dst.append(running_node_index)
            src.append(running_node_index)
            dst.append(embedding_index[fn.name])
            running_node_index += 1

            # j -> m -> f
            src.append(embedding_index[j])
            dst.append(running_node_index)
            src.append(running_node_index)
            dst.append(embedding_index[fn.name])
            running_node_index += 1

        for fn in function_nodes:
            i = fn.row_vn.name
            j = fn.col_vn.name
            # f -> m -> i
            src.append(embedding_index[fn.name])
            dst.append(running_node_index)
            src.append(running_node_index)
            dst.append(embedding_index[i])
            running_node_index += 1

            # f -> m -> j
            src.append(embedding_index[fn.name])
            dst.append(running_node_index)
            src.append(running_node_index)
            dst.append(embedding_index[j])
            running_node_index += 1

        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long, device=device)

        self.function_nodes = function_nodes
        self.vn_id_embed = torch.tensor(self.vn_id_embed, dtype=torch.long, device=device)  # [num_vn, ]
        self.vn_prefix = torch.tensor(VAR_ID, dtype=torch.float32, device=device)
        self.vn_prefix = self.vn_prefix.repeat(len(variable_nodes), 1)  # [num_vn, 4]

        self.var_to_fun_hidden = torch.zeros(2 * len(function_nodes), hidden_dim, dtype=torch.float32, device=device)
        self.fun_to_var_hidden = torch.zeros(2 * len(function_nodes), hidden_dim, dtype=torch.float32, device=device)
        v2f_prefix = torch.tensor(V2F_ID, dtype=torch.float32, device=device)
        v2f_prefix = v2f_prefix.repeat(2 * len(function_nodes), 1)  # [2 * num_fn, 4]
        f2v_prefix = torch.tensor(F2V_ID, dtype=torch.float32, device=device)
        f2v_prefix = f2v_prefix.repeat(2 * len(function_nodes), 1)  # [2 * num_fn, 4]
        self.msg_prefix = torch.cat([v2f_prefix, f2v_prefix], dim=0)
        self.device = device

        all_nodes = list(variable_nodes.values()) + self.function_nodes
        for node in all_nodes:
            node.register_feature_extractor(self)

        self.neighbor_idx_mapping = dict()
        self.neighbor_idx_info = dict()
        for vn in variable_nodes.values():
            in_idxes = []
            out_indexes = []
            self.neighbor_idx_mapping[vn.name] = dict()
            self.neighbor_idx_info[vn.name] = dict()
            for fn in vn.ordered_neighbors:
                fn = vn.neighbors[fn]
                idx = function_nodes.index(fn)
                in_idxes.append(idx)
                out_indexes.append(idx)
                self.neighbor_idx_mapping[vn.name][fn.name] = out_indexes[-1]
            for i in range(len(out_indexes)):
                idxes = list(in_idxes)
                idxes[i] = out_indexes[i]
                self.neighbor_idx_info[vn.name][out_indexes[i]] = idxes

    def update_message(self):
        v2f = []
        f2v = []
        for fn in self.function_nodes:
            i = fn.row_vn
            j = fn.col_vn
            v2f.append(msg_to_tensor(fn.incoming_msg[i.name], self.device))
            f2v.append(msg_to_tensor(i.incoming_msg[fn.name], self.device))
            v2f.append(msg_to_tensor(fn.incoming_msg[j.name], self.device))
            f2v.append(msg_to_tensor(j.incoming_msg[fn.name], self.device))
        return torch.stack(v2f), torch.stack(f2v)
