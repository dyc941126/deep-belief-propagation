from centralized.utilities import elementwise_add
from entities import VariableNode, FunctionNode, FactorGraph


class ADVPVariableNode(VariableNode):
    def __init__(self, name, dom_size):
        super().__init__(name, dom_size)
        self.prev_neighbors = set()
        self.succ_neighbors = set()

    def compute_msgs(self):
        if len(self.prev_neighbors) + len(self.succ_neighbors) == 0:
            for func in self.neighbors.values():
                oppo = func.row_vn if func.col_vn == self else func.col_vn
                if oppo.name > self.name:
                    self.succ_neighbors.add(func.name)
                else:
                    self.prev_neighbors.add(func.name)
        for nei in self.succ_neighbors:
            assert self.neighbors[nei].prev == self.name
            msg = [0] * self.dom_size
            for other_nei in self.neighbors:
                if other_nei == nei:
                    continue
                if other_nei not in self.incoming_msg:
                    continue
                msg = elementwise_add(msg, self.incoming_msg[other_nei])
            norm = min(msg)
            msg = [x - norm for x in msg]
            # damping & normalizing
            if nei in self.prev_sent and 0 < VariableNode.damp_factor < 1:
                prev = self.prev_sent[nei]
                msg = [(1 - VariableNode.damp_factor) * x + VariableNode.damp_factor * y for x, y in zip(msg, prev)]
                norm = min(msg)
                msg = [x - norm for x in msg]
            self.prev_sent[nei] = list(msg)
            # send the message to nei
            self.neighbors[nei].incoming_msg[self.name] = msg

    def alternate(self):
        tmp = self.prev_neighbors
        self.prev_neighbors = self.succ_neighbors
        self.succ_neighbors = tmp


class ADVPFunctionNode(FunctionNode):
    def __init__(self, name, matirx, row_vn, col_vn):
        super().__init__(name, matirx, row_vn, col_vn)
        self.val_prop = False
        self.prev = min(row_vn.name, col_vn.name)
        self.succ = max(row_vn.name, col_vn.name)
        self.neighbors = {row_vn.name: row_vn, col_vn.name: col_vn}

    def alternate(self):
        tmp = self.prev
        self.prev = self.succ
        self.succ = tmp

    def compute_msgs(self):
        nei = self.neighbors[self.succ]
        msg = [0] * nei.dom_size
        if nei == self.row_vn:
            belief = [0] * self.col_vn.dom_size if self.col_vn.name not in self.incoming_msg else self.incoming_msg[
                self.col_vn.name]
            for val in range(self.row_vn.dom_size):
                utils = [x + y for x, y in zip(belief, self.matrix[val])]
                if not self.val_prop:
                    msg[val] = FunctionNode.op(utils)
                else:
                    msg[val] = utils[self.col_vn.val_idx]
        else:
            belief = [0] * self.row_vn.dom_size if self.row_vn.name not in self.incoming_msg else self.incoming_msg[
                self.row_vn.name]
            for val in range(self.col_vn.dom_size):
                local_vec = [self.matrix[i][val] for i in range(self.row_vn.dom_size)]
                utils = [x + y for x, y in zip(belief, local_vec)]
                if not self.val_prop:
                    msg[val] = FunctionNode.op(utils)
                else:
                    msg[val] = utils[self.row_vn.val_idx]
        nei.incoming_msg[self.name] = msg


class ADVPFactorGraph(FactorGraph):

    def __init__(self, pth, function_node_type, variable_node_type, phase=2):
        super().__init__(pth, function_node_type, variable_node_type)
        self.iteration_num = 0
        self.phase = 2

    def step(self):
        self.iteration_num += 1
        if self.iteration_num % len(self.variable_nodes) == 0:
            for func in self.function_nodes:
                func.alternate()
            for variable in self.variable_nodes.values():
                variable.alternate()
            if int(self.iteration_num / len(self.variable_nodes)) == self.phase:
                for func in self.function_nodes:
                     func.val_prop = True
        for func in self.function_nodes:
            func.compute_msgs()
        for variable in self.variable_nodes.values():
            variable.compute_msgs()
            variable.make_decision()
        cost = 0
        for func in self.function_nodes:
            cost += func.matrix[func.row_vn.val_idx][func.col_vn.val_idx]
        return cost