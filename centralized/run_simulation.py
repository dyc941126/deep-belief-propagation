from entities import FactorGraph, VariableNode, FunctionNode


if __name__ == '__main__':
    pth = '../problem_instance/randomDCOPs/100/0.05/10/0.xml'
    best_cost = 99999999
    VariableNode.damp_factor = .9
    fg = FactorGraph(pth, FunctionNode, VariableNode)
    for it in range(1000):
        cost = fg.step()
        best_cost = min(cost, best_cost)
        print(f'Iteration {it}\t Cost: {cost}, Best cost: {best_cost}')