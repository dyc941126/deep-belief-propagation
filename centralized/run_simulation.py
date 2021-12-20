import os

from entities import FactorGraph, VariableNode, FunctionNode


def run(problem_dir, cycle=1000, damped_factor=.9):
    cic = []
    bcic = []
    cnt = 0
    for f in os.listdir(problem_dir):
        if not f.endswith('.xml'):
            continue
        cnt += 1
        VariableNode.damp_factor = damped_factor
        fg = FactorGraph(os.path.join(problem_dir, f), FunctionNode, VariableNode)
        cost = []
        best_cost = []
        for it in range(cycle):
            cost.append(fg.step())
            if len(best_cost) == 0:
                best_cost.append(cost[-1])
            else:
                best_cost.append(min(best_cost[-1], cost[-1]))
        print(f, best_cost[-1])
        if len(cic) == 0:
            cic = cost
            bcic = best_cost
        else:
            cic = [x + y for x, y in zip(cost, cic)]
            bcic = [x + y for x, y in zip(best_cost, bcic)]
    return [x / cnt for x in cic], [x / cnt for x in bcic]


if __name__ == '__main__':
    pth = '../problem_instance/randomDCOPs/test'
    result_pth = '../maxsum_results'
    if not os.path.exists(result_pth):
        os.makedirs(result_pth)
    for d in os.listdir(pth):
        c, bc = run(os.path.join(pth, d, '0.1'))
        with open(f'{result_pth}/dms_{d}_0.1_cic.txt', 'a') as rf:
            rf.write(str(c) + '\n')
        with open(f'{result_pth}/dms_{d}_0.1_bcic.txt', 'a') as rf:
            rf.write(str(bc) + '\n')
        c, bc = run(os.path.join(pth, d, '0.1_wgc'))
        with open(f'{result_pth}/dms_{d}_0.1_wgc_cic.txt', 'a') as rf:
            rf.write(str(c) + '\n')
        with open(f'{result_pth}/dms_{d}_0.1_wgc_bcic.txt', 'a') as rf:
            rf.write(str(bc) + '\n')