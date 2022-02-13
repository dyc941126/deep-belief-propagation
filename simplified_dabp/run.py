import os

import torch

from simplified_dabp.dabp_entities import AttentiveFactorGraph
from simplified_dabp.feature_extractor import FeatureExtractor
from simplified_dabp.model import AttentiveBP


def solve(m, pth, scale=10, cycle=1000, fine_tune=0, fine_tune_cycle=100, optimizer=None, in_channels=11, device='cpu'):
    cic = []
    bcic = []
    if fine_tune > 0:
        assert optimizer
        best_trial_cost = 1000000
        for trial in range(fine_tune):
            fg = AttentiveFactorGraph(pth, scale=scale, splitting_ratio=-1)
            fe = FeatureExtractor(fg.variable_nodes.values(), fg.function_nodes, node_embed_dim=in_channels,
                                  device=device)
            losses = []
            costs = []
            for ts in range(fine_tune_cycle):
                c, l = fg.step(m, fe, True, ts == 0)
                if ts != 0:
                    c = c * scale
                    losses.append(l)
                    costs.append(c)
            indexes = sorted(range(len(costs)), key=costs.__getitem__)
            k = 2
            topk_costs = []
            loss = 0
            for i in range(k):
                topk_costs.append(costs[indexes[i]])
                loss = loss + losses[indexes[i]]
            loss = loss / k
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            fe.hidden1 = fe.hidden1.detach().clone()
            fe.hidden2 = fe.hidden2.detach().clone()
            for vn in fg.variable_nodes.values():
                vn.reset()
            for fn in fg.function_nodes:
                fn.reset()

            for ts in range(fine_tune_cycle, cycle):
                c, _ = fg.step(m, fe, False, ts == 0)
                c = int(c * scale)
                costs.append(c)
            if min(costs) < best_trial_cost:
                best_trial_cost = min(costs)
                cic = costs
            print(f'Fine tune {trial}: {loss.item():.4f} {min(costs)}, {min(cic)}')
        for c in cic:
            if len(bcic) == 0:
                bcic.append(c)
            else:
                bcic.append(min(c, bcic[-1]))
        return cic, bcic

    fg = AttentiveFactorGraph(pth, scale=scale, splitting_ratio=-1)
    fe = FeatureExtractor(fg.variable_nodes.values(), fg.function_nodes, node_embed_dim=in_channels,
                          device=device)
    for ts in range(cycle):
        c, _ = fg.step(m, fe, False, ts == 0)
        c = int(c * scale)
        cic.append(c)
        if len(bcic) == 0:
            bcic.append(c)
        else:
            bcic.append(min(c, bcic[-1]))
    return cic, bcic


if __name__ == '__main__':
    model_pth = '../models/1800.pth'
    device = 'cuda:7'
    m = AttentiveBP(11, 16, 4)
    m.to(device)
    m.load_state_dict(torch.load(model_pth, map_location=device))
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.0005, weight_decay=5e-5)

    problem_path = '../problem_instance/wgc/60/0.1'
    for file in os.listdir(problem_path):
        if not file.endswith('.xml'):
            continue
        cost_in_cycle, best_cost_in_cycle = solve(m, f'{problem_path}/{file}', device=device, fine_tune=100, cycle=100, optimizer=optimizer)
        print(file, min(best_cost_in_cycle))
        with open(f'{problem_path}/simplified_dabp_1800_finetune_5_cic.txt', 'a') as f:
            f.write(str(cost_in_cycle) + '\n')
        with open(f'{problem_path}/simplified_dabp_1800_finetune_5_bcic.txt', 'a') as f:
            f.write(str(best_cost_in_cycle) + '\n')