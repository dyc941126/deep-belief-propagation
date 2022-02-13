import os
import random

import torch

from model import AttentiveBP

from feature_extractor import FeatureExtractor
from dabp_entities import AttentiveFactorGraph
from centralized.utilities import elementwise_add


if __name__ == '__main__':
    train_problems_pth = '../problem_instance/randomDCOPs/train'
    valid_problems_pth = '../problem_instance/randomDCOPs/valid'
    model_save_pth = '../models_1000'

    if not os.path.exists(model_save_pth):
        os.makedirs(model_save_pth)

    nb_epoch = 10000
    nb_iteration = 10
    nb_each_timestep = 100
    nb_bp_cnt = 10
    valid_interval = nb_iteration
    in_channels = 11

    m = AttentiveBP(in_channels, 16, 4)
    device = 'cuda:4'
    m.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.0005, weight_decay=5e-5)
    train_list, valid_list, test_list = [], [], []
    for f in os.listdir(train_problems_pth):
        if f.endswith('.xml'):
            train_list.append(os.path.join(train_problems_pth, f))
    for f in os.listdir(valid_problems_pth):
        if f.endswith('.xml'):
            valid_list.append(os.path.join(valid_problems_pth, f))

    total_training_steps = 0
    scale = 10
    for ep in range(nb_epoch):
        pth = random.choice(train_list)
        printed = False
        for it in range(nb_iteration):
            total_training_steps += 1
            m.train()
            fg = AttentiveFactorGraph(pth, scale=scale, splitting_ratio=-1)
            fe = FeatureExtractor(fg.variable_nodes.values(),  fg.function_nodes, node_embed_dim=in_channels, device=device)
            if not printed:
                print(f'{len(fg.variable_nodes)} variables, {len(fg.function_nodes)} factors')
                printed = True
            best_cost = 100000
            for bp_it in range(nb_bp_cnt):
                fe.hidden1 = fe.hidden1.detach().clone()
                fe.hidden2 = fe.hidden2.detach().clone()
                for vn in fg.variable_nodes.values():
                    vn.reset()
                for fn in fg.function_nodes:
                    fn.reset()
                losses = []
                costs = []
                for ts in range(nb_each_timestep):
                    c, l = fg.step(m, fe, True, ts == 0)
                    if ts != 0:
                        c = c * scale
                        losses.append(l)
                        costs.append(c)
                        best_cost = min(best_cost, c)
                indexes = sorted(range(len(costs)), key=costs.__getitem__)
                k = 2
                topk_costs = []
                loss = 0
                for i in range(k):
                    topk_costs.append(costs[indexes[i]])
                    loss = loss + losses[indexes[i]]
                loss = loss / k
                print(f'{it} / {bp_it}', '{:.4f}'.format(loss.item()), int(sum(costs) / len(costs)), int(best_cost), int(sum(topk_costs) / k))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_training_steps != 0 and total_training_steps % valid_interval == 0:
                m.eval()
                cost_in_timestep = [0] * (nb_each_timestep * nb_bp_cnt)
                best_cost_in_timestep = [0] * (nb_each_timestep * nb_bp_cnt)
                for valid_pth in valid_list:
                    fg = AttentiveFactorGraph(valid_pth, scale=scale, splitting_ratio=-1)
                    fe = FeatureExtractor(fg.variable_nodes.values(), fg.function_nodes, node_embed_dim=in_channels,
                                          device=device)
                    costs = []
                    best_costs = []
                    for ts in range(nb_each_timestep * nb_bp_cnt):
                        c, _ = fg.step(m, fe, False, it == 0)
                        costs.append(c * scale)
                        if len(best_costs) == 0:
                            best_costs.append(costs[-1])
                        else:
                            best_costs.append(min(best_costs[-1], costs[-1]))
                    cost_in_timestep = elementwise_add(costs, cost_in_timestep)
                    best_cost_in_timestep = elementwise_add(best_costs, best_cost_in_timestep)
                cost_in_timestep = [x / len(valid_list) for x in cost_in_timestep]
                best_cost_in_timestep = [x / len(valid_list) for x in best_cost_in_timestep]
                tag = int(total_training_steps / valid_interval)
                print(f"Validation {tag}: default min cost {int(min(cost_in_timestep))} / anytime min cost {int(min(best_cost_in_timestep))}")
                torch.save(m.state_dict(), f'{model_save_pth}/{total_training_steps}.pth')