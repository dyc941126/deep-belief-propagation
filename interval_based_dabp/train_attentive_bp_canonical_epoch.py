import os
import random

import torch

from model import AttentiveBP

from attentive_bp_entities import FeatureConstructor, AttentiveFactorGraph
from centralized.utilities import elementwise_add


if __name__ == '__main__':
    train_problems_pth = '../problem_instance/randomDCOPs/train'
    valid_problems_pth = '../problem_instance/randomDCOPs/test'
    model_save_pth = '../models'
    if not os.path.exists(model_save_pth):
        os.makedirs(model_save_pth)
    nb_epoch = 10000
    nb_iteration = 100
    nb_timestep = 1000
    update_interval = 20
    valid_interval = nb_iteration
    m = AttentiveBP(in_channels=8, out_channels=16, ass_to_sum_hid_dim=8, sum_to_ass_hid_dim=8, edge_dim=8, num_gru_layers=2, num_heads=4)
    device = 'cuda:3'
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
            fg = AttentiveFactorGraph(pth, scale=scale)
            fe = FeatureConstructor(fg.variable_nodes.values(), fg.function_nodes, ass_to_sum_hidden=8, sum_to_ass_hidden=8, num_gru_layers=2, device=device)
            if not printed:
                print(f'{len(fg.variable_nodes)} variables, {len(fg.function_nodes)} factors')
                printed = True
            losses = []
            costs = []
            best_cost = 100000
            running_loss = 0
            running_cost = 0
            cic = []
            for ts in range(nb_timestep):
                ts += 1
                c, l = fg.step(m, fe, True, ts % update_interval == 0)
                if ts > update_interval:
                    running_loss += l
                    running_cost += c
                    if ts % update_interval == 0:
                        losses.append(running_loss / update_interval)
                        costs.append(running_cost * scale / update_interval)
                        running_loss = running_cost = 0
                c = c * scale
                best_cost = min(best_cost, c)
                cic.append(c)
            indexes = sorted(range(len(costs)), key=costs.__getitem__)
            k = 10
            topk_costs = []
            loss = 0
            for i in range(k):
                topk_costs.append(costs[indexes[i]])
                loss = loss + losses[indexes[i]]
            loss = loss / k
            print(it, '{:.4f}'.format(loss.item()), int(sum(cic) / len(cic)), int(best_cost), int(sum(topk_costs) / k))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if total_training_steps != 0 and total_training_steps % valid_interval == 0:
                m.eval()
                cost_in_timestep = [0] * nb_timestep
                for valid_pth in valid_list:
                    fg = AttentiveFactorGraph(valid_pth, scale=scale)
                    fe = FeatureConstructor(fg.variable_nodes.values(), fg.function_nodes, ass_to_sum_hidden=8, sum_to_ass_hidden=8, num_gru_layers=2,
                                            device=device)
                    costs = []
                    for ts in range(nb_timestep):
                        c, _ = fg.step(m, fe, False, ts != 0 and ts % update_interval == 0)
                        costs.append(c * scale)
                    cost_in_timestep = elementwise_add(costs, cost_in_timestep)
                cost_in_timestep = [f'{x / len(valid_list): .2f}' for x in cost_in_timestep]
                tag = int(total_training_steps / valid_interval)
                print(f"Validation {tag}: {' '.join(cost_in_timestep)}")
                torch.save(m.state_dict(), f'{model_save_pth}/{total_training_steps}.pth')