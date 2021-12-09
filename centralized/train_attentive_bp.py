import os
import random

import torch

from model import AttentiveBP

from attentive_bp_entities import FeatureConstructor, AttentiveFactorGraph
from utilities import elementwise_add


if __name__ == '__main__':
    train_problems_pth = '../problem_instance/randomDCOPs/train'
    valid_problems_pth = '../problem_instance/randomDCOPs/valid'
    model_save_pth = '../models'
    if not os.path.exists(model_save_pth):
        os.makedirs(model_save_pth)
    nb_epoch = 10000
    nb_iteration = 100
    nb_timestep = 50
    valid_interval = nb_iteration
    m = AttentiveBP(8, 16, 1, 7, 1, 8, 4)
    device = 'cuda:0'
    m.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-5)
    train_list, valid_list, test_list = [], [], []
    for f in os.listdir(train_problems_pth):
        if f.endswith('.xml'):
            train_list.append(os.path.join(train_problems_pth, f))
    for f in os.listdir(valid_problems_pth):
        if f.endswith('.xml'):
            valid_list.append(os.path.join(valid_problems_pth, f))
    total_training_steps = 0
    for ep in range(nb_epoch):
        pth = random.choice(train_list)
        for it in range(nb_iteration):
            total_training_steps += 1
            m.train()
            fg = AttentiveFactorGraph(pth, scale=1)
            fe = FeatureConstructor(fg.variable_nodes.values(), fg.function_nodes, ass_to_sum_hidden=7, device=device)
            loss = 0
            cost = 0
            best_cost = 100000
            for ts in range(nb_timestep):
                c, l = fg.step(m, fe, True, ts == 0)
                loss = loss + l
                cost += c
                best_cost = min(best_cost, c)
            loss = loss / (nb_timestep - 1)
            print(loss.item(), int(cost / nb_timestep), best_cost)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if total_training_steps != 0 and total_training_steps % valid_interval == 0:
                m.eval()
                cost_in_timestep = [0] * nb_timestep
                for valid_pth in valid_list:
                    fg = AttentiveFactorGraph(valid_pth, scale=1)
                    fe = FeatureConstructor(fg.variable_nodes.values(), fg.function_nodes, ass_to_sum_hidden=7,
                                            device=device)
                    costs = []
                    for ts in range(nb_timestep):
                        c, _ = fg.step(m, fe, False, it == 0)
                        costs.append(c)
                    cost_in_timestep = elementwise_add(costs, cost_in_timestep)
                cost_in_timestep = [f'{x / len(valid_list): .2f}' for x in cost_in_timestep]
                tag = int(total_training_steps / valid_interval)
                print(f"Validation {tag}: {' '.join(cost_in_timestep)}")
                torch.save(m.state_dict(), f'{model_save_pth}/{total_training_steps}.pth')