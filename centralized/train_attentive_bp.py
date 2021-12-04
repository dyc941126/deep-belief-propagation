import os
import random

from attentive_bp_entities import FeatureConstructor, AttentiveFactorGraph


if __name__ == '__main__':
    train_problems_pth = ''
    valid_problems_pth = ''
    test_problems_pth = ''
    nb_epoch = 10000
    nb_iteration = 10
    nb_timestep = 20
    train_list, valid_list, test_list = [], [], []
    for f in os.listdir(train_problems_pth):
        if f.endswith('.xml'):
            train_list.append(os.path.join(train_problems_pth, f))
    for f in os.listdir(valid_problems_pth):
        if f.endswith('.xml'):
            valid_list.append(os.path.join(valid_problems_pth, f))
    for f in os.listdir(test_problems_pth):
        if f.endswith('.xml'):
            test_list.append(os.path.join(test_problems_pth, f))
    for ep in range(nb_epoch):
        pth = random.choice(train_list)
        for it in range(nb_iteration):
            fg = AttentiveFactorGraph(pth)
            fe = FeatureConstructor(fg.variable_nodes.values(), fg.function_nodes, ass_to_sum_hidden=7)