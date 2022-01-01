import os.path
import random
import shutil
from problem import Problem
import networkx as nx

if __name__ == '__main__':
    nb_instances = 20
    pth = '../problem_instance/wgc'
    for ag in range(60, 101, 10):
        for i in range(nb_instances):
            p = Problem()
            p.random_binary(ag, 3, 0.1, gc=True, weighted=True, decimal=5)
            p.save(f'{pth}/{ag}/0.1/{i}.xml')