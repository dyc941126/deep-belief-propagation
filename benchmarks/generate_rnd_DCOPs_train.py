import os.path
import random
import shutil
from problem import Problem


if __name__ == '__main__':
    nb_instances = 20
    pth = '../problem_instance/randomDCOPs/train'
    for i in range(nb_instances):
        nb_agent = random.randint(30, 55)
        density = random.random() * 0.25
        density = max(.1, density)
        dom_size = random.randint(3, 10)
        max_density = 200 / ((nb_agent - 1) * nb_agent * 0.5)
        density = min(max_density, density)
        p = Problem()
        p.random_binary(nb_agent, dom_size, density)
        p.save(f'{pth}/{i}.xml')