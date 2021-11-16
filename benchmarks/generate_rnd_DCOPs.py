import os.path
import shutil
from problem import Problem


if __name__ == '__main__':
    nb_instances = 20
    nb_agents = 100
    density = 0.05
    domain_size = 10
    pth = f'../problem_instance/randomDCOPs/{nb_agents}/{density}/{domain_size}'
    if os.path.exists(pth):
        shutil.rmtree(pth, ignore_errors=True)
    for i in range(nb_instances):
        p = Problem()
        p.random_binary(nb_agents, domain_size, density)
        p.save(f'{pth}/{i}.xml')