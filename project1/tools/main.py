import sys
import json
from easydict import EasyDict as edict
from pprint import pprint
sys.path.append("..")
from algorithms.GA import GA
from algorithms.DP import DP
from algorithms.Astar import Astar
from tools.dataset import Data_loader


def get_config(cfg_path):
    return edict(json.load(open(cfg_path, 'r')))


def cost(S1, S2):
    """
    Computes cost of 2 seq
    """
    assert len(S1) == len(S2)
    cost = 0
    for i in range(len(S1)):
        if S1[i] == S2[i]:
            cost = cost
        elif S1[i] == '-' or S2[i] == '-':
            cost += 2
        else:
            cost += 3
    return cost


def run_genetic_algorithm(cfg, data_loader):
    print("Testing Genetic Algorithm...")
    ga = GA(cfg=cfg, loader=data_loader, cost_func=cost)
    ga.optimize()


def run_dynamic_programming(cfg, data_loader):
    print("Testing Dynamic Programming Algorithm...")
    dp = DP(cfg, data_loader)
    dp.search()


def run_Astar(cfg, data_loader):
    print("Testing Astar Algorithm...")
    astar = Astar(cfg, data_loader)
    astar.search()


if __name__ == '__main__':
    cfg = get_config('../config/default.json')
    data_loader = Data_loader(cfg)

    run_dynamic_programming(cfg, data_loader)

    run_Astar(cfg, data_loader)

    run_genetic_algorithm(cfg, data_loader)
