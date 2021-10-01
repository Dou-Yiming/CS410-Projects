import sys
import json
from easydict import EasyDict as edict
from tqdm import tqdm
sys.path.append("..")

from tools.dataset import Data_loader
from algorithms.GA import GA

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


if __name__ == '__main__':
    cfg = get_config('../config/default.json')
    data_loader = Data_loader(cfg)
    ga = GA(cfg = cfg, loader=data_loader,cost_func = cost)
    ga.eval_ppl()
    ga.optimize()