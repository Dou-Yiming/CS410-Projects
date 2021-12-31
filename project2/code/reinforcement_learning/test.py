from kaggle_environments import make
from functools import reduce

from torch._C import device
from torch.random import seed
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from model import *
from util import *
from obs import * 
from Buffer import *
from RL_agent import *
from baseline_agent import *
from train import run_one_episode

def test(agent, episodes):
    reward_list = []
    winner_list = []
    for i in range(1, episodes + 1):
        reward, dqn_loss, expert_loss, winner = run_one_episode(
            agent, training=False, epsilon=0)
        print("(TEST) Epsisode {}/{}: reward: {}, winner: {}".format(i, episodes, reward, winner))
        reward_list.extend(reward)
        winner_list.extend(winner)
    print("(TEST) test results: {}, win rate: {}".format(
        reduce(lambda x, y: x + y, reward_list) / len(reward_list),
        reduce(lambda x, y: x + y, list(map(lambda x: True if x == 'RL' else 0, winner_list))) / len(winner_list)
    ))

if __name__=='__main__':
    agent = ActorAgent(15, 8, 6)    
    if PRETRAIN:
        agent.restore_model('./model/dqn_model')
    test(agent, 10)