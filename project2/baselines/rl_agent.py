from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math
import sys
import random

import numpy as np
from numpy.lib.shape_base import split

class Obs():
    def __init__(self, observation):
        self.observation = observation

        self.wood_map = np.zeros((32,32))
        self.coal_map = np.zeros((32,32))
        self.uran_map = np.zeros((32,32))

        self.worker_cooldown = np.zeros((2,32,32))
        self.worker_capacity = np.zeros((2, 32, 32))

        self.cart_cooldown = np.zeros((2, 32, 32))
        self.cart_capacity = np.zeros((2, 32, 32))

        self.city_tiles_cooldown = np.zeros((2, 32, 32))
        self.city_tiles_fuel = np.zeros((2, 32, 32))

        self.step = self.observation["step"]

        pad_size_width = (32 - self.observation["width"]) // 2
        pad_size_length = (32 - self.observation["height"]) // 2

        self.worker_pos_dict = {}
        self.ct_pos_dict = {}

        ups = observation["updates"]
        cities = {}
        for row in ups:
            splits = row.split(" ")
            if splits[0] == "r":
              if splits[1] == "wood":
                self.wood_map[(
                  int(splits[2]) + pad_size_width,
                  int(splits[3]) + pad_size_length)] = int(float(splits[4]))
              elif splits[1] == "uranium":
                self.uran_map[(
                  int(splits[2]) + pad_size_width, 
                  int(splits[3]) + pad_size_length)] = int(float(splits[4]))
              elif splits[1] == "coal":
                self.coal_map[(
                    int(splits[2]) + pad_size_width, 
                    int(splits[3]) + pad_size_length)] = int(float(splits[4]))
            elif splits[0] == "c":
              cities[splits[2]] = int(splits[3])
            elif splits[0] == "u":
              self.worker_capacity[(
                int(splits[2]),
                int(splits[4]) + pad_size_width,
                int(splits[5]) + pad_size_length
              )] = int(splits[7]) + int(splits[8]) + int(splits[9])
              self.worker_cooldown[(
                int(splits[2]),
                int(splits[4]) + pad_size_width,
                int(splits[5]) + pad_size_length
              )] = int(splits[6])
              self.worker_pos_dict[(
                int(splits[4]) + pad_size_width, 
                int(splits[5]) + pad_size_length)] = splits[3] 
            elif splits[0] == "ct":
              city_fuel = cities.get( splits[2] )
              self.city_tiles_cooldown[(
                int(splits[1]), 
                int(splits[3]) + pad_size_width, 
                int(splits[4]) + pad_size_length)] = int(splits[5])
              self.city_tiles_fuel[(
                int(splits[1]), 
                int(splits[3]) + pad_size_width, 
                int(splits[4]) + pad_size_length)] = int(city_fuel)
              self.ct_pos_dict[(
                int(splits[3]) + pad_size_width, 
                int(splits[4]) + pad_size_length)] = splits[2]
                
        self.wood_map = np.expand_dims(self.wood_map, axis=0)
        self.uran_map= np.expand_dims(self.uran_map, axis=0)
        self.coal_map = np.expand_dims(self.coal_map, axis=0)

        self.state = np.concatenate((
        self.wood_map / 1000, self.uran_map / 1000, self.coal_map / 1000, 
        self.worker_cooldown / 2, self.worker_capacity / 100, 
        self.city_tiles_fuel / 1000, self.city_tiles_cooldown / 10 ), axis=0)


def log_to_action(entity_action_prob, is_worker = True):
    entity_action_dim = {
        0: "n",
        1: "s",
        2: "w",
        3: "e",
        4: "stay",
        5: "bcity",
        6: "bw",
        7: "r",
        8: "None"
    }

    if is_worker:
        ordered_actions = [(entity_action_dim[i], entity_action_prob[i]) for i in range(6)]
    else:
        ordered_actions = [(entity_action_dim[i], entity_action_prob[i]) for i in range(6, 9)]

        ordered_actions = sorted(ordered_actions, key=lambda x: x[1], reverse=True)

    return ordered_actions

def action_to_tensor(action_list, worker_pos_dict, ct_pos_dict):
  action_dict = {}
  for action in action_list:
    splits = action.split(" ")
    #print(splits)
    if splits[0] == "m":
      action_dict[splits[1]] = splits[2]
    elif splits[0] == "bcity":
      action_dict[splits[1]] = splits[0]
    elif splits[0] == "bw":
      action_dict[(splits[1], splits[2])] = splits[0]
    elif splits[0] == "r":
      action_dict[(splits[1], splits[2])] = splits[0]

  actions = {
    "n": 0,
    "s": 1,
    "w": 2,
    "e": 3,
    "stay":4,
    "bcity": 5,
    "bw":6,
    "r":7,
    "n":8
  }
#   print(action_dict)

  entity_action_tensor = np.zeros((9, 32, 32))
  if len(worker_pos_dict) > 0:
    for pos, id in worker_pos_dict.items():
      if id not in action_dict:
        entity_action_tensor[5, int(pos[0]), int(pos[1])] = 1
      else:
#         print(action_dict[id])
#         print(actions[action_dict[id]])
        entity_action_tensor[actions[action_dict[id]], pos[0], pos[1]] = 1
    
  if len(ct_pos_dict) > 0:
    for pos, id in ct_pos_dict.items():
      if id not in action_dict:
        entity_action_tensor[6, int(pos[0]), int(pos[1])] = 1
      else:
        entity_action_tensor[actions[action_dict[(int(pos[0]), int(pos[1]))]], int(pos[0]), int(pos[1])] = 1
  #print(entity_action_tensor.shape)
  return entity_action_tensor

import torch 
import torch.nn as nn
import torch.nn.functional as F

def single_conv5(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 5),
    nn.BatchNorm2d(out_channels, eps = 1e-5, momentum=0.1),
    nn.Tanh()
  )

def single_conv3(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3),
    nn.BatchNorm2d(out_channels, eps = 1e-5, momentum=0.1),
    nn.Tanh()
  )

def single_conv2(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 2),
    nn.BatchNorm2d(out_channels, eps = 1e-5, momentum=0.1),
    nn.Tanh()
  )

class Actor(nn.Module):
    def __init__(self, Cin, out_size, seed):

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.maxpool = nn.MaxPool2d(2)

        self.layer1 = single_conv3(Cin, 16)
        self.layer2_1 = single_conv5(16, 32)
        self.layer2_2 = single_conv5(32, 32)
        self.layer2_3 = single_conv3(32, 32)
        self.layer3_1 = single_conv3(32, 32)
        self.layer3_2 = single_conv3(32, 64)
        self.layer4 = single_conv5(64, 128)

        self.fc1 = nn.Sequential(
        nn.Linear(128*12*12, 64),
        nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(64, out_size)

    def forward(self, x1):
        x1 = self.layer1(x1)

        x1 = self.layer2_1(x1)
        #print(f'conv1: {x1.data.cpu().numpy().shape}')

        x1 = self.layer2_2(x1)
        #print(f'conv2: {x1.data.cpu().numpy().shape}')

        x1 = self.layer2_3(x1)
        x1 = self.layer3_1(x1)
        #print(f'conv3: {x1.data.cpu().numpy().shape}')
        x1 = self.layer3_2(x1)
        x1 = self.layer4(x1)
        #print(f'conv4: {x1.data.cpu().numpy().shape}')

        x1 = x1.view(-1, 128*12*12)

        x = self.fc1(x1)
        #print(f'lconv6: {x.data.cpu().numpy().shape}')
        out = self.fc2(x)

        return out
import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict
import os

from torch._C import device
from torch.random import seed

import torch
import torch.nn.functional as F 
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

# BUFFER_SIZE = int(2e5)
BATCH_SIZE = 32
LR_ACTOR = 5e-5 # learning rate of the actor
LR_DECAY = 0.9
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
EPISODES = 1
epsilon = 1
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.001
SHOW_EVERY = 1000
REPEAT = 300

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calc_loss(pred, target, metrics):
    criterion = nn.MSELoss()

    loss = criterion(pred, target)

    pred = torch.argmax(pred, dim=1)

    acc = np.sum(pred.data.cpu().numpy() == target.data.cpu().numpy()) / len(target.data.cpu().numpy())

    metrics['loss'] += loss.data.cpu().numpy()
    #metrics['acc'] = acc
    return loss

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "new_state", "reward", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, new_state, reward, done):
        e = self.experience(state=state, action=action, new_state=new_state, reward=reward, done=done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states1 = torch.from_numpy(np.stack([e.state for e in experiences if e is not None], axis=0)).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None], axis=0)).float().to(device)
        states2 = torch.from_numpy(np.stack([e.new_state for e in experiences if e is not None], axis=0)).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None], axis=0)).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None], axis=0)).float().to(device)

        return (states1, actions, states2, rewards, dones)

    def __len__(self):
        return len(self.memory)


class ActorAgent():
    def __init__(self, cin, out_worker, out_ctiles, random_seed):
        self.cin = cin
        self.out_worker = out_worker
        self.out_ctiles = out_ctiles
#         self.out_ctiles = out_ctiles
        self.seed = random.seed(random_seed)

        self.worker_model = Actor(Cin=cin, out_size=out_worker, seed=random_seed).to(device)
        self.worker_model_optimizer = optim.Adam(self.worker_model.parameters(), lr=LR_ACTOR)
        self.worker_model_scheduler = optim.lr_scheduler.ExponentialLR(self.worker_model_optimizer, gamma=LR_DECAY)
        
        self.target_worker_model = Actor(Cin=cin, out_size=out_worker, seed=random_seed).to(device)
        self.target_worker_model_optimizer = optim.Adam(self.worker_model.parameters(), lr=LR_ACTOR)
        self.target_worker_model_scheduler = optim.lr_scheduler.ExponentialLR(self.worker_model_optimizer, gamma=LR_DECAY)
        
        self.target_worker_model.load_state_dict(self.worker_model.state_dict())
        
        self.ctiles_model = Actor(Cin=cin, out_size=out_ctiles, seed=random_seed).to(device)
        self.ctiles_model_optimizer = optim.Adam(self.ctiles_model.parameters(), lr=LR_ACTOR)
        self.ctiles_model_scheduler = optim.lr_scheduler.ExponentialLR(self.ctiles_model_optimizer, gamma=LR_DECAY)

        self.target_ctiles_model = Actor(Cin=cin, out_size=out_ctiles, seed=random_seed).to(device)
        self.target_ctiles_model_optimizer = optim.Adam(self.ctiles_model.parameters(), lr=LR_ACTOR)
        self.target_ctiles_model_scheduler = optim.lr_scheduler.ExponentialLR(self.ctiles_model_optimizer, gamma=LR_DECAY) 
        
        self.target_ctiles_model.load_state_dict(self.ctiles_model.state_dict())
        
        self.worker_memory = ReplayBuffer(int(1e6), BATCH_SIZE, random_seed)
        self.ctiles_memory = ReplayBuffer(int(1e4), BATCH_SIZE, random_seed)
        
        self.target_update_counter = 0
        
    def learnworker(self, states, target):
        states = states.to(device)
        target = target.to(device)
        pred = self.worker_model(states)

        #     print(pred.shape)
        #     print(target.shape)
        metrics = defaultdict(float)

        loss = calc_loss(pred, target, metrics)

        self.worker_model_optimizer.zero_grad()
        loss.backward()
        self.worker_model_optimizer.step()
        
        return metrics

    def learnctiles(self, states, target):
        states = states.to(device)
        target = target.to(device)

        pred = self.ctiles_model(states)

        metrics = defaultdict(float)
        loss = calc_loss(pred, target, metrics)

        self.ctiles_model_optimizer.zero_grad()
        loss.backward()
        self.ctiles_model_optimizer.step()
        return metrics
    
    def act(self, state, is_worker = True):
        state = torch.from_numpy(state).float().to(device)
        if is_worker:
            self.worker_model.eval()
            with torch.no_grad():
                out = self.worker_model(state)
                out = out.cpu().data.numpy()
            self.worker_model.train()
        else:
            self.ctiles_model.eval()
            with torch.no_grad():
                out = self.ctiles_model(state)
                out = out.cpu().data.numpy()
            self.ctiles_model.train()
        return out    

    def add(self, state, action, new_state, reward, done, is_worker = True):
        if is_worker:
            self.worker_memory.add(state, action, new_state, reward, done)
        else:
            self.ctiles_memory.add(state, action, new_state, reward, done)

    def lr_step(self):
        self.worker_model_scheduler.step()
        self.ctiles_model_scheduler.step()

    def step(self):
        worker_metric_mean = defaultdict(float)
        ctiles_metric_mean = defaultdict(float)
        if len(self.worker_memory) > 20_000:
            for _ in range(100):
                experiences = self.worker_memory.sample()
                worker_metrics = self.learnworker(experiences)
                for key, val in worker_metrics.items():
                    worker_metric_mean[key] += val
            for key, val in worker_metric_mean.items():
                worker_metric_mean[key] = val / 1000
        if len(self.ctiles_memory) > 20_000:
            for _ in range(100):
                experiences = self.ctiles_memory.sample()
                ctiles_metrics = self.learnctiles(experiences)
                for key, val in ctiles_metrics.items():
                    ctiles_metric_mean[key] += val

            for key, val in ctiles_metric_mean.items():
                ctiles_metric_mean[key] = val/ 100

        return worker_metric_mean, ctiles_metric_mean 
    
    def train(self, terminal_state):
        if len(self.worker_memory) < 20_000 or len(self.ctiles_memory) < 10_000:
            return
        worker_minibatch = self.worker_memory.sample()
        ctiles_minibatch = self.ctiles_memory.sample()

        #print(ctiles_minibatch)
        #  transition: (states1, actions, states2, rewards, dones)
        current_worker_states = worker_minibatch[0].to(device)
        current_worker_qs_list = self.worker_model(current_worker_states).to(device)
        new_current_worker_state = worker_minibatch[2].to(device)
        future_qs_worker_list = self.target_worker_model(new_current_worker_state).to(device)

        current_ctiles_states = ctiles_minibatch[0].to(device)
        current_ctiles_qs_list = self.ctiles_model(current_ctiles_states).to(device)
        new_current_ctiles_state = ctiles_minibatch[2].to(device)
        future_qs_ctiles_list = self.target_ctiles_model(new_current_ctiles_state).to(device)

        X_worker = torch.tensor([]).to(device)
        y_worker = torch.tensor([]).to(device)

        for index in range(len(worker_minibatch[0])):
            state = worker_minibatch[0][index]
            action = int(worker_minibatch[1][index])
            new_state = worker_minibatch[2][index]
            reward = worker_minibatch[3][index]
            done = worker_minibatch[4][index]

            if not done:
                max_future_worker_q = torch.max(future_qs_worker_list[index])
                new_q_worker = reward + DISCOUNT * max_future_worker_q
            else:
                new_q_worker = reward
            current_qs = current_worker_qs_list[index]
            current_qs[action] = new_q_worker

            X_worker = torch.cat((X_worker, torch.unsqueeze(state, 0)))
            y_worker = torch.cat((y_worker, torch.unsqueeze(current_qs, 0)))

        X_ctiles = torch.tensor([]).to(device)
        y_ctiles = torch.tensor([]).to(device)

        for index in range(len(ctiles_minibatch)):
            state = ctiles_minibatch[0][index]
            action = int(ctiles_minibatch[1][index])
            new_state = ctiles_minibatch[2][index]
            reward = ctiles_minibatch[3][index]
            done = ctiles_minibatch[4][index]            

            if not done:
                max_future_ctiles_q = torch.max(future_qs_ctiles_list[index])
                new_q_ctiles = reward + DISCOUNT * max_future_ctiles_q
            else:
                new_q_ctiles = reward
            current_qs = current_ctiles_qs_list[index]
            if action > 2: 
                action = 2
            current_qs[action] = new_q_ctiles

            X_ctiles = torch.cat((X_ctiles, torch.unsqueeze(state, 0)))
            y_ctiles = torch.cat((y_ctiles, torch.unsqueeze(current_qs, 0)))

        self.learnworker(X_worker, y_worker)
        self.learnctiles(X_ctiles, y_ctiles)
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_worker_model.load_state_dict(self.worker_model.state_dict())
            self.target_ctiles_model.load_state_dict(self.ctiles_model.state_dict())
            self.target_update_counter = 0
            
def generate_offset_map(Hmap, row_c, col_c):
    Hmap_copy = Hmap.copy()
    v_shift = row_c - 16
    h_shift = col_c - 16
    h_index = (np.arange(32) + h_shift) % 32
    v_index = (np.arange(32) + v_shift) % 32
    temp = Hmap_copy[:, v_index]
    
    return temp[:, :, h_index]

direction_dict = {
        0: Constants.DIRECTIONS.NORTH,
        1: Constants.DIRECTIONS.SOUTH,
        2: Constants.DIRECTIONS.WEST,
        3: Constants.DIRECTIONS.EAST,
}


def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    agent = ActorAgent(cin = 11, out_worker=6, out_ctiles=3, random_seed=42)
    agent.worker_model.load_state_dict(torch.load("./worker_model_v26", map_location=torch.device("cpu")), strict=False, )
    agent.ctiles_model.load_state_dict(torch.load("./ctiles_model_v26", map_location=torch.device("cpu")), strict=False)
    ### AI Code goes down here! ### 
    
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    
    obs = Obs(observation)
    
    actions = []
    current_state = obs.state
    unit_actions = {}
    ctiles_ation = {}
    #print("HERE 2")
    for unit in player.units:
        if unit.can_act():
            if unit.is_worker():
                offset_state = generate_offset_map(current_state, unit.pos.x, unit.pos.y)
                offset_state_expand = np.expand_dims(offset_state, axis=0)
                if np.random.random() > epsilon:

                    player_action = np.argmax( agent.act(offset_state_expand, is_worker=True) )

                    unit_actions[(unit.pos.x, unit.pos.y)] = [offset_state, player_action]
                else:
                    player_action = np.random.randint(0, 6)
                    unit_actions[(unit.pos.x, unit.pos.y)] = [offset_state, player_action]
                #print("WORKER ACTION:",player_action, "STEP:", obs.step)
                action = None
                if player_action < 4:
                    #try:
                        action = unit.move(direction_dict[player_action])
              
                    #except:
                    #    pass
                elif player_action == 5:
                    #try:
                        action = unit.build_city()
                    #except:
                    #    pass
                if action is not None:
                    actions.append(action)
    #print("HERE 3")
    cities = list(player.cities.values())
    if len(cities) > 0:
        for city in cities:
            for city_tile in city.citytiles[::-1]:
                if city_tile.can_act():
                    offset_state = generate_offset_map(current_state, city_tile.pos.x, city_tile.pos.y)
                    offset_state_expand = np.expand_dims(offset_state, axis=0)
                    if np.random.random() > epsilon:
                        ctile_action = np.argmax( agent.act( offset_state_expand, is_worker=False ))

                        ctiles_ation[(city_tile.pos.x, city_tile.pos.y)] = [offset_state, ctile_action]
                    else:
                        ctile_action = np.random.randint(0, 3)
                        ctiles_ation[(city_tile.pos.x, city_tile.pos.y)] = [offset_state, ctile_action]
                    
                    #print("CTILE ACTION:", ctile_action, "STEP: ", obs.step)
                    action = None
                    if ctile_action == 0:
                        #try:
                            action = city_tile.build_worker()
                        #except:
                        #    pass
                    elif ctile_action == 1:
                        #try:
                            action = city_tile.research()
                        #except:
                        #    pass
                    if action is not None:
                        actions.append(action)
    #print("STEP: ",obs.step,"ACTION:", actions)
    return actions
