from Memory import Memory
import torch.optim as optim
from torch.random import seed
from torch._C import device
from tqdm import tqdm
import os
import time
from collections import namedtuple, deque, defaultdict
import copy
import torch.nn.functional as F
import torch.nn as nn
import torch
from lux.game import Game, Player
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import numpy as np
from model import *
from train import reward_F
from util import *
from obs import * 
from Buffer import *
from RL_agent import *
from baseline_agent import *
from IL_agent import *

game_state = None
path = '/kaggle_simulations/agent' if os.path.exists(
    '/kaggle_simulations') else '.'

actor_agent = ActorAgent(map_feature_size=15, global_feature_size=8, out_size=6, demostrations=None)
actor_agent.policy_net.load_state_dict(torch.load(
    f"{path}/model/dqn.h5", map_location=torch.device("cpu")), strict=False, )

epsilon = 0
last_reward = 0

def agent(observation, configuration):
    global game_state, last_reward

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])

    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]

    obs = Obs(game_state, player, opponent)
    
    new_reward = reward_F(game_state=game_state, player=player)
    print("last reward: {}".format(new_reward))
    print("delta reward: {}".format(new_reward - last_reward))
    last_reward = new_reward

    current_state = obs.state
    actions = []
    current_turn = game_state.turn
    # avoid move collision
    # no padding here (just for output check, not for model input)
    unit_cant_move_to = [
        [0 for i in range(game_state.map_height)] for j in range(game_state.map_width)]
    # opponent's citytile
    for city in list(opponent.cities.values()):
        for city_tile in city.citytiles:
            unit_cant_move_to[city_tile.pos.x][city_tile.pos.y] = 1

    # avoid building failure
    unit_cant_build_on = [
        [0 for i in range(game_state.map_height)] for j in range(game_state.map_width)]

    # Night No Out! check
    player_ct = [[0 for i in range(game_state.map_height)]
                 for j in range(game_state.map_width)]
    # player's citytile
    for city in list(player.cities.values()):
        for city_tile in city.citytiles:
            unit_cant_build_on[city_tile.pos.x][city_tile.pos.y] = 1
            player_ct[city_tile.pos.x][city_tile.pos.y] = 1

    # resource
    for i in range(game_state.map_width):
        for j in range(game_state.map_height):
            if game_state.map.get_cell(i, j).has_resource():
                unit_cant_build_on[i][j] = 1

    # padding for model input
    pad_size_width = (32 - game_state.map_width) // 2
    pad_size_height = (32 - game_state.map_height) // 2


    # act on state
    out = actor_agent.act(current_state)

    active_worker_count = 0
    for unit in player.units:
        if unit.can_act():
            if unit.is_worker():
                active_worker_count += 1

                pos = unit.pos

                # In the real match, we doesn't use the IL model
                if np.random.random() > epsilon:
                    player_action_pred = out[(
                        pad_size_width + pos.x), (pad_size_height + pos.y)]
                    player_action_sort = np.argsort(
                        player_action_pred)[::-1]

                    available = False
                    current_action_index = 0
                    player_action = player_action_sort[current_action_index]
                    while not available and current_action_index < 5:
                        # avoid collision
                        pos_offset = pos_offset_dict[player_action]
                        unit_new_pos_x = pos.x + pos_offset[0]
                        unit_new_pos_y = pos.y + pos_offset[1]
                        if player_action < 5:
                            if unit_new_pos_x < game_state.map_width and unit_new_pos_x >= 0 and \
                                    unit_new_pos_y < game_state.map_height and unit_new_pos_y >= 0:
                                if unit_cant_move_to[unit_new_pos_x][unit_new_pos_y] != 0:
                                    available = False
                                elif current_turn % 40 >= 30 and \
                                        player_ct[pos.x][pos.y] == 1 and player_ct[unit_new_pos_x][unit_new_pos_y] == 0:
                                    # Night No Out Check
                                    available = False
                                else:
                                    unit_cant_move_to[unit_new_pos_x][unit_new_pos_y] = 1
                                    available = True
                            else:
                                available = False
                        # avoid building failure
                        elif player_action == 5:
                            if unit_new_pos_x < game_state.map_width and unit_new_pos_x >= 0 and \
                                    unit_new_pos_y < game_state.map_height and unit_new_pos_y >= 0 and \
                                    unit_cant_build_on[unit_new_pos_x][unit_new_pos_y] == 1:
                                available = False
                            elif unit.cargo.wood + unit.cargo.coal + unit.cargo.uranium != 100:
                                available = False
                            else:
                                available = True
                        if not available:
                            current_action_index += 1
                            player_action = player_action_sort[current_action_index]
                else:
                    player_action = np.random.randint(0, 6)
                print(unit.id)
                print(player_action_pred, player_action)
                print(player_action_sort)
                
                action = None
                if player_action < 5:
                    action = unit.move(direction_dict[player_action])
                elif player_action == 5:
                    action = unit.build_city()

                if action is not None:
                    actions.append(action)
    cities = list(player.cities.values())
    number_of_citytiles = player.city_tile_count
    number_of_units = len(player.units)
    if len(cities) > 0:
        for city in cities:
            for city_tile in city.citytiles[::-1]:
                if city_tile.can_act():
                    ctile_action = 0
                    if number_of_units < number_of_citytiles:
                        ctile_action = 0
                        number_of_units += 1
                    else:
                        ctile_action = 1
                    action = None
                    if ctile_action == 0:
                        action = city_tile.build_worker()
                    elif ctile_action == 1:
                        action = city_tile.research()

                    if action is not None:
                        actions.append(action)
    return actions
