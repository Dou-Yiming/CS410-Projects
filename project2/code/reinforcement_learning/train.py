from kaggle_environments import make
import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict
from functools import reduce
import time

import os
from tqdm import tqdm

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
from IL_agent import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

move_offset = {
    0: (0, -1), # 0: North
    1: (1, 0),  # 1: East
    2: (0, 1),  # 2: South
    3: (-1, 0), # 3: West
    4: (0, 0),  # 4: Stay
    5: (0, 0)   # 5: Build
}

direct_move = {
    0: Constants.DIRECTIONS.NORTH,
    1: Constants.DIRECTIONS.EAST,
    2: Constants.DIRECTIONS.SOUTH,
    3: Constants.DIRECTIONS.WEST,
    4: Constants.DIRECTIONS.CENTER
}

def reward_F(game_state: Game, player: Player):
    '''Reward Function
    Args: current game configuration: 
        game_state (Game), player (Player)
    Returns:
        float: reward of the current game
    '''
    reward_workers=0
    reward_citytiles=0
    step = game_state.turn
    remain_steps = 361 - step
    exist_citys=[]
    day_remain_steps= remain_steps % 40 if (remain_steps % 40)!=0 else 40
    if day_remain_steps % 40 < 10:
        night_steps = remain_steps % 10
    else:
        night_steps = 10
    light_steps = day_remain_steps - 10 if day_remain_steps> 10 else 0
    for city in player.cities.values():
        reward_citytiles += min(math.sqrt((city.fuel) / (city.light_upkeep * night_steps)), 1) * VALUE_WEIGHT1 * len(city.citytiles)
        lx,rx,ly,ry=(50,0,50,0)
        # Calculate the surrounding resources
        for citytile in city.citytiles:
            exist_citys.append(citytile)
            pos=citytile.pos
            lx=min(lx,pos.x)
            ly=min(ly,pos.y)
            rx=max(rx,pos.x)
            ry=max(ry,pos.y)
        bbox_area=(rx-lx+1)*(ry-ly+1)
        fuel_r=0
        for i in range(lx,rx+1):
            for j in range(ly,ry+1):
                cell = game_state.map.get_cell(i, j)
                if cell.has_resource():
                    if cell.resource.type == "wood":
                        fuel_r+=1
                    if cell.resource.type == "coal":
                        fuel_r+=10
                    if cell.resource.type == "uranium":
                        fuel_r+=40
        surround_resources=fuel_r/bbox_area
        reward_citytiles += VALUE_WEIGHT2 * surround_resources
                
    for unit in player.units:
        if unit.is_worker():
            fuel_w=(unit.cargo.wood * 1 + unit.cargo.coal * 10 + unit.cargo.uranium * 40)
            distance_to_city= 10000
            unit_pos=unit.pos
            for citytile in exist_citys:
                pos=citytile.pos
                cur_dis=math.sqrt((pos.x-unit_pos.x)*(pos.x-unit_pos.x)+(pos.y-unit_pos.y)*(pos.y-unit_pos.y))
                distance_to_city
            far = distance_to_city < (light_steps - unit.cooldown) 
            if far:
                reward_workers += min(math.sqrt(fuel_w/(4*night_steps)),1) * VALUE_WEIGHT_W * fuel_w
            else:
                reward_workers += min(math.sqrt(fuel_w/distance_to_city) if distance_to_city!=0 else fuel_w,1) * VALUE_WEIGHT_W * fuel_w
            
    reward_proportion=1/(1 + math.exp(-(reward_workers+reward_citytiles)))

    return reward_proportion*((reward_citytiles+reward_workers)**(1/3))

def run_one_episode(agent: ActorAgent, training=False, epsilon=0):# run one episode
    epsilon_1=epsilon
    epsilon_2=IL_RAND
    
    env = make("lux_ai_2021", configuration={"seed": np.random.randint(100000000, 999999999), "loglevel": 0}, debug=True)
    reward_list = []
    total_dqn_loss = 0
    total_expert_loss = 0
    winner_list = []
    for re in range(2):
        # train both side player
        current_obs = env.reset()

        observation = current_obs[0]["observation"]
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        
        player = game_state.players[re]
        opponent = game_state.players[(re + 1) % 2]

        parsed_obs = Obs(game_state, player, opponent)
        current_state = parsed_obs.state

        game_end = False
        last_reward = 0
        dqn_loss_per_episode, expert_loss_per_episode = 0, 0
        step_count = 0
    
        while not game_end:
            # opponent actions
            opponent_actions = baseline_agent(observation, game_state, player=opponent, opponent=player)

            current_turn = game_state.turn
            actions = []

            # avoid move collision
            unit_cant_move_to = [[0 for i in range(game_state.map_height)] for j in range(game_state.map_width)]
            
            # opponent's citytile 
            for city in list(opponent.cities.values()):
                for city_tile in city.citytiles:
                    unit_cant_move_to[city_tile.pos.x][city_tile.pos.y] = 1

            # Night No Out! check
            player_ct = [[0 for i in range(game_state.map_height)] for j in range(game_state.map_width)]

            # avoid building failure
            unit_cant_build_on = [[0 for i in range(game_state.map_height)] for j in range(game_state.map_width)]
            
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

            # active worker map
            worker_active_map = np.zeros((MAP_WIDTH, MAP_HEIGHT))

            # act on state
            out = agent.act(current_state)

            # action map
            action_map = np.zeros((MAP_WIDTH, MAP_HEIGHT))

            active_worker_count = 0
            for unit in player.units:
                if unit.can_act():
                    if unit.is_worker():
                        active_worker_count += 1

                        pos = unit.pos

                        # update active map for memory replay (for model input, so it need padding)
                        worker_active_map[(pad_size_width + pos.x, pad_size_height + pos.y)] = 1

                        random_choice=np.random.random()
                        
                        if random_choice > epsilon_1:
                            player_action_pred = out[(pad_size_width + pos.x), (pad_size_height + pos.y)]
                            player_action_sort = np.argsort(player_action_pred)[::-1]

                            available = False
                            current_action_index = 0
                            player_action = player_action_sort[current_action_index]
                            while not available and current_action_index < 5:
                                # avoid collision
                                pos_offset = move_offset[player_action]
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
                        elif random_choice> epsilon_2:
                            player_action=IL_agent(observation, game_state, player=player, opponent=opponent)
                        else:
                            player_action = np.random.randint(0, 6)
                        
                        # add to action map (for memory replay, so need padding)
                        action_map[(pos.x + pad_size_width, pos.y + pad_size_height)] = player_action
                        action = None
                        if player_action < 5:
                            action = unit.move(direct_move[player_action])
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
            if re == 0:
                new_obs = env.step([actions, opponent_actions])
            else:
                new_obs = env.step([opponent_actions, actions])
            
            observation = new_obs[0]["observation"]
            game_state._update(observation["updates"])
            player = game_state.players[re]
            opponent = game_state.players[(re + 1) % 2]

            parsed_new_obs = Obs(game_state, player, opponent)

            new_reward = reward_F(game_state, player)
            reward = new_reward - last_reward
            last_reward = new_reward

            game_end = new_obs[0]["status"] == "DONE"
            new_state = parsed_new_obs.state

            if active_worker_count != 0 and training:
                agent.store_entries(StepEntry(current_state,action_map,new_state,reward,game_end,worker_active_map,False))
            if training: 
                dqn_loss_per_step, expert_loss_per_step = agent.train_Q_network(pretrain=False, update=True)
                dqn_loss_per_episode += dqn_loss_per_step
                expert_loss_per_episode += expert_loss_per_step
            
            current_obs = new_obs
            parsed_obs = parsed_new_obs
            current_state = new_state
            step_count += 1
        # there we ouput the information of the winner
        winner = None
        if re == 0:
            winner = "Baseline" if current_obs[0]["observation"]["reward"] < current_obs[1]["observation"]["reward"] else "RL"
        else:
            winner = "Baseline" if current_obs[0]["observation"]["reward"] > current_obs[1]["observation"]["reward"] else "RL"
        winner_list.append(winner)
        reward_list.append(last_reward)
        total_dqn_loss += dqn_loss_per_episode / step_count
        total_expert_loss += expert_loss_per_episode / step_count
    return reward_list, total_dqn_loss / 2, total_expert_loss / 2, winner_list

def evaluate(agent: ActorAgent, episodes):
    reward_list = []
    winner_list = []
    for i in range(1, episodes + 1):
        reward, dqn_loss, expert_loss, winner = run_one_episode(
            agent, training=False, epsilon=0)
        print("(EVAL) Epsisode {}/{}: reward: {}, winner: {}".format(i, episodes, reward, winner))
        reward_list.extend(reward)
        winner_list.extend(winner)
    print("(EVAL) Results: {}, win rate: {}".format(
        reduce(lambda x, y: x + y, reward_list) / len(reward_list),
        reduce(lambda x, y: x + y, list(map(lambda x: True if x == 'RL' else 0, winner_list))) / len(winner_list)
    ))


def train(agent: ActorAgent):
    if not PRETRAIN:
        agent.pre_train()
    epsilon = START_EPSILON
    with tqdm(total=EPISODES, desc="(TRAIN)") as t:
        for episode in range(EPISODES):
            t.set_description("Episode: {}".format(episode))
            reward, dqn_loss, expert_loss, winner = run_one_episode(agent, training=True, epsilon=epsilon)
            t.set_postfix(dqn_loss=dqn_loss, expert_loss=expert_loss, memory_size=len(agent.replay_memory), epsilon=epsilon)
            t.update(1)

            if episode % EVAL_MODEL_INTERVAL == 0 and episode != 0:
                evaluate(agent, 5)
                torch.save(agent.policy_net.state_dict(), "./model/dqn_latest")
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

if __name__=='__main__':
    # IL model load
    print("Creating model..")
    IL_model =get_IL_model(game_state)
    print("Load model weight..")
    try:
        IL_model.load_weights( str(p/'IL.h5'),  by_name=True, skip_mismatch=True)
    except Exception as e:
        print('Error in model load')
        print(e)
#   model = tf.keras.models.load_model('IL.h5')
    print("Done crating mdoel")
    agent = ActorAgent(15, 8, 6)    
    if PRETRAIN:
        agent.restore_model('./model/dqn_model')
    train(agent)