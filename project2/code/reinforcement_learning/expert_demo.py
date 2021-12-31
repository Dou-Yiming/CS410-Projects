from lux.game import Game
from tqdm import tqdm
import numpy as np
import json
import os

from model import *
from train import reward_F
from util import *
from obs import * 
from Buffer import *
from RL_agent import *
from baseline_agent import *
from IL_agent import *

demostrations = []
path = "./data"
files = os.listdir(path)

file_count = 0
stop_loading = False
for file in files:
    file_count += 1
    with open(path + '/' + file, 'r') as f:
        player_index = np.random.randint(0, 2)
        json_data = json.load(f)
        steps = json_data['steps']

        current_obs = steps[0]
        observation = current_obs[0]['observation']
        
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"])
        game_state.id = observation["player"]

        player = game_state.players[player_index]
        opponent = game_state.players[(player_index + 1) % 2]

        parsed_obs = Obs(game_state, player, opponent)
        current_state = parsed_obs.state

        done = False

        last_reward = 0

        pad_size_width = (32 - game_state.map_width) // 2
        pad_size_height = (32 - game_state.map_height) // 2

        for i in tqdm(range(len(steps))):
            step = steps[i]
            opponent_actions = step[(player_index + 1) % 2]["action"]
            player_actions = step[player_index]["action"]

            new_obs = step
            observation = new_obs[0]['observation']
            game_state._update(observation["updates"])
            player = game_state.players[player_index]
            opponent = game_state.players[(player_index + 1) % 2]

            parsed_new_obs = Obs(game_state, player, opponent)
            new_reward = reward_F(game_state, player)
            reward = new_reward - last_reward
            
            done = new_obs[0]["status"] == "DONE"
            new_state = parsed_new_obs.state

            # parse actions
            actions = {}
            if player_actions is not None:
                for action in player_actions:
                    splits = action.split(" ")

                    if splits[0] == 'm':
                        if splits[2] == 'n':
                            actions[splits[1]] = 0
                        elif splits[2] == 'e':
                            actions[splits[1]] = 1
                        elif splits[2] == 's':
                            actions[splits[1]] = 2
                        elif splits[2] == 'w':
                            actions[splits[1]] = 3
                        elif splits[2] == 'c':
                            actions[splits[1]] = 4
                    elif splits[0] == 'bcity':
                        actions[splits[1]] = 5

            # add action to demostrations
            worker_active_map = np.zeros((MAP_WIDTH, MAP_WIDTH))
            action_map = np.zeros((MAP_WIDTH,MAP_HEIGHT))
            for unit in player.units:
                if unit.can_act():
                    pos = unit.pos
                    unit_action = actions.get(unit.id)
                    if unit_action is None:
                        continue
                    worker_active_map[(pos.x + pad_size_width, pos.y + pad_size_height)] = 1
                    action_map[(pos.x + pad_size_width, pos.y + pad_size_height)] = unit_action
            demostrations.append(StepEntry(current_state, action_map, new_state, reward, done, worker_active_map, is_demo=True))

            current_obs = new_obs
            parsed_obs = parsed_new_obs
            current_state = new_state

            if (len(demostrations) == DEMO_BUFFER_SIZE):
                stop_loading = True
                break
    if stop_loading:
        break