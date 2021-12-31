from lux.game import Game, Player
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import numpy as np

class State:
    def __init__(self, map_feature, global_feature):
        self.map_feature = map_feature
        self.global_feature = global_feature

class Obs():
    def __init__(self, game_state: Game, player: Player, opponent: Player):
        ''' generate state vector for model input (switch player and opponent when changing sides)

        Args:
            game_state (Game): Game object (after update)

            player (player): player object 
            
            opponent (player): the opponent player
        '''

        width = game_state.map_width
        height = game_state.map_height
        
        # padding
        pad_size_width = (32 - width) // 2
        pad_size_height = (32 - height) // 2

        # ----- map features -----

        # resources
        self.wood_map = np.zeros((32, 32))
        self.coal_map = np.zeros((32, 32))
        self.uran_map = np.zeros((32, 32))

        for i in range(width):
            for j in range(width):
                cell = game_state.map.get_cell(i, j)
                if cell.has_resource():
                    if cell.resource.type == "wood":
                        self.wood_map[(pad_size_width +  i, pad_size_height + j)] = cell.resource.amount
                    if cell.resource.type == "coal":
                        self.coal_map[(pad_size_width +  i, pad_size_height + j)] = cell.resource.amount
                    if cell.resource.type == "uranium":
                        self.uran_map[(pad_size_width +  i, pad_size_height + j)] = cell.resource.amount
        self.wood_map = np.expand_dims(self.wood_map, axis=0)
        self.coal_map = np.expand_dims(self.coal_map, axis=0)
        self.uran_map = np.expand_dims(self.uran_map, axis=0)
        

        # worker state for both side (0 for friendly, 1 for opponent)
        self.worker_exist = np.zeros((2, 32, 32))
        self.worker_cooldown = np.zeros((2, 32, 32))
        self.worker_capacity = np.zeros((2, 32, 32))
        self.player_unit_count = np.array([len(player.units)])
        self.opponent_unit_count = np.array([len(opponent.units)])

        for unit in player.units:
            pos = unit.pos
            self.worker_exist[(0, pad_size_width + pos.x, pad_size_height + pos.y)] = 1
            self.worker_cooldown[(0, pad_size_width + pos.x, pad_size_width + pos.y)] = unit.cooldown
            self.worker_capacity[(0, pad_size_width + pos.x, pad_size_width + pos.y)] = \
                unit.cargo.wood + unit.cargo.coal + unit.cargo.uranium
        
        
            
        for unit in opponent.units:
            pos = unit.pos
            self.worker_exist[(1, pad_size_width + pos.x, pad_size_height + pos.y)] = 1
            self.worker_cooldown[(1, pad_size_width + pos.x, pad_size_width + pos.y)] = unit.cooldown
            self.worker_capacity[(1, pad_size_width + pos.x, pad_size_width + pos.y)] = \
                unit.cargo.wood + unit.cargo.coal + unit.cargo.uranium
        

        # city tile state for both side (0 for friendly, 1 for opponent)

        self.city_tiles_exist = np.zeros((2, 32, 32))
        self.city_tiles_cooldown = np.zeros((2, 32, 32))
        self.city_tiles_duration = np.zeros((2, 32, 32))
        self.player_city_tiles_count = np.array([player.city_tile_count])
        self.opponent_city_tiles_count = np.array([opponent.city_tile_count])

        for city in player.cities.values():
            for city_tile in city.citytiles:
                pos = city_tile.pos
                self.city_tiles_exist[(0, pad_size_width + pos.x, pad_size_height + pos.y)] = 1
                self.city_tiles_cooldown[(0, pad_size_width + pos.x, pad_size_height + pos.y)] = city_tile.cooldown
                self.city_tiles_duration[(0, pad_size_width + pos.x, pad_size_height + pos.y)] = city.fuel / city.light_upkeep

        for city in opponent.cities.values():
            for city_tile in city.citytiles:
                pos = city_tile.pos
                self.city_tiles_exist[(1, pad_size_width + pos.x, pad_size_height + pos.y)] = 1
                self.city_tiles_cooldown[(1, pad_size_width + pos.x, pad_size_height + pos.y)] = city_tile.cooldown
                self.city_tiles_duration[(1, pad_size_width + pos.x, pad_size_height + pos.y)] = city.fuel / city.light_upkeep
                

        # ----- global info -----

        self.step = np.array([game_state.turn])
        self.day_in_cycle = np.array([game_state.turn % 40])

        # research (0 for player, 1 for opponent)
        self.player_research_points = np.array([player.research_points])
        self.opponent_research_points = np.array([opponent.research_points])


        # concat state (Normalization here)
        self.state = State(
            map_feature= np.concatenate((
                self.wood_map / 1000, self.uran_map / 1000, self.coal_map / 1000,    # normalization
                self.worker_exist, self.worker_cooldown / 4, self.worker_capacity / 100, 
                self.city_tiles_exist, self.city_tiles_cooldown / 20, self.city_tiles_duration / 90,
            ), axis=0),
            global_feature= np.concatenate((
                self.step / 360,
                self.day_in_cycle / 40,
                self.player_research_points / 200,
                self.opponent_research_points / 200,
                self.player_unit_count / (width * height),
                self.opponent_unit_count / (width * height),
                self.player_city_tiles_count / (width * height),
                self.opponent_city_tiles_count / (width * height)
            ), axis=0)
        )
