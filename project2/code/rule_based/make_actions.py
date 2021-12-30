# functions executing the actions

import builtins as __builtin__
from typing import Tuple, List

from numpy import matrix, vectorize

from lux.game import Game, Mission, Missions
from lux.game_objects import CityTile, Unit
from lux.game_position import Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS

from heuristics import find_best_cluster, find_city_tokeep, find_best_citypos

DIRECTIONS = Constants.DIRECTIONS
eps = 0.2
use_Astar = False
class anode:
    def __init__(self, x: int,y:int,distance:float = 0, f:float = 0):
        self.x:int = x
        self.y:int = y
        self.distance:float = y
        self.f:float = f


def make_city_actions(game_state: Game, missions: Missions, DEBUG=False) -> List[str]:
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    player = game_state.player
    missions.cleanup(player,
                     game_state.player_city_tile_xy_set,
                     game_state.opponent_city_tile_xy_set,
                     game_state.convolved_collectable_tiles_xy_set)
    game_state.repopulate_targets(missions)

    units_cap = sum([len(x.citytiles) for x in player.cities.values()])
    units_cnt = len(player.units)  # current number of units

    actions: List[str] = []

    def do_research(city_tile: CityTile):
        action = city_tile.research()
        game_state.player.research_points += 1
        actions.append(action)

    def build_workers(city_tile: CityTile):
        nonlocal units_cnt
        action = city_tile.build_worker()
        actions.append(action)
        units_cnt += 1

    city_tiles: List[CityTile] = []
    for city in player.cities.values():
        for city_tile in city.citytiles:
            city_tiles.append(city_tile)
    if not city_tiles:
        return []

    city_tiles.sort(key=lambda city_tile:
        (city_tile.pos.x*game_state.x_order_coefficient, city_tile.pos.y*game_state.y_order_coefficient))

    for city_tile in city_tiles:
        if not city_tile.can_act():
            continue

        unit_limit_exceeded = (units_cnt >= units_cap)

        cluster_leader = game_state.xy_to_resource_group_id.find(tuple(city_tile.pos))
        cluster_unit_limit_exceeded = \
            game_state.xy_to_resource_group_id.get_point(tuple(city_tile.pos)) <= len(game_state.resource_leader_to_locating_units[cluster_leader])
        if cluster_unit_limit_exceeded:
            print("unit_limit_exceeded", city_tile.cityid, tuple(city_tile.pos))

        if player.researched_uranium() and unit_limit_exceeded:
            print("skip city", city_tile.cityid, tuple(city_tile.pos))
            continue

        if not player.researched_uranium() and game_state.turns_to_night < 3:
            print("research and dont build units at night", tuple(city_tile.pos))
            do_research(city_tile)
            continue

        nearest_resource_distance = game_state.distance_from_collectable_resource[city_tile.pos.y, city_tile.pos.x]
        travel_range = game_state.turns_to_night // GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
        resource_in_travel_range = nearest_resource_distance < travel_range

        if resource_in_travel_range and not unit_limit_exceeded and not cluster_unit_limit_exceeded:
            print("build_worker", city_tile.cityid, city_tile.pos.x, city_tile.pos.y, nearest_resource_distance, travel_range)
            build_workers(city_tile)
            continue

        if not player.researched_uranium():
            print("research", tuple(city_tile.pos))
            do_research(city_tile)
            continue

        # otherwise don't do anything

    return actions


def make_unit_missions(game_state: Game, missions: Missions, DEBUG=False) -> Missions:
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    player = game_state.player
    missions.cleanup(player,
                     game_state.player_city_tile_xy_set,
                     game_state.opponent_city_tile_xy_set,
                     game_state.convolved_collectable_tiles_xy_set)

    unit_ids_with_missions_assigned_this_turn = set()

    player.units.sort(key=lambda unit:
        (unit.pos.x*game_state.x_order_coefficient, unit.pos.y*game_state.y_order_coefficient, unit.encode_tuple_for_cmp()))

    for unit in player.units:
        # mission is planned regardless whether the unit can act
        isgiven = False
        current_mission: Mission = missions[unit.id] if unit.id in missions else None
        current_target_position = current_mission.target_position if current_mission else None

        # avoid sharing the same target
        game_state.repopulate_targets(missions)

        # if the unit is waiting for dawn at the side of resource
        stay_up_till_dawn = (unit.get_cargo_space_left() <= 4 and (not game_state.is_day_time or game_state.turn%40 == 0))
        # 
        # go to an empty tile and build a citytile
        # print(unit.id, unit.get_cargo_space_left())

        #build ciy mission
        if (unit.get_cargo_space_left() == 0 or stay_up_till_dawn)and game_state.turns_to_night > 3:
            nearest_best_position, distance_with_features = find_best_citypos(game_state, unit, current_target_position = current_target_position, DEBUG=DEBUG)
            if stay_up_till_dawn or distance_with_features[0] * 2 <= game_state.turns_to_night - 3:
                print("plan mission to build citytile", unit.id, unit.pos, "->", nearest_best_position)
                mission = Mission(unit.id, nearest_best_position, unit.build_city())
                missions.add(mission)
                continue

        if unit.id in missions:
            mission: Mission = missions[unit.id]
            if mission.target_position == unit.pos:
                # take action and not make missions if already at position
                continue

        if unit.id in missions:
            # the mission will be recaluated if the unit fails to make a move after make_unit_actions
            continue

        # keep a city
        if unit.cargo.coal + unit.cargo.uranium > 10 or (game_state.turns_to_night < 10 and unit.cargo.wood + unit.cargo.coal + unit.cargo.uranium > 0 ):
            city_position, best_city_value, best_city_id = find_city_tokeep(game_state, unit, DEBUG=DEBUG)
            if best_city_value != 0:
                print("plan to keep city", unit.id, unit.pos, "->", city_position)
                mission = Mission(unit.id, city_position, None, cityid = int(best_city_id.split('_')[-1]))
                missions.add(mission)
                unit_ids_with_missions_assigned_this_turn.add(unit.id)
                isgiven = True

        if unit.id in missions:
            mission: Mission = missions[unit.id]
            if mission.target_position == unit.pos:
                # take action and not make missions if already at position
                continue

        if unit.id in missions:
            # the mission will be recaluated if the unit fails to make a move after make_unit_actions
            continue
        #find resource
        if isgiven:
            continue

        best_position, best_cell_value = find_best_cluster(game_state, unit, DEBUG=DEBUG)
        # [TODO] what if best_cell_value is zero
        if best_cell_value != (0,0,0,0):
            distance_from_best_position = game_state.retrieve_distance(unit.pos.x, unit.pos.y, best_position.x, best_position.y)
            print("plan mission adaptative", unit.id, unit.pos, "->", best_position)
            mission = Mission(unit.id, best_position, None)
            missions.add(mission)
            unit_ids_with_missions_assigned_this_turn.add(unit.id)
        
        

        # [TODO] just let units die perhaps

    return missions


def make_unit_actions(game_state: Game, missions: Missions, DEBUG=False) -> Tuple[Missions, List[str]]:
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    player, opponent = game_state.player, game_state.opponent
    actions = []

    units_with_mission_but_no_action = set(missions.keys())
    prev_actions_len = -1

    # repeat attempting movements for the units until no additional movements can be added
    while prev_actions_len < len(actions):
      prev_actions_len = len(actions)

      for unit in player.units:
        if not unit.can_act():
            units_with_mission_but_no_action.discard(unit.id)
            continue

        # if there is no mission, continue
        if unit.id not in missions:
            units_with_mission_but_no_action.discard(unit.id)
            continue

        mission: Mission = missions[unit.id]
        print("attempting action for", unit.id, unit.pos, "->", mission.target_position)

        # if the location is reached, take action
        if unit.pos == mission.target_position:
            units_with_mission_but_no_action.discard(unit.id)
            print("location reached and make action", unit.id, unit.pos)
            action = mission.target_action

            # do not build city at last light
            if action and action[:5] == "bcity" and game_state.turn%40 == 30:
                del missions[unit.id]
                continue

            if action:
                actions.append(action)
            del missions[unit.id]
            continue

        # attempt to move the unit
        if use_Astar:
            path= Astarpathing(game_state, unit, mission.target_position)
            if  path != None: 
                direction = path
        else:
            direction = attempt_direction_to(game_state, unit, mission.target_position)
        if direction != "c":
            units_with_mission_but_no_action.discard(unit.id)
            action = unit.move(direction)
            print("make move", unit.id, unit.pos, direction, unit.pos.translate(direction, 1))
            actions.append(action)
            continue

        # [TODO] make it possible for units to swap positions

    # if the unit is not able to make an action, delete the mission
    for unit_id in units_with_mission_but_no_action:
        mission: Mission = missions[unit_id]
        mission.delays += 1
        if mission.delays >= 1:
            del missions[unit_id]

    return missions, actions


def attempt_direction_to(game_state: Game, unit: Unit, target_pos: Position) -> DIRECTIONS:

    smallest_cost = [2,2,2,2]
    closest_dir = DIRECTIONS.CENTER
    closest_pos = unit.pos

    for direction in game_state.dirs:
        newpos = unit.pos.translate(direction, 1)

        cost = [0,0,0,0]

        # do not go out of map
        if tuple(newpos) in game_state.xy_out_of_map:
            continue

        # discourage if new position is occupied, not your city tile and not your current position
        if tuple(newpos) in game_state.occupied_xy_set:
            if tuple(newpos) not in game_state.player_city_tile_xy_set:
                if tuple(newpos) != tuple(unit.pos):
                    cost[0] = 3

        # discourage going into a city tile if you are carrying substantial wood
        if tuple(newpos) in game_state.player_city_tile_xy_set and (unit.cargo.wood + unit.cargo.coal + unit.cargo.uranium >= 50 and game_state.turns_to_night > 5):
            cost[0] = 1

        # path distance as main differentiator
        path_dist = game_state.retrieve_distance(newpos.x, newpos.y, target_pos.x, target_pos.y)
        cost[1] = path_dist

        # manhattan distance to tie break
        manhattan_dist = (newpos - target_pos)
        cost[2] = manhattan_dist

        # prefer to walk on tiles with resources
        aux_cost = game_state.convolved_collectable_tiles_matrix[newpos.y, newpos.x]
        cost[3] = -aux_cost

        # if starting from the city, consider manhattan distance instead of path distance
        if tuple(unit.pos) in game_state.player_city_tile_xy_set:
            cost[1] = manhattan_dist

        # update decision
        if cost < smallest_cost or newpos == target_pos:
            smallest_cost = cost
            closest_dir = direction
            closest_pos = newpos

    if closest_dir != DIRECTIONS.CENTER:
        game_state.occupied_xy_set.discard(tuple(unit.pos))
        if tuple(closest_pos) not in game_state.player_city_tile_xy_set:
            game_state.occupied_xy_set.add(tuple(closest_pos))
        unit.cooldown += 2

    return closest_dir

def Astarpathing(game_state: Game,unit: Unit,target_pos: Position) -> DIRECTIONS:
    def caculate_neighbors(curposition = None):
        num = 0
        if (curposition.x + 1, curposition.y + 0) in game_state.player_city_tile_xy_set:
            num += 1
        if (curposition.x - 1, curposition.y + 0) in game_state.player_city_tile_xy_set:
            num += 1
        if (curposition.x + 0, curposition.y + 1) in game_state.player_city_tile_xy_set:
            num += 1
        if (curposition.x + 0, curposition.y - 1) in game_state.player_city_tile_xy_set:
            num += 1
        return num

    begin_point=[unit.pos.x,unit.pos.y]
    target_point=[target_pos.x,target_pos.y]
    # the cost map which pushes the path closer to the goal
    heuristic = [[0 for row in range(game_state.map_height)] for col in range(game_state.map_width)]
    for i in range(game_state.map_height):
        for j in range(game_state.map_width):
            heuristic[i][j] = abs(i - target_point[0]) + abs(j - target_point[1])-eps*caculate_neighbors(Position(i,j))
            if (i,j) in game_state.xy_out_of_map or (i,j) in game_state.occupied_xy_set:
                heuristic[i][j] = 99  # added extra penalty in the heuristic map



    close_matrix = [[0 for col in range(game_state.map_width)] for row in range(game_state.map_height)]  # the referrence grid
    close_matrix[begin_point[0]][begin_point[1]] = 1
    action_matrix = [[0 for col in range(game_state.map_width)] for row in range(game_state.map_height)]  # the action grid

    x = begin_point[0]
    y = begin_point[1]
    g = 0
    f = g + heuristic[begin_point[0]][begin_point[0]]
    cell = [[f, g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False  # flag set if we can't find expand

    while not found and not resign:
        if len(cell) == 0:
            resign = True
            return None, None
        else:
            cell.sort()  # to choose the least costliest action so as to move closer to the goal
            cell.reverse()
            next = cell.pop()
            x = next[2]
            y = next[3]
            g = next[1]
            f = next[0]

            if x == target_point[0] and y == target_point[1]:
                found = True
            else:
                # delta have four steps
                for i in game_state.dirs:
                    newpos = unit.pos.translate(i, 1)
                    if tuple(newpos) in game_state.occupied_xy_set and tuple(newpos) not in game_state.player_city_tile_xy_set and tuple(newpos) != tuple(unit.pos):
                        continue
                      # to try out different valid actions
                    x2 = newpos.x
                    y2 = newpos.y
                    if tuple(newpos) not in game_state.xy_out_of_map: 
                        if close_matrix[x2][y2] == 0:
                            g2 = g + 1
                            f2 = g2 + heuristic[x2][y2]
                            cell.append([f2, g2, x2, y2])
                            close_matrix[x2][y2] = 1
                            action_matrix[x2][y2] = i
    invpath = []
    x = target_point[0]
    y = target_point[1]
    invpath.append([x, y]) 
    while x != begin_point[0] or y != begin_point[1]:
        newpos = Position(x,y).transtranslate(action_matrix[x][y],1)
        x = newpos.x
        y = newpos.y
        invpath.append(action_matrix[x][y])

    path = []
    for i in range(len(invpath)):
        path.append(invpath[len(invpath) - 1 - i])
    return path[0]
