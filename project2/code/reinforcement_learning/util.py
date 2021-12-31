from lux.game import Game, Player
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import numpy as np

import math
import sys
import random


BATCH_SIZE = 256
REPLAY_BUFEER_SIZE = int(2e5)
DEMO_BUFFER_SIZE = int(5e4)
LR_ACTOR = 1e-3
LR_DECAY = 0.99
LR_SCHED_INTERVAL = 200
DISCOUNT = 0.985
UPDATE_TARGET_INTERVAL = 200
SAMPLE_DEMO_NUM = 128
VALUE_WEIGHT1 = 1000
VALUE_WEIGHT2 = 1000
VALUE_WEIGHT_W = 500
LAMBDA = 0.1
EPISODES = 800
START_EPSILON = 1.0
IL_RAND=0.1
EPSILON_DECAY = 0.995
MIN_EPSILON = 1e-3
PRETRAIN_STEPS = 5000
REPLAY_START = 1e4
MAP_WIDTH = 32
MAP_HEIGHT = 32
EXPERT_MARGIN = 0.8
EVAL_MODEL_INTERVAL = 20
WEIGHT_DECAY = 1e-5
PRETRAIN = False
