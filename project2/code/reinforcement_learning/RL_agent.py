import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict

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
from obs import State 
from Buffer import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorAgent():
    def __init__(self, map_feature_size, global_feature_size, out_size, demostrations):
        self.policy_net = DQNet(map_feature_size, global_feature_size, out_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=LR_DECAY)
        self.DQN_criterion = F.mse_loss


        self.target_net = DQNet(map_feature_size, global_feature_size, out_size).to(device)
        self.update_target_net()
        self.replay_buffer= RLbuffer(capacity=REPLAY_BUFEER_SIZE, permanent_data=(len(demostrations) if demostrations is not None else 0))
        self.demo_buffer = RLbuffer(capacity=DEMO_BUFFER_SIZE, permanent_data=DEMO_BUFFER_SIZE)
        if demostrations is not None:
            self.add_demo_to_buffer(demostrations)

        self.step = 0



    def update_target_net(self):
        '''update target network with policy network
        '''
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        print("[INFO] Target Network Updated")


    def add_demo_to_buffer(self, demo_steps):
        for i in tqdm(range(len(demo_steps))):
            self.demo_buffer.store(np.array(demo_steps[i], dtype=object))
            self.replay_buffer.store(np.array(demo_steps[i], dtype=object))
    
    def store_entries(self, entry: StepEntry):
        self.replay_buffer.store(entry)

    def pre_train(self):
        with tqdm(total=PRETRAIN_STEPS) as t:
            for i in range(PRETRAIN_STEPS):
                dqn_loss, expert_loss = self.train_Q_network(pretrain=True)
                t.set_description("Pretrain")
                t.set_postfix(dqn_loss=dqn_loss, 
                    expert_loss=expert_loss
                )
                t.update(1)

    def save_model(self):
        filepath = "./saved_model/DQN_model_{}".format(time.strftime("%m_%d_%H_%M_%S", time.localtime()))
        torch.save(self.policy_net.state_dict(
        ), filepath)
        return filepath

    def restore_model(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path))
        self.target_net.load_state_dict(torch.load(model_path))

    def DQN_loss(self, state_q_values, next_state_q_values, worker_mask):
        '''Get DQN Loss

        Args:
            state_q_values (tensor (BATCH_SIZE, 32, 32)): state q values
            next_state_q_values (tensor (BATCH_SIZE, 32, 32)): next state q values
            worker_mask (Booltensor): hot tensor showing the active workers

        Returns:
            float: DQN Loss
        '''
        masked_state_q_values = state_q_values[worker_mask]
        masked_next_state_q_values = next_state_q_values[worker_mask]
        return self.DQN_criterion(masked_next_state_q_values, masked_state_q_values)

    def expert_loss(self, state_q_values_pred, action_real, worker_mask, is_demo_mask):
        '''calculate the expert margin loss

        Args:
            state_q_values_pred (tensor (BATCH_SIZE, 32, 32, 6)): the state action q ored
            action_real (tensor (BATCH_SIZE, 32, 32, 6)): the action value 
            worker_mask ([type]): [description]
            is_demo_mask (bool): [description]

        Returns:
            [type]: [description]
        '''
        state_q_values_pred_ = state_q_values_pred[is_demo_mask]
        if state_q_values_pred_.size(0) == 0:
            return 0
        expert_margin = torch.zeros(state_q_values_pred.shape, device=device)
        expert_margin[( 1- F.one_hot(action_real)).bool()] = EXPERT_MARGIN
        q_l = expert_margin + state_q_values_pred
        error = q_l.max(-1)[0] - state_q_values_pred.gather(-1, action_real.unsqueeze(-1)).squeeze()
        error_in_demo = error[is_demo_mask]
        error_in_demo_worker = error_in_demo[worker_mask[is_demo_mask]]
        mask = error_in_demo_worker > 0
        return error_in_demo_worker[mask].sum()

    def act(self, state: State)->np.ndarray:
        
        state = (torch.from_numpy(state.map_feature).float().to(device).unsqueeze(0), 
            torch.from_numpy(state.global_feature).float().to(device).unsqueeze(0))
        self.policy_net.eval()
        
        with torch.no_grad():
            out = self.policy_net(state)
            out = out.cpu().squeeze(0).numpy()
        self.policy_net.train()

        return out
        
    def train_Q_network(self, pretrain=False, update=True):
        '''
        train Q network

        Args:
            pretrain (bool, optional): whether to pretrain. Defaults to False.
            update (bool, optional): whether to update target model. Defaults to True.
        '''

        actual_buffer = self.demo_buffer if pretrain else self.replay_buffer
        
        if not actual_buffer.full():
            return 0, 0
        

        tree_index, minibatch, IS_weight = actual_buffer.sample(BATCH_SIZE)

        np.random.shuffle(minibatch)
        minibatch = minibatch.squeeze(-1)
        # state batch
        state_batch = [data.state for data in minibatch]
        # (map_features, global_feature)
        state_batch = (
            torch.tensor([state.map_feature for state in state_batch], device=device).float(),
            torch.tensor([state.global_feature for state in state_batch], device=device).float()
        )

        # action batch
        action_batch = torch.tensor([data.action for data in minibatch], device=device).long()

        # next state batch
        next_state_batch = [data.next_state for data in minibatch]

        # reward batch
        
        reward_state_batch = torch.tensor([data.reward for data in minibatch], device=device).float()
        reward_state_batch = reward_state_batch.repeat_interleave(MAP_WIDTH * MAP_HEIGHT, 0) \
            .view(BATCH_SIZE, MAP_WIDTH, MAP_HEIGHT)

        # Over batch
        end_batch = [data.game_end for data in minibatch]

        # worker_batch (mask for active worker)
        worker_batch = torch.tensor([data.worker for data in minibatch], device=device).bool()

        number_of_active_worker_batch = worker_batch.long().sum().cpu().item()
        
        demo_batch = torch.tensor([data.is_demo for data in minibatch], device=device).bool()

        number_of_demo_active_worker_batch = worker_batch[demo_batch].long().sum().cpu().item()

        # get masked states
        Cont_mask = torch.tensor(tuple(map(lambda game_end: not game_end, end_batch)), device=device)
        Cont_next_states = (
            torch.tensor([next_state_batch[i].map_feature for i in range(BATCH_SIZE) if not end_batch[i]]).float().to(device),
            torch.tensor([next_state_batch[i].global_feature for i in range(BATCH_SIZE) if not end_batch[i]]).float().to(device)
        )
        state_q_values_pred = self.policy_net(state_batch)
        state_q_values = state_q_values_pred.gather(-1, action_batch.unsqueeze(-1))
        next_state_q_values = torch.zeros(BATCH_SIZE, MAP_WIDTH, MAP_HEIGHT, device=device).float()
        next_state_q_values[Cont_mask] = self.target_net(Cont_next_states).max(-1)[0].detach()

        expected_q_values = (next_state_q_values * DISCOUNT) + reward_state_batch
        
        dqn_loss = self.DQN_loss(state_q_values, expected_q_values.unsqueeze(-1), worker_batch)
        
        expert_loss = self.expert_loss(state_q_values_pred, action_batch, worker_batch, demo_batch)

        average_dqn_loss = dqn_loss.item() / number_of_active_worker_batch if number_of_active_worker_batch != 0 else 1
        average_expert_loss = expert_loss.item() / number_of_demo_active_worker_batch if number_of_demo_active_worker_batch != 0 else 1

        loss = dqn_loss + LAMBDA * expert_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if self.step % LR_SCHED_INTERVAL == 0 and self.step != 0:
            self.lr_scheduler.step()

        if self.step % UPDATE_TARGET_INTERVAL == 0 and self.step != 0:
            self.update_target_net()

        return average_dqn_loss, average_expert_loss

demostrations=[]
agent = ActorAgent(15, 8, 6, demostrations=demostrations)
