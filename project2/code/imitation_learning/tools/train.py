import numpy as np
import json
from easydict import EasyDict as edict
import yaml
from pathlib import Path
import os
import argparse
import random
import math
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from utils.metric import get_map


# def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs):
#     best_acc = 0.0

#     for epoch in range(num_epochs):

#         for phase in ['train', 'val']:
#             model.cuda()
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             epoch_loss = 0.0
#             epoch_acc = 0

#             dataloader = dataloaders_dict[phase]
#             for i, item in tqdm(enumerate(dataloader), leave=False):
#                 states = item[0].cuda().float()
#                 actions = item[1].cuda().long()

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     policy = model(states)

#                     import ipdb
#                     ipdb.set_trace()
#                     loss = criterion(policy, actions)
#                     _, preds = torch.max(policy, 1)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#                         scheduler.step()

#                     epoch_loss += loss.item() * len(policy)
#                     epoch_acc += torch.sum(preds == actions.data)

#             data_size = len(dataloader.dataset)
#             epoch_loss = epoch_loss / data_size
#             epoch_acc = epoch_acc.double() / data_size

#             print(
#                 f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

#             if phase == 'val' and epoch_acc > best_acc:
#                 print(
#                     "Best Score {:.4f} reached, saving model...".format(epoch_acc))
#                 model.eval()
#                 traced = torch.jit.trace(
#                     model.module.cpu(), torch.rand(1, 20, 32, 32))
#                 traced.save('./saved_models/model_global_linear.pth')
#                 best_acc = epoch_acc


def train_epoch(model, dataloader, criterion, optimizer, scheduler):
    model.cuda()
    model.train()

    epoch_loss = 0.0
    epoch_acc = 0
    for item in tqdm(dataloader, leave=False):
        states = item[0].cuda().float()
        actions = item[1].cuda().long()

        optimizer.zero_grad()
        policy = model(states)
        loss = criterion(policy, actions)
        _, preds = torch.max(policy, 1)

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item() * len(policy)
        epoch_acc += torch.sum(preds == actions.data)
    data_size = len(dataloader.dataset)
    epoch_loss = epoch_loss / data_size
    epoch_acc = epoch_acc.double() / data_size
    return epoch_loss, epoch_acc


def eval_epoch(model, dataloader, criterion):
    model.cuda()
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0
    scores, labels = np.array([]), np.array([])
    for item in tqdm(dataloader, leave=False):
        states = item[0].cuda().float()
        actions = item[1].cuda().long()

        policy = model(states)
        loss = criterion(policy, actions)
        _, preds = torch.max(policy, 1)
        epoch_loss += loss.item() * len(policy)
        epoch_acc += torch.sum(preds == actions.data)
        if scores.shape[0] == 0:
            scores = F.softmax(policy.detach(), dim=1).cpu().numpy()
        else:
            scores = np.concatenate(
                (scores, F.softmax(policy.detach(), dim=1).cpu().numpy()), axis=0)
        if labels.shape[0] == 0:
            labels = np.eye(5)[actions.detach().cpu().numpy()]
        else:
            labels = np.concatenate(
                (labels, np.eye(5)[actions.detach().cpu().numpy()]), axis=0)

    data_size = len(dataloader.dataset)
    epoch_loss = epoch_loss / data_size
    epoch_acc = epoch_acc.double() / data_size
    return epoch_loss, epoch_acc, scores, labels


def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs):
    bst = 0.0
    lowest_loss = math.inf
    for epoch in range(num_epochs):
        # eval
        epoch_loss, epoch_acc, scores, labels = eval_epoch(
            model, dataloaders_dict['val'], criterion)
        ap = get_map(labels, scores)
        mAP = np.mean(ap) * 100
        print(
            f'Epoch {epoch + 1}/{num_epochs} | eval | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | mAP: {mAP:.2f} | AP: {ap * 100}')
        # save
        if epoch_acc > bst:
            print(
                "Best score {:.6f} reached, saving model...".format(epoch_acc))
            model.eval()
            traced = torch.jit.trace(
                model.module.cpu(), torch.rand(1, 20, 32, 32))
            traced.save('./saved_models/model_finetune_top_on_1800_on_top.pth')
            # lowest_loss = epoch_loss
            bst = epoch_acc
        # train
        epoch_loss, epoch_acc = train_epoch(
            model, dataloaders_dict['train'], criterion, optimizer, scheduler)
        print(
            f'Epoch {epoch + 1}/{num_epochs} | train | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
