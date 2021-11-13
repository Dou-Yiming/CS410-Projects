import numpy as np
import json
from easydict import EasyDict as edict
import yaml
from pathlib import Path
import os
import argparse
import random
from tqdm.notebook import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split

from utils.utils import seed_everything, create_dataset_from_json
from dataset.lux_dataset import LuxDataset
from models.lux_net import LuxNet
from tools.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description='n-gram language model')
    parser.add_argument('--data_path', dest='data_path',
                        help='path of dataset',
                        default='./DATA/', type=str)
    parser.add_argument('--config_path', dest='config_path',
                        help='path of config',
                        default='./configs/default.yml', type=str)
    parser.add_argument('--seed', dest='seed',
                        help='seed',
                        default='42', type=int)
    args = parser.parse_args()
    return args


def get_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    return edict(config)


def main(args, cfg):
    seed_everything(args.seed)

    obses, samples = create_dataset_from_json(args.data_path)
    print('obs:', len(obses), 'sample:', len(samples))
    labels = [sample[-1] for sample in samples]

    model = LuxNet()
    train, val = train_test_split(
        samples, test_size=0.1, random_state=args.seed, stratify=labels)
    train_loader = DataLoader(
        LuxDataset(obses, train),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        LuxDataset(obses, val),
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.TRAIN.OPTIMIZER.BASE_LR)
    
    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=cfg.TRAIN.MAX_EPOCH)


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.config_path)
    main(args, cfg)
