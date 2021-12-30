import numpy as np
import json
from easydict import EasyDict as edict
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import yaml
from pathlib import Path
import os
import argparse
import random
import pickle
from tqdm.notebook import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

from utils.utils import seed_everything, create_dataset_from_json
from dataset.lux_dataset import LuxDataset
from models.lux_net import LuxNet
from models.lux_unet import LuxUNet
from tools.train import train_model
from utils.losses import FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(description='n-gram language model')
    parser.add_argument('--data_path', dest='data_path',
                        help='path of dataset',
                        default='E:/Datasets/LUX_ai/DATA/top/', type=str)
    parser.add_argument('--config_path', dest='config_path',
                        help='path of config',
                        default='./configs/default.yml', type=str)
    parser.add_argument('--seed', dest='seed',
                        help='seed',
                        default='42', type=int)
    parser.add_argument('--ckpt', dest='ckpt',
                        help='path of check point',
                        default='', type=str)
    args = parser.parse_args()
    return args


def get_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    return edict(config)


def load_data(data_path):
    with open(os.path.join(data_path, 'obses.pkl'), 'rb') as f:
        obses = pickle.load(f)
    with open(os.path.join(data_path, 'train_top.pkl').format(data_path), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(data_path, 'val_top.pkl').format(data_path), 'rb') as f:
        val = pickle.load(f)
    return obses, train, val


loss_dict = {
    'FocalLoss': FocalLoss(num_class=5, gamma=2),
    'CrossEntropyLoss': nn.CrossEntropyLoss()
}


def main(args, cfg):
    seed_everything(args.seed)

    obses, train, val = load_data(args.data_path)
    print('obs:', len(obses), 'sample:', len(train)+len(val))
    print("Train: {}, Val: {}".format(len(train), len(val)))

    # load ckpt
    ckpt = args.ckpt
    if ckpt == '':
        model = LuxNet(cfg=cfg.MODEL)
    else:
        print("Loading checkpoint from {}".format(ckpt))
        model = torch.jit.load(ckpt)

    train_loader = DataLoader(
        LuxDataset(obses, train),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )
    val_loader = DataLoader(
        LuxDataset(obses, val),
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}

    print("Data loader: Train: {}, Val: {}".format(
        len(train_loader), len(val_loader)))

    criterion = loss_dict[cfg.Loss]
    if cfg.TRAIN.OPTIMIZER.TYPE == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.TRAIN.OPTIMIZER.BASE_LR,
            weight_decay=1e-4)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.TRAIN.OPTIMIZER.BASE_LR,
            momentum=0.9, weight_decay=1e-6)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train) // cfg.TRAIN.BATCH_SIZE * 20,
        T_mult=2,
        eta_min=cfg.TRAIN.SCHEDULER.MIN_LR)

    if cfg.MODEL.PARALLEL:
        model = nn.DataParallel(model)

    train_model(model, dataloaders_dict, criterion,
                optimizer, scheduler,
                num_epochs=cfg.TRAIN.MAX_EPOCH)


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.config_path)
    main(args, cfg)
