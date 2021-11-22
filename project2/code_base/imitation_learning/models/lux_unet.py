# Neural Network for Lux AI
import numpy as np
import json
from pathlib import Path
import os
import random
from tqdm.notebook import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pprint import pprint

from utils.UNet import UNet


class LuxUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers, filters = cfg.CONV.LAYERS, cfg.CONV.FILTERS
        self.backbone = UNet(input_channel=20, output_channel=cfg.CONV.FILTERS)
        self.linear = nn.Sequential(
            nn.Linear(cfg.CONV.FILTERS, cfg.LINEAR.DIM[0], bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(cfg.LINEAR.DIM[0], 5, bias=False)
        )
    def forward(self, x):
        h=self.backbone(x)
        h = (h * x[:, :1]).view(h.size(0),h.size(1), -1).sum(-1)  # (bs, filters)
        p = self.linear(h)
        return p
