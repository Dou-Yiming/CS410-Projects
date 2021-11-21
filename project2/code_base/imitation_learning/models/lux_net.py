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


class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class LuxNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers, filters = cfg.CONV.LAYERS, cfg.CONV.FILTERS
        self.conv0 = BasicConv2d(20, filters, (3, 3), True)
        self.conv_blocks = nn.ModuleList(
            [BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.bottle_neck = nn.Sequential(
            BasicConv2d(filters, filters//2, (5, 5), True),
            nn.LeakyReLU(inplace=True),
            BasicConv2d(filters//2, filters//4, (5, 5), True),
            nn.LeakyReLU(inplace=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(filters+filters//4, cfg.LINEAR.DIM[0], bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(cfg.LINEAR.DIM[0], cfg.LINEAR.DIM[1], bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(cfg.LINEAR.DIM[1], 5, bias=False)
        )

    def forward(self, x):
        print("city: {} {}".format(np.where(np.array(x[0][10].cpu())>0)[0].shape,np.where(np.array(x[0][10].cpu())>0)[1].shape))
        h = F.leaky_relu_(self.conv0(x))
        for block in self.conv_blocks:
            h = F.leaky_relu_(h + block(h))
        f1 = (h * x[:, :1]).view(h.size(0),h.size(1), -1).sum(-1)  # (bs, filters)
        h = self.bottle_neck(h)
        f2 = (h * x[:, :1]).view(h.size(0),h.size(1), -1).sum(-1)  # (bs, filters)
        f=torch.cat((f1,f2),1)
        p = self.linear(f)
        return p
