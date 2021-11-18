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


class LuxNetTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.l1 = nn.Linear(20480, 5)

    def forward(self, x):
        src = torch.transpose(torch.flatten(x, 2, -1), 0, 1) # (20, bs, 1024)
        out = self.encoder(src) # (20, bs, 1024)
        out = torch.transpose(out, 0, 1) # (bs, 20, 1024)
        out = torch.flatten(out, 1, -1) # (bs, 20480)
        p = self.l1(out)
        return p
