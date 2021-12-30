# Neural Network for Lux AI
import numpy as np
import os
import random
from sklearn.utils.validation import check_non_negative
import torch
from torch import nn, einsum
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import ipdb
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 正则化
        self.fn = fn  # 具体的操作

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # 前向传播
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    # attention
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 计算最终进行全连接操作时输入神经元的个数
        project_out = not (heads == 1 and dim_head ==
                           dim)  # 多头注意力并且输入和输出维度相同时为True

        self.heads = heads  # 多头注意力中“头”的个数
        self.scale = dim_head ** -0.5  # 缩放操作，论文 Attention is all you need 中有介绍

        self.attend = nn.Softmax(dim=-1)  # 初始化一个Softmax操作
        self.to_qkv = nn.Linear(
            dim, inner_dim * 3, bias=False)  # 对Q、K、V三组向量先进性线性操作

        # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的“头”数
        print(x[:, 0].shape)
        print(self.to_qkv)
        x = self.to_qkv(x[:, 0])
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 先对Q、K、V进行线性操作，然后chunk乘三三份
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), qkv)  # 整理维度，获得Q、K、V

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * \
            self.scale  # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子

        attn = self.attend(dots)  # 做Softmax运算

        # Softmax运算结果与Value向量相乘，得到最终结果
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新整理维度
        return self.to_out(out)  # 做线性的全连接操作或者空操作（空操作直接输出out）


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # Transformer包含多个编码器的叠加
        for _ in range(depth):
            # 编码器包含两大块：自注意力模块和前向传播模块
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                                       dim_head=dim_head, dropout=dropout)),  # 多头自注意力模块
                PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout))  # 前向传播模块
            ]))

    def forward(self, x):
        print(x.shape)
        for attn, ff in self.layers:
            # 自注意力模块和前向传播模块都使用了残差的模式
            x = attn(x) + x
            x = ff(x) + x
        return x


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


class LuxNetTransformer(nn.Module):
    def __init__(self, image_size=32,
                 patch_size=4,
                 num_classes=5,
                 embedding_dim=64,
                 num_layers=12,
                 heads=2,
                 mlp_dim=64,
                 channels=20,
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        self.conv0 = BasicConv2d(20, 256, (3, 3), True)
        self.conv_blocks = nn.ModuleList(
            [BasicConv2d(256, 256, (3, 3), True) for _ in range(12)])
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * (patch_size ** 2)
        # assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1, 32))
        # self.patch_to_embedding = nn.Linear(
        #     patch_dim, embedding_dim)  # patch->embedding
        # self.patch_to_embedding = nn.Sequential(
        #     nn.Linear(patch_dim, embedding_dim),
        #     nn.LeakyReLU(),
        # )
        self.cls_token = nn.Parameter(torch.randn(1, 1, 32))
        self.dropout = nn.Dropout(emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes, bias=True),
        )
    def forward(self, x, mask=None):
        # h = F.leaky_relu_(self.conv0(x))
        # for block in self.conv_blocks:
        #     h = F.leaky_relu_(h + block(h))
        # h = (h * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)
        # h = torch.stack(torch.split(h, 32, dim=1), dim=1)
        # b, n, _ = h.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # h = torch.cat((cls_tokens, h), dim=1)
        # # print(h.shape)

        # # h += self.pos_embedding[:, :(n + 1)]
        # # print(h.shape)
        # # h = self.dropout(h)

        # print(h.view(h.size(1), h.size(0), h.size(2)).shape)
        # h=h.view(h.size(1), h.size(0), h.size(2))
        print(self.encoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8).cuda()
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).cuda()
        h = torch.randn(9, 128, 32).cuda()
        out = transformer_encoder(h)
        # h = self.encoder(h)
        # h=torch.flatten(h,1,-1)
        print(h.shape)
        # prob=self.linear(h)

        h = self.to_cls_token(h[:, 0])
        prob = self.mlp_head(h)
        return prob
