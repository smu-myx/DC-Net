import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.block_others import conv_block_k3, attention_block


class Decomposition(nn.Module):

    def __init__(self):
        super().__init__()
        self.pre_conv = conv_block_k3(1, 32)
        self.B1 = attention_block(32, 32, 64)
        self.B2 = attention_block(64, 64, 128)
        self.B3 = attention_block(128, 256, 256)
        self.B4 = attention_block(256 + 128, 128, 64)
        self.B5 = attention_block(64 + 64, 64, 32)
        self.fore = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())
        self.back = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

        self.drop_1 = nn.Dropout(p=0.3)
        self.drop_2 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pre_conv(x)
        att1 = self.B1(x)
        x = F.max_pool2d(att1, kernel_size=4)
        att2 = self.B2(x)
        x = F.max_pool2d(att2, kernel_size=4)
        x = self.drop_1(x)
        x = self.B3(x)
        x = self.drop_2(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.B4(torch.cat([x, att2], dim=1))
        x = F.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.B5(torch.cat([x, att1], dim=1))
        fore = self.fore(x)
        back = self.back(1 - x)
        return fore, back