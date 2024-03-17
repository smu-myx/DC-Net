import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.block_others import conv_block_k1, conv_block_k3

class dilated_conv_block(nn.Module):

    def __init__(self, ch, dilated_rate=(1,3,5)):
        super().__init__()
        self.dconv0 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, dilation=dilated_rate[0], stride=1, padding=dilated_rate[0]),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
        self.dconv1 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, dilation=dilated_rate[1], stride=1, padding=dilated_rate[1]),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, dilation=dilated_rate[2], stride=1, padding=dilated_rate[2]),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.dconv0(x), self.dconv1(x), self.dconv2(x)

class DaR_simple(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.C = 1e+6
        self.fore = nn.Sequential(
            conv_block_k1(ch, int(ch / 2)),
            conv_block_k3(int(ch / 2), int(ch / 2))
        )
        self.back = nn.Sequential(
            conv_block_k1(ch, int(ch / 2)),
            conv_block_k3(int(ch / 2), int(ch / 2))
        )

    def forward(self, fore, back):
        fore = self.fore(fore)
        back = self.back(back)
        fore = torch.clamp(fore, min=0, max=self.C)
        back = torch.clamp(back, min=0, max=self.C)
        return torch.cat([fore, 1 - back], dim=1)

class DaR(nn.Module):

    def __init__(self, ch, theta=0.2):
        super().__init__()
        self.theta = theta

        self.pre_fore = nn.Sequential(
            conv_block_k1(ch, int(ch / 2)),
            conv_block_k3(int(ch / 2), int(ch / 2))
        )
        self.pre_back = conv_block_k1(ch, int(ch / 2))
        self.post_back = conv_block_k3(int(ch / 2), int(ch / 2))

        self.key_dconv = dilated_conv_block(int(ch / 2))
        self.key_conv = conv_block_k1(int(3 * ch / 2), int(ch / 2))

        self.value_dconv = dilated_conv_block(int(ch / 2))
        self.value_conv = conv_block_k1(int(3 * ch / 2), int(ch / 2))

        self.out_conv = conv_block_k1(int(ch / 2), ch)

    def forward(self, fore, back, config=None):
        fore = self.pre_fore(fore)
        back = self.pre_back(back)
        post_back = self.post_back(back)

        key = self.key_dconv(back)
        key = self.GAP_block(key)
        key = self.key_conv(key).squeeze(3)

        value = self.value_dconv(back)
        value = self.GAP_block(value)
        value = self.value_conv(value).squeeze(3).permute(0, 2, 1)

        query = fore.clone().flatten(2).permute(0, 2, 1)

        weight = (query @ key)
        weight = torch.sigmoid(weight)
        out = weight @ value
        out = out.permute(0, 2, 1).view(back.shape)

        out = self.out_conv(out)
        out = out * self.theta + torch.cat([fore, post_back], dim=1)

        if config:
            print(weight)
            print(config)
        return out

    def GAP_block(self, x_list):
        x0 = F.adaptive_avg_pool2d(x_list[0], 1)
        x1 = F.adaptive_avg_pool2d(x_list[1], 1)
        x2 = F.adaptive_avg_pool2d(x_list[2], 1)
        return torch.cat([x0, x1, x2], dim=1)
