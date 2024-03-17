import torch
import torch.nn as nn
from torchsummary import summary

from models.block_others import conv_block_k3

class CCF(nn.Module):

    def __init__(self, ch1, ch2=None):
        super().__init__()
        if not ch2:
            ch2 = ch1
        self.en_conv = nn.Conv1d(in_channels=ch1, out_channels=ch1, kernel_size=ch2)
        self.de_conv = nn.Conv1d(in_channels=ch2, out_channels=ch2, kernel_size=ch1)
        self.final_conv = conv_block_k3(ch1 + ch2, ch1)

    def forward(self, en_feat, de_feat):
        weight = de_feat.flatten(2) @ en_feat.flatten(2).permute(0, 2, 1)
        en_w = self.en_conv(weight.permute(0, 2, 1)).unsqueeze(-1)
        de_w = self.de_conv(weight).unsqueeze(-1)
        return self.final_conv(torch.cat([en_w * en_feat, de_w * de_feat], dim=1))
