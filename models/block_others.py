import torch
import torch.nn as nn

from torchsummary import summary

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_block_k1(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(inplace=True))

def conv_block_k3(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(inplace=True))

class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block1 = conv_block_k3(in_ch, out_ch)
        self.block2 = conv_block_k3(out_ch, out_ch)

    def forward(self, x):
        return self.block2(self.block1(x))

class attention_block(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = conv_block_k1(in_ch, mid_ch)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            nn.Linear(mid_ch, mid_ch),
            nn.Softmax(dim=1))
        self.conv_out = conv_block_k3(mid_ch, out_ch)

    def forward(self, x):
        x = self.conv_in(x)
        weight = self.weight(self.gap(x).squeeze(2).squeeze(2))
        x = x * weight.unsqueeze(2).unsqueeze(2)
        x = self.conv_out(x)
        return x

