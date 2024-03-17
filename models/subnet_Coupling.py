import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.block_CCF import CCF
from models.block_DCP import DCP
from models.block_DaR import DaR, DaR_simple
from models.block_others import conv_block_k3


class Coupling(nn.Module):

    def __init__(self):
        super().__init__()
        self.back_conv1 = conv_block_k3(2, 32)
        self.back_conv2 = conv_block_k3(32, 64)
        self.back_conv3 = conv_block_k3(64, 128)
        self.back_conv4 = conv_block_k3(128, 256)
        self.back_conv5 = conv_block_k3(256, 512)
        self.back_conv6 = conv_block_k3(512 + 256, 256)
        self.back_conv7 = conv_block_k3(256 + 128, 128)
        self.back_conv8 = conv_block_k3(128 + 64, 64)
        self.back_conv9 = conv_block_k3(64 + 32, 32)

        self.fore_conv1 = conv_block_k3(2, 32)
        self.fore_conv2 = conv_block_k3(32, 64)
        self.fore_conv3 = conv_block_k3(64, 128)
        self.fore_conv4 = conv_block_k3(128, 256)
        self.fore_conv5 = conv_block_k3(256, 512)
        self.fore_conv6 = conv_block_k3(512, 256)
        self.fore_conv7 = conv_block_k3(256, 128)
        self.fore_conv8 = conv_block_k3(128, 64)
        self.fore_conv9 = conv_block_k3(64, 32)

        self.fore_final = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

        self.dcp1 = DCP(32)
        self.dcp2 = DCP(64)
        self.dcp3 = DCP(128)
        self.dcp4 = DCP(256)

        self.ccf1 = CCF(256)
        self.ccf2 = CCF(128)
        self.ccf3 = CCF(64)
        self.ccf4 = CCF(32)

        self.dar1 = DaR_simple(32)
        self.dar2 = DaR_simple(64)
        self.dar3 = DaR_simple(128)
        self.dar4 = DaR_simple(256)
        self.dar5 = DaR(512)
        self.dar6 = DaR_simple(256)
        self.dar7 = DaR_simple(128)
        self.dar8 = DaR_simple(64)
        self.dar9 = DaR_simple(32)

        self.drop_b1 = nn.Dropout(p=0.3)
        self.drop_b2 = nn.Dropout(p=0.3)
        self.drop_f1 = nn.Dropout(p=0.3)
        self.drop_f2 = nn.Dropout(p=0.3)

    def forward(self, img, fore, back, config=None):
        # back
        back_input = torch.cat([img, back], dim=1)
        back_conv1 = self.back_conv1(back_input)
        back_pool1 = F.max_pool2d(back_conv1, kernel_size=2)
        back_conv2 = self.back_conv2(back_pool1)
        back_pool2 = F.max_pool2d(back_conv2, kernel_size=2)
        back_conv3 = self.back_conv3(back_pool2)
        back_pool3 = F.max_pool2d(back_conv3, kernel_size=2)
        back_conv4 = self.back_conv4(back_pool3)
        back_pool4 = F.max_pool2d(back_conv4, kernel_size=2)
        drop = self.drop_b1(back_pool4)
        back_conv5 = self.back_conv5(drop)
        drop = self.drop_b2(back_conv5)
        back_up6 = F.interpolate(drop, scale_factor=2, mode='bilinear')
        back_conv6 = self.back_conv6(torch.cat([back_up6, back_conv4], dim=1))
        back_up7 = F.interpolate(back_conv6, scale_factor=2, mode='bilinear')
        back_conv7 = self.back_conv7(torch.cat([back_up7, back_conv3], dim=1))
        back_up8 = F.interpolate(back_conv7, scale_factor=2, mode='bilinear')
        back_conv8 = self.back_conv8(torch.cat([back_up8, back_conv2], dim=1))
        back_up9 = F.interpolate(back_conv8, scale_factor=2, mode='bilinear')
        back_conv9 = self.back_conv9(torch.cat([back_up9, back_conv1], dim=1))
        # fore
        fore_input = torch.cat([img, fore], dim=1)

        fore_conv1 = self.fore_conv1(fore_input)
        fore_pool1 = self.dcp1(fore_conv1)
        enhance = self.dar1(fore_pool1, back_pool1)
        fore_conv2 = self.fore_conv2(enhance)
        fore_pool2 = self.dcp2(fore_conv2)
        enhance = self.dar2(fore_pool2, back_pool2)
        fore_conv3 = self.fore_conv3(enhance)
        fore_pool3 = self.dcp3(fore_conv3)
        enhance = self.dar3(fore_pool3, back_pool3)
        fore_conv4 = self.fore_conv4(enhance)
        fore_pool4 = self.dcp4(fore_conv4)
        enhance = self.dar4(fore_pool4, back_pool4)
        drop = self.drop_f1(enhance)
        fore_conv5 = self.fore_conv5(drop)
        enhance = self.dar5(fore_conv5, back_conv5, config)
        drop = self.drop_f2(enhance)
        fore_up6 = F.interpolate(drop, scale_factor=2, mode='bilinear')
        fore_conv6 = self.fore_conv6(fore_up6)
        ccf1 = self.ccf1(fore_conv6, fore_conv4)
        enhance = self.dar6(ccf1, back_conv6)
        fore_up7 = F.interpolate(enhance, scale_factor=2, mode='bilinear')
        fore_conv7 = self.fore_conv7(fore_up7)
        ccf2 = self.ccf2(fore_conv7, fore_conv3)
        enhance = self.dar7(ccf2, back_conv7)
        fore_up8 = F.interpolate(enhance, scale_factor=2, mode='bilinear')
        fore_conv8 = self.fore_conv8(fore_up8)
        ccf3 = self.ccf3(fore_conv8, fore_conv2)
        enhance = self.dar8(ccf3, back_conv8)
        fore_up9 = F.interpolate(enhance, scale_factor=2, mode='bilinear')
        fore_conv9 = self.fore_conv9(fore_up9)
        ccf4 = self.ccf4(fore_conv9, fore_conv1)
        enhance = self.dar9(ccf4, back_conv9)
        final = self.fore_final(enhance)
        return final
