import torch
import torch.nn as nn
from torchsummary import summary

from models.subnet_Coupling import Coupling
from models.subnet_Decomposition import Decomposition


class DC_Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.Decomposition = Decomposition()
        self.Coupling = Coupling()
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, config=None):
        fore, back = self.Decomposition(x)
        final = self.Coupling(x, fore, back, config)
        return final, fore, back
