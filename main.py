import os

import torch
from models.dc_net import DC_Net

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DC_Net().to(my_device)
    img = torch.randn(1, 256, 256, 1).to(my_device)
    print(model(img))