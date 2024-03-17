import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

pading_size = 2
pading_size2 = 4

class conv_block_k2(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=2, bias=True, padding_layer=nn.ZeroPad2d, dilation=1):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.conv = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_ch, out_ch, kernel_size, bias=bias, dilation=dilation)
        )

    def forward(self, x):
        return self.conv(x)

def local_reshape(x, shape):
    xf_0 = x[:,0::pading_size,:,:]
    xf_1 = torch.reshape(xf_0,[shape[0],shape[1],int(shape[2]/2),shape[3]])
    xb_0 = x[:,1::pading_size,:,:]
    xb_1 = torch.reshape(xb_0,[shape[0],shape[1],int(shape[2]/2),shape[3]])
    xl = torch.cat([xf_1, xb_1], dim=2)
    return xl

def round_reshape(a, shape, xlocal, ylocal):
    xfront = xlocal*pading_size+pading_size
    yfront = ylocal*pading_size+pading_size
    ashape = a.size()
    xfinal = ashape[2]+xfront
    yfinal = ashape[1]+yfront
    a = a.permute(0, 3, 1, 2)
    ap = F.pad(a, [pading_size, pading_size, pading_size, pading_size], mode='reflect')
    ap = ap.permute(0, 2, 3, 1)
    yf = ap[:,yfront:yfinal:pading_size,:,:]
    xf = yf[:,:,xfront:xfinal,:]
    af = torch.reshape(xf,[shape[0],shape[1],int(shape[2]/2),shape[3]])
    yb = ap[:,(yfront+1):yfinal:pading_size,:,:]
    xb = yb[:,:,xfront:xfinal,:]
    ab = torch.reshape(xb,[shape[0],shape[1],int(shape[2]/2),shape[3]])
    afinal = torch.cat([af, ab], 2)
    return afinal

def dcp_function(x, w):
    x = x.permute(0, 2, 3, 1)
    w = w.permute(0, 2, 3, 1)
    width = int(x.shape[1] * x.shape[2] / pading_size2)
    channel = int(x.shape[3])
    final_w = int(x.shape[1]/pading_size)
    final_h = int(x.shape[2]/pading_size)
    x2 = local_reshape(x, [-1, width, pading_size2, channel])
    w2 = local_reshape(w, [-1, width, pading_size2, 1])
    w_s0 = round_reshape(w, [-1, width, pading_size2, 1], xlocal=-1, ylocal=-1)
    w_s1 = round_reshape(w, [-1, width, pading_size2, 1], xlocal=0, ylocal=-1)
    w_s2 = round_reshape(w, [-1, width, pading_size2, 1], xlocal=1, ylocal=-1)
    w_s3 = round_reshape(w, [-1, width, pading_size2, 1], xlocal=-1, ylocal=0)
    w_s4 = round_reshape(w, [-1, width, pading_size2, 1], xlocal=1, ylocal=0)
    w_s5 = round_reshape(w, [-1, width, pading_size2, 1], xlocal=-1, ylocal=1)
    w_s6 = round_reshape(w, [-1, width, pading_size2, 1], xlocal=0, ylocal=1)
    w_s7 = round_reshape(w, [-1, width, pading_size2, 1], xlocal=1, ylocal=1)
    one_mul = x2 * w2
    one_out = torch.sum(one_mul,dim=2)
    one_mul_s0 = x2 * w_s0
    one_out_s0 = torch.sum(one_mul_s0, dim=2)
    one_mul_s1 = x2 * w_s1
    one_out_s1 = torch.sum(one_mul_s1, dim=2)
    one_mul_s2 = x2 * w_s2
    one_out_s2 = torch.sum(one_mul_s2, dim=2)
    one_mul_s3 = x2 * w_s3
    one_out_s3 = torch.sum(one_mul_s3, dim=2)
    one_mul_s4 = x2 * w_s4
    one_out_s4 = torch.sum(one_mul_s4, dim=2)
    one_mul_s5 = x2 * w_s5
    one_out_s5 = torch.sum(one_mul_s5, dim=2)
    one_mul_s6 = x2 * w_s6
    one_out_s6 = torch.sum(one_mul_s6, dim=2)
    one_mul_s7 = x2 * w_s7
    one_out_s7 = torch.sum(one_mul_s7, dim=2)
    one_put = (one_out + one_out_s0 + one_out_s1 + one_out_s2 + one_out_s3 + one_out_s4 + one_out_s5 + one_out_s6 + one_out_s7)/9
    one_put = torch.reshape(one_put, [-1, final_w, final_h, channel])
    one_put = one_put.permute(0, 3, 1, 2)
    return one_put

class DCP(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w = conv_block_k2(ch, 1, 2)

    def forward(self, x):
        w = self.w(x)
        pool = dcp_function(x, w)
        return pool