import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from convs.xception_m2tr import xception
class FeedForward1D(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(FeedForward1D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FeedForward2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeedForward2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, padding=2, dilation=2
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim,h=299,w=299, fp32fft=True):
        super().__init__()
        w_dim=w//2+1
        self.complex_weight = nn.Parameter(
            torch.randn(h, w_dim, dim, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h
        self.fp32fft = fp32fft

    def forward(self, x):
        b, _, a, b = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.fp32fft:
            dtype = x.dtype
            x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm="ortho")

        if self.fp32fft:
            x = x.to(dtype)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x



class FreqBlock(nn.Module):
    def __init__(self, dim,h=299,w=299, fp32fft=True):
        super().__init__()
        self.filter = GlobalFilter(dim, h=h, w=w, fp32fft=fp32fft)
        self.feed_forward = FeedForward2D(in_channel=dim, out_channel=dim)

    def forward(self, x):
        x = x + self.feed_forward(self.filter(x))
        return x


class m2(nn.Module):
    def __init__(self,mode='first',input_dim=3,h=299,w=299,feature_layer='b3'):
        super().__init__()
        self.mode=mode
        self.net = xception(2)
        self.feature_layer=feature_layer
        if self.mode!='first':
            with torch.no_grad():
                layers = self.net(torch.zeros(1, 3, 299, 299))
            input_dim = layers[feature_layer].shape[1]
            h=layers[feature_layer].shape[2]
            w = layers[feature_layer].shape[3]

        self.freq=FreqBlock(dim=input_dim, h=h, w=w)
    def forward(self,x):
        if self.mode=='first':
            x_freq = self.freq(x)
            layers = self.net(x_freq)

        if self.mode!='first':
            x_fea=self.net(x,self.feature_layer,'fea',0)
            x_freq=self.freq(x_fea)
            layers=self.net(x_freq,self.feature_layer,'fea',1)

        return layers



