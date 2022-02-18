import torch
import torch.nn as nn
from .common import Conv2d

class Inception_A(nn.Module):
    def __init__(self, in_channel, scale = 1.0):
        super().__init__()
        self.scale = scale
        self.activate = nn.ReLU()
        self.b1 = Conv2d(in_channel, 32, 1, 1, padding='same')
        self.b2 = nn.Sequential(
            Conv2d(in_channel, 32, 1, 1, padding='same'),
            Conv2d(32, 32, 3, 1, padding='same')
        )
        self.b3 = nn.Sequential(
            Conv2d(in_channel, 32, 1, 1, padding='same'),
            Conv2d(32, 48, 3, 1, padding='same'),
            Conv2d(48, 64, 3, 1, padding='same')
        )
        self.concat_conv = Conv2d(128, 384, 1, 1, padding='same', act=False)

    def forward(self, X):
        # X = self.activate(X)
        b1 = self.b1(X)
        b2 = self.b2(X)
        b3 = self.b3(X)
        concat = self.concat_conv(torch.cat([b1, b2, b3], dim = 1))
        return self.activate(X + concat * self.scale)


class Inception_B(nn.Module):
    def __init__(self, in_channel, scale = 1.0):
        super().__init__()
        self.scale = scale
        self.activate = nn.ReLU()
        self.b1 = Conv2d(in_channel, 192, 1, 1, padding='same')
        self.b2 = nn.Sequential(
            Conv2d(in_channel, 128, 1, 1, padding='same'),
            Conv2d(128, 160, (1, 7), 1, padding='same'),
            Conv2d(160, 192, (7, 1), 1, padding='same')
        )
        self.concat_conv = Conv2d(384, 1152, 1, 1, padding='same', act = False)

    def forward(self, X):
        b1 = self.b1(X)
        b2 = self.b2(X)
        concat = self.concat_conv(torch.cat([b1, b2], dim = 1))
        return self.activate(X + concat * self.scale)


class Inception_C(nn.Module):
    def __init__(self, in_channel, scale = 1.0):
        super().__init__()
        self.scale = scale
        self.activate = nn.ReLU()
        self.b1 = Conv2d(in_channel, 192, 1, 1, padding='same')
        self.b2 = nn.Sequential(
            Conv2d(in_channel, 192, 1, 1, padding='same'),
            Conv2d(192, 224, (1, 3), 1, padding='same'),
            Conv2d(224, 256, (3, 1), 1, padding='same')
        )
        self.concat_conv = Conv2d(448, 2144, 1, 1, padding='same', act = False)

    def forward(self, X):
        b1 = self.b1(X)
        b2 = self.b2(X)
        concat = self.concat_conv(torch.cat([b1, b2], dim = 1))
        return self.activate(X + concat * self.scale)