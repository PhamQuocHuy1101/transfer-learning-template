import torch
import torch.nn as nn
from .layer import Conv2d


class Reduction_A(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.b1 = nn.MaxPool2d(3, 2, padding = 0)
        self.b2 = Conv2d(in_channel, 384, 3, 2, padding='valid')
        self.b3 = nn.Sequential(
            Conv2d(in_channel, 256, 1, 1, padding = 'same'),
            Conv2d(256, 256, 3, 1, padding = 'same'),
            Conv2d(256, 384, 3, 2, padding = 'valid'),
        )
    def forward(self, X):
        b1 = self.b1(X)
        b2 = self.b2(X)
        b3 = self.b3(X)
        return torch.cat([b1, b2, b3], dim = 1)


class Reduction_B(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.b1 = nn.MaxPool2d(3, 2, padding = 0)
        self.b2 = nn.Sequential(
            Conv2d(in_channel, 256, 1, 1, padding='same'),
            Conv2d(256, 384, 3, 2, padding='valid')
        )
        self.b3 = nn.Sequential(
            Conv2d(in_channel, 256, 1, 1, padding='same'),
            Conv2d(256, 288, 3, 2, padding='valid')
        )
        self.b4 = nn.Sequential(
            Conv2d(in_channel, 256, 1, 1, padding = 'same'),
            Conv2d(256, 288, 3, 1, padding = 'same'),
            Conv2d(288, 320, 3, 2, padding = 'valid')
        )
    def forward(self, X):
        b1 = self.b1(X)
        b2 = self.b2(X)
        b3 = self.b3(X)
        b4 = self.b4(X)
        
        return torch.cat([b1, b2, b3, b4], dim = 1)