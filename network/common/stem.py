import torch
import torch.nn as nn
from .layer import Conv2d

class Stem(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        # block 1
        self.b1_conv1 = nn.Sequential(
            Conv2d(in_channel, 32, 3, 2, padding='valid'),
            Conv2d(32, 32, 3, 1, padding='valid'),
            Conv2d(32, 64, 3, 1, padding='same')
        )
        self.b1_conv2 = Conv2d(64, 96, 3, 2, padding='valid')
        self.b1_maxpool = nn.MaxPool2d(3, 2)

        # block 2
        self.b2_conv1 = nn.Sequential(
            Conv2d(160, 64, 1, 1, padding='same'),
            Conv2d(64, 64, (7, 1), 1, padding='same'),
            Conv2d(64, 64, (1, 7), 1, padding='same'),
            Conv2d(64, 96, 3, 1, padding='valid')
        )

        self.b2_conv2 = nn.Sequential(
            Conv2d(160, 64, 1, 1, padding='same'),
            Conv2d(64, 96, 3, 1, padding='valid')
        )

        # block 3
        self.b3_maxpool = nn.MaxPool2d(3, 2, padding=0)
        self.b3_conv = Conv2d(192, 192, 3, 2, padding='valid')

    def forward(self, X):
        b1_conv1 = self.b1_conv1(X)
        b1 = torch.cat([self.b1_conv2(b1_conv1), self.b1_maxpool(b1_conv1)], dim = 1)

        b2 = torch.cat([self.b2_conv1(b1), self.b2_conv2(b1)], dim = 1)

        b3 = torch.cat([self.b3_maxpool(b2), self.b3_conv(b2)], dim = 1)

        return b3
