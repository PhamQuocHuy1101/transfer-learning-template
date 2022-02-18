import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding = 'valid', act = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
            nn.BatchNorm2d(out_channel)
        )
        self.act = nn.ReLU() if act == True else None
    def forward(self, X):
        X = self.conv(X)
        if self.act != None:
            X = self.act(X)
        return X