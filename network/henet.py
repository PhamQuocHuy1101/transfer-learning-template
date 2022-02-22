import torch
import torch.nn as nn
from torchvision import models


class HENet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        n_features = 512
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.reduce_channel = nn.Conv2d(n_features, n_class, 1, 1)
        # self.mask = torch
    def forward(self, X):
        feature_map = self.reduce(self.backbone(X))
        mask = torch.