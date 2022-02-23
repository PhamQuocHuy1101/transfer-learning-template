from numpy import block
import torch
import torch.nn as nn
from torchvision import models
from common import HEBlock

class HENet(nn.Module):
    def __init__(self, n_class, beta, block_expansion = 1):
        super().__init__()
        n_features = 512
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.HE_block = HEBlock(n_features, n_class * block_expansion, beta)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_class * block_expansion, n_class)
        
    def forward(self, X):
        '''
            X: (batch_size, channel, w, h)
        '''
        feature_map = self.backbone(X)
        feature_map = self.HE_block(feature_map)
        out = self.pool(feature_map)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out