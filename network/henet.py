from numpy import block
import torch
import torch.nn as nn
from torchvision import models
from .common import HEBlock

class HENet(nn.Module):
    '''
        https://arxiv.org/pdf/2110.10872v1.pdf
    '''
    def __init__(self, n_class, beta, block_expansion = 1):
        super().__init__()
        n_features = 512
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.reduce_channel = nn.Conv2d(n_features, n_class * block_expansion, 1, 1)
        self.HE_block = HEBlock(beta)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_class * block_expansion, n_class)

        self.backbone_params = list(self.backbone.parameters())
        self.head_params = [p for n, p in self.named_parameters() if 'backbone' not in n]
        
    def forward(self, X):
        '''
            X: (batch_size, channel, w, h)
        '''
        feature_map = self.reduce_channel(self.backbone(X))
        if self.training:
            feature_map = self.HE_block(feature_map)
        out = self.pool(feature_map)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
