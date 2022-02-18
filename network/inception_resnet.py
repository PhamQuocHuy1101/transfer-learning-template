from re import A
import torch
import torch.nn as nn

from .inception_resnet_common import *

class Inception_resnet(nn.Module):
    '''
        Inception Resnet V2
    '''
    def __init__(self, in_channel, n_class, n_block_A, n_block_B, n_block_C, drop_r = 0.2):
        super().__init__()

        self.stem = Stem(in_channel) # 299, in_channel -> 35, 384
        self.inception_A = nn.Sequential(*[Inception_A(384, 0.1) for _ in range(n_block_A)]) # 35, 384 -> 35, 384
        self.reduce_A = Reduction_A(384) # 35, 384 -> 17, 1152
        self.inception_B = nn.Sequential(*[Inception_B(1152, 0.2) for _ in range(n_block_B)]) # 17, 1152 -> 17, 1152
        self.reduce_B = Reduction_B(1152) # 17, 1152 -> 8, 2144
        self.inception_C = nn.Sequential(*[Inception_C(2144, 0.3) for _ in range(n_block_C)]) # 8, 2144 -> 8, 2144
        self.pool = nn.AvgPool2d(8) # 2144
        self.drop = nn.Dropout(drop_r)
        self.classifier = nn.Linear(2144, n_class)

        self.backbone_params = [p for n, p in self.named_parameters() if 'classifier' not in n]
        self.head_params = [p for n, p in self.named_parameters() if 'classifier' in n]
    
    def forward(self, X):
        out = self.stem(X)
        out = self.inception_A(out)
        out = self.reduce_A(out)
        out = self.inception_B(out)
        out = self.reduce_B(out)
        out = self.inception_C(out)
        out = self.pool(out)
        out = out.view(out.shape[0], -1)
        out = self.drop(out)
        out = self.classifier(out)
        return out