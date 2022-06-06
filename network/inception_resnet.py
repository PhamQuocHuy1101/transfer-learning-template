from re import A
import torch
import torch.nn as nn

from .common import *

class Inception_resnet(nn.Module):
    '''
        Inception Resnet V2
    '''
    def __init__(self, in_channel, n_class, block_A, block_B, block_C, drop_r = 0.2):
        '''
            block_A, B, C: 
                - n_block: # blocks
                - args: list of params
        '''
        super().__init__()

        self.stem = Stem(in_channel) # 299, in_channel -> 35, 384
        self.inception_A = nn.Sequential(*[Inception_A(*block_A.args) for _ in range(block_A.n_block)]) # 35, 384 -> 35, 384
        self.reduce_A = Reduction_A(384) # 35, 384 -> 17, 1152
        self.inception_B = nn.Sequential(*[Inception_B(*block_B.args) for _ in range(block_B.n_block)]) # 17, 1152 -> 17, 1152
        self.reduce_B = Reduction_B(1152) # 17, 1152 -> 8, 2144
        self.inception_C = nn.Sequential(*[Inception_C(*block_C.args) for _ in range(block_C.n_block)]) # 8, 2144 -> 8, 2144
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