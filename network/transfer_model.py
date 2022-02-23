import torch.nn as nn
from torchvision import models
from .common import Backbone

class TransferModel(Backbone):
    def __init__(self, n_class, backbone = 'b0', drop = 0.2):
        super(TransferModel, self).__init__(backbone)
        setattr(self.backbone, self.bb_info['last_layer'], nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(self.bb_info['in_features'], n_class)
            ))
        self.__set_params()

    def __set_params(self):
        self.backbone_params = [p for n, p in self.backbone.named_parameters() if self.bb_info['last_layer'] not in n]
        self.head_params = [p for n, p in self.backbone.named_parameters() if self.bb_info['last_layer'] in n]

    def forward(self, X):
        return self.backbone(X)(X)
