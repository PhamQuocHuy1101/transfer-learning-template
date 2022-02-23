from abc import abstractmethod
import torch.nn as nn
from torchvision import models

BACKBONE_MAPPING = {
    'b0': {
        'm': models.efficientnet_b0,
        'last_layer': 'classifier',
        'in_features': 1280
    },
    'b1': {
        'm': models.efficientnet_b0,
        'last_layer': 'classifier',
        'in_features': 1280
    },
    'resnet18': {
        'm': models.resnet18,
        'last_layer': 'fc',
        'in_features': 512
    },
    'resnet50': {
        'm': models.resnet50,
        'last_layer': 'fc',
        'in_features': 2048
    }
}


class Backbone(nn.Module):
    def __init__(self, backbone = 'b0'):
        super(Backbone, self).__init__()
        self.bb_info = BACKBONE_MAPPING.get(backbone)
        self.backbone = self.bb_info['m'](pretrained=True)
    
    @abstractmethod
    def __set_params(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass
