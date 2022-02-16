import torch.nn as nn
from torchvision import models

BACKBONE_MAPPING = {
    'b0': models.efficientnet_b0,
    'b1': models.efficientnet_b1,
    'b2': models.efficientnet_b2,
    'b3': models.efficientnet_b3,
    'b4': models.efficientnet_b4,
    'b5': models.efficientnet_b5,
    'b6': models.efficientnet_b6,
    'b7': models.efficientnet_b7,
    'resnet18': models.resnet18,
}

class Model(nn.Module):
    def __init__(self, n_class, backbone = 'b0', drop = 0.2):
        super(Model, self).__init__()

        self.net = BACKBONE_MAPPING.get(backbone)(pretrained=True)
        
        if 'resnet' in backbone:
            in_features = self.net.fc.in_features
            self.net.fc = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(in_features, n_class)
            )
        if 'b' in backbone:
            in_features = self.net.classifier[1].in_features
            self.net.classifier = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(in_features, n_class)
            )
        
    def forward(self, X):
        return self.net(X)
