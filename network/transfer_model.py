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

class TransferModel(nn.Module):
    def __init__(self, n_class, backbone = 'b0', drop = 0.2):
        super(TransferModel, self).__init__()
        net = BACKBONE_MAPPING.get(backbone)
        self.net = net['m'](pretrained=True)
        last_layer = net['last_layer']
        in_features = net['in_features']
        setattr(self.net, last_layer, nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(in_features, n_class)
            ))

        self.backbone_params = [p for n, p in self.net.named_parameters() if last_layer not in n]
        self.head_params = [p for n, p in self.net.named_parameters() if last_layer in n]
    def forward(self, X):
        return self.net(X)
