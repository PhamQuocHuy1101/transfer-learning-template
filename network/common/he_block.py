import torch
import torch.nn as nn


class HEBlock(nn.Module):
    def __init__(self, n_features, n_class, beta):
        super().__init__()
        self.reduce_channel = nn.Conv2d(n_features, n_class, 1, 1)
        self.beta = beta
    
    def forward(self, X):
        '''
            X: (batch_size, channel, w, h)
        '''
        feature_map = self.reduce_channel(X)
        shape = feature_map.shape
        max_value, _ = torch.max(feature_map.view(*shape[:2], -1), dim = -1)
        mask = feature_map == max_value[:, :, None, None]
        feature_map[mask] *= self.beta
        return feature_map
