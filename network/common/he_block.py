import torch
import torch.nn as nn


class HEBlock(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    
    def forward(self, X):
        '''
            X: (batch_size, channel, w, h)
        '''
        shape = X.shape
        max_value, _ = torch.max(X.view(*shape[:2], -1), dim = -1)
        mask = X == max_value[:, :, None, None]
        X[mask] *= self.beta
        return X
