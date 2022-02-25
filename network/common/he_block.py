import torch
import torch.nn as nn


class HEMaxBlock(nn.Module):
    def __init__(self, beta):
        super(HEMaxBlock, self).__init__()
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

class HESigmoidBlock(nn.Module):
    def __init__(self, threshold, beta):
        super(HESigmoidBlock, self).__init__()
        self.threshold = threshold
        self.beta = beta

    def forward(self, X):
        '''
            X: (batch_size, channel, w, h)
        '''
        shape = X.shape
        act_value = torch.sigmoid(X)
        mask = act_value > self.threshold
        X[mask] *= self.beta
        return X
