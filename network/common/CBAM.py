import torch
import torch.nn as nn


class CAM(nn.Module):
    '''
        Channel attention module
    '''
    def __init__(self, in_channel, n_hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, in_channel)
        )
        self.act = nn.Sigmoid()

    def forward(self, X):
        '''
            X.shape = (batch, channel, w, h)
            out.shape = (batch, channel)
        '''
        shape = X.shape
        X = X.reshape(*shape[:2], -1).contiguous()
        max_f, _ = torch.max(X, dim = -1)
        avg_f = torch.mean(X, dim = -1)
        out = self.mlp(torch.cat([max_f, avg_f], dim = 0))
        out = self.act(out[:shape[0]] + out[shape[0]:])
        return out.reshape(*shape[:2], 1, 1).contiguous()

class SAM(nn.Module):
    '''
        Spatial attention module
    '''
    def __init__(self, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, stride, padding)
        self.act = nn.Sigmoid()
    
    def forward(self, X):
        '''
            X.shape = (batch, channel, w, h)
            out.shape = (batch, 1, w, h)
        '''
        max_f, _ = torch.max(X, dim = 1)
        avg_f = torch.mean(X, dim = 1)
        out = self.conv(torch.stack([max_f, avg_f], dim = 1))
        return self.act(out)

class CBAM(nn.Module):
    def __init__(self, in_channel, n_hidden, kernel, stride, padding):
        super().__init__()
        self.cam = CAM(in_channel, n_hidden)
        self.sam = SAM(kernel, stride, padding)
    
    def forward(self, X):
        cam = self.cam(X)
        out = torch.mul(X, cam)
        sam = self.sam(out)
        out = torch.mul(out, sam)
        return torch.add(X, out)
