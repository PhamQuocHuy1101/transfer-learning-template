import torch
import torch.nn as nn
import torch.nn.functional as F

class HighFreqProjection(nn.Module):
    def __init__(self, n_channel, n_hidden, max_pooling_size = 3, max_pooling_stride = 1, dw_kernel = 3, dw_stride = 1, activate = True):
        super(HighFreqProjection, self).__init__()
        self.h1 = n_channel // 2
        self.h2 = n_channel - self.h1

        self.proj_pooling = nn.Sequential(
            nn.MaxPool2d(max_pooling_size, max_pooling_stride),
            nn.Linear(n_hidden, n_hidden)
        )

        self.proj_depthwise = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Conv2d(self.h2, self.h2, kernel_size = dw_kernel, stride = dw_stride, groups=self.h2)
        )
        self.activate = F.relu if activate else nn.Identity()
    
    def forward(self, X):
        '''
            X: batch, C, E
        '''
        X1 = X[:, :self.h1, :]
        X2 = X[:, sefl.h1:, :]
        out_1 = self.proj_pooling(X1)
        out_2 = self.proj_depthwise(X2)
        return self.activate(torch.cat((out_1, out_2), dim = 1))

class LowFreqProjection(nn.Module):
    def __init__(self, n_hidden, avg_pooling_size = 2, avg_pooling_stride = 2, n_head = 4, drop = 0.1):
        super(LowFreqProjection, self).__init__()
        
        self.proj = nn.Sequential(
            nn.AvgPool2d(kernel_size = avg_pooling_size, stride = avg_pooling_stride),
            nn.MultiheadAttention(n_hidden, n_head, drop),#, batch_first = True),
            nn.Upsample(scale_factor = avg_pooling_stride)
        )
    def forward(self, X):
        return self.proj(X)
    

class FusionMixer(nn.Module):
    def __init__(self, n_channel, n_hidden, kernel_size = 3, stride = 1):
        super(FusionMixer, self).__init__()
        self.linear = nn.Linear(n_hidden, n_hidden)
        self.conv = nn.Conv2d(n_channel, n_channel, kernel_size = kernel_size, stride = stride, groups = n_channel)

    def forward(self, X):
        return self.linear(X + self.conv(X))

class InceptionMixer(nn.Module):
    def __init__(self, C_h, C_l, n_hidden, high_proj_args, low_proj_args, fusion_args):
        super(InceptionMixer, self).__init__()
        self.C_h = C_h
        self.C_l = C_l

        self.high_proj = HighFreqProjection(n_channel=C_h, n_hidden = n_hidden, **high_proj_args)
        self.low_proj = LowFreqProjection(n_hidden = n_hidden, **low_proj_args)
        self.fusion = FusionMixer(n_channel = C_h + C_l, n_hidden = n_hidden, **fusion_args)

    def forward(self, X):
        X_high = X[:, :self.C_h, :]
        X_low = X[:, self.C_h, :]

        out_high = self.high_proj(X_high)
        out_low = self.low_proj(X_low)

        out_combine = torch.cat((out_high, out_low), dim = 1)
        return self.fusion(otu_combine)

class IFormerBlock(nn.Module):
    def __init__(self, n_channel, n_hidden, mixer_args, activate = True):
        super(IFormerBlock, self).__init__()
        self.block_1 = nn.Sequential(
            nn.LayerNorm(n_hidden),
            InceptionMixer(n_hidden = n_hidden, **mixer_args)
        )

        self.block_2 = nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_hidden)
        )
        
        self.activate = F.relu if activate else nn.Identity()
    
    def forward(self, X):
        out = self.activate(X + self.block_1(X))
        out = self.activate(out + self.block_2)


# Swin-Transformer code

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbedding(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
