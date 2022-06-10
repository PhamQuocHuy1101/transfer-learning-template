import torch
import torch.nn as nn
from .common import PatchEmbedding, PatchMerging, IFormerBlock

def build_iform_block(n_layer, n_channel, n_hidden, rCh, rCl, n_smoothing, smoothing, args):
    rCh, rCl = eval(rCh), eval(rCl)
    ratio = [(n_channel * rCh, n_channel * rCl) for i in range(n_layer)]
    blocks = []
    for i in range(n_layer):
        rh, rl = rCh, rCl
        if i > 0 and i % smoothing == 0:
            k = i // smoothing
            rh = rCh - k/n_smoothing*rCh
            rl = rCl + k/n_smoothing*rCl

        h_channel = int(n_channel * rh)
        l_channel = int(n_channel * rl)
        args.mixer_args.C_h = h_channel
        args.mixer_args.C_l = l_channel

        blocks.append(IFormerBlock(n_channel = n_channel, n_hidden = n_hidden, **args))
    
    iforms = nn.Sequential(*blocks)
    return iforms

class InceptionTransformer(nn.Module):
    def __init__(self, iform_blocks, patch_embeddings):
        super(InceptionTransformer, self).__init__()
        assert len(iform_blocks) == len(patch_embeddings), '#block not match'
        self.n_stage = len(iform_blocks)

        self.stages = []
        for iform, patch_emb in zip(iform_blocks, patch_embeddings):
            patch_layer = PatchEmbedding(norm_layer = nn.LayerNorm, **patch_emb)
            iforms = build_iform_block(iform.n_layer, iform.n_channel, iform.n_hidden, iform.rCh, iform.rCl, iform.n_smoothing, iform.smoothing, iform.block_args)
            self.stages.append((patch_layer, iforms))
    
    def forward(self, X):
        pass