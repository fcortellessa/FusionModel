import torch
import torch.nn as nn
from .base import BaseFusion

class AttentionFusion(BaseFusion):
    def __init__(self, dim, num_streams):
        super().__init__(dim, num_streams)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

    def forward(self, features_list):
        x = torch.stack(features_list, dim=1)
        out, _ = self.attn(x, x, x)
        return out.mean(dim=1)