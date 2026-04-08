import torch
from .base import BaseFusion
from . import register_fusion

@register_fusion("concat")
class ConcatFusion(BaseFusion):
    def forward(self, features_list):
        return torch.cat(features_list, dim=1)