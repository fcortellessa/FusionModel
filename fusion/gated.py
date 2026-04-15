import torch
import torch.nn as nn
from .base import BaseFusion

from . import register_fusion

@register_fusion("simple_gated")
class SimpleGatedFusion(BaseFusion):
    def __init__(self, dim, num_streams):
        super().__init__(dim, num_streams)
        self.gate = nn.Sequential(
            nn.Linear(dim * num_streams, num_streams),
            nn.Softmax(dim=1)
        )

    def forward(self, features_list):
        combined = torch.cat(features_list, dim=1)
        weights = self.gate(combined)  # (B, num_streams)
        weights = weights.unsqueeze(-1)  # (B, num_streams, 1)
        stacked = torch.stack(features_list, dim=1)
        print(f"SimpleGatedFusion Weights: {weights.squeeze()}")  # Debug print
        return (stacked * weights).sum(dim=1)



@register_fusion("gated")
class GatedFusion(BaseFusion):
    def __init__(self, dim, num_streams, temperature=100.0):
        super().__init__(dim, num_streams)
        self.temperature = temperature
        self.gate = nn.Sequential(
            nn.Linear(dim * num_streams, 128), # Add a hidden layer for capacity
            nn.ReLU(),
            nn.Linear(128, num_streams)
        )

    def forward(self, features_list):
        # 1. Ensure features are normalized so one doesn't dominate by scale
        # stacked: (B, num_streams, dim)
        stacked = torch.stack(features_list, dim=1) 
        
        # 2. Calculate logits
        combined = torch.cat(features_list, dim=1)
        logits = self.gate(combined) # (B, num_streams)
        
        # 3. Apply Temperature-scaled Softmax
        weights = torch.softmax(logits / self.temperature, dim=1)
        
        # 4. Apply weighted sum
        out = (stacked * weights.unsqueeze(-1)).sum(dim=1)

        # print(f"GatedFusion Logits: {logits}, weights: {weights.squeeze()}")  # Debug print
        return out, weights#.detach().mean(dim=0) # return weights for logging