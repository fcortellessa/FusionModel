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
    
        # Force initial weights and bias to zero, because nn.Linear creates small random values
        # nn.init.uniform_(self.gate[0].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

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
            nn.Linear(dim * num_streams, 32), # ToDo: set higher if augmented data set is used
            nn.ReLU(),
            nn.Dropout(0.3), # to prevent overfitting on small dataset
            nn.Linear(32, num_streams)
        )
        nn.init.uniform_(self.gate[3].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.gate[3].bias)

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
        return out, weights