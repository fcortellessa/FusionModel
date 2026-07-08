import torch
import torch.nn as nn

from .base import BaseFusion
from . import register_fusion


@register_fusion("spatial_gated")
class SpatialGatedFusion2D(BaseFusion):
    """
    Fuses two spatial streams by generating pixel-wise (spatial) gating weights.
    """
    def __init__(self, dim, num_streams=2): # dim represents here channels per stream
        super().__init__(dim, num_streams)
        # Takes concatenated streams and outputs a gating weight map for each stream
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * num_streams, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, num_streams, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=1) # Softmax across streams per pixel

    def forward(self, stream_features):
        # stream_features: List of Tensors, each [B, C, H, W]
        x_concat = torch.cat(stream_features, dim=1) # [B, C * num_streams, H, W]
        
        # Predict spatial attention weights
        gate_weights = self.gate_conv(x_concat) # [B, num_streams, H, W]
        gate_weights = self.softmax(gate_weights) # Normalized per pixel across modalities
        
        # Apply spatial gating element-wise
        fused = 0
        for i, feat in enumerate(stream_features):
            spatial_gate = gate_weights[:, i:i+1, :, :] # Extract [B, 1, H, W] map
            fused += spatial_gate * feat
            
        return fused, gate_weights