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



@register_fusion("gated_Softmax")
class GatedSoftmaxFusion(BaseFusion):
    def __init__(self, dim, num_streams, temperature=50.0):
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
    

@register_fusion("gated_Sigmoid")
class GatedSigmoidFusion(BaseFusion):
    def __init__(self, dim, num_streams):
        super().__init__(dim, num_streams)
        self.gate = nn.Sequential(
            nn.Linear(dim * num_streams, 32), # ToDo: set higher if augmented data set is used
            nn.ReLU(),
            nn.Dropout(0.3), # to prevent overfitting on small dataset
            nn.Linear(32, num_streams),
            nn.Sigmoid()
        )
        nn.init.uniform_(self.gate[3].weight, -1e-3, 1e-3)
        nn.init.constant_(self.gate[3].bias, -0.5) # conservative initialization (~38%) lower gradient at the start, to prevent early dominance of one stream

    def forward(self, features_list):
        # stacked: (B, num_streams, dim)
        stacked = torch.stack(features_list, dim=1) 
        
        # 2. Calculate independent weights for each stream
        combined = torch.cat(features_list, dim=1)
        weights = self.gate(combined) # (B, num_streams) -> values independent [0, 1]
        
        # Apply the weights to each stream
        gated_features = stacked * weights.unsqueeze(-1)

        # Add a tiny epsilon to prevent division by zero
        weight_sum = weights.sum(dim=1, keepdim=True) + 1e-6
        
        # # 4. Apply weighted sum
        # out = gated_features.sum(dim=1)
        out = gated_features.sum(dim=1) / weight_sum  # Normalize by sum of weights to prevent scale issues
        return out, weights
    
@register_fusion("ARGate_Fusion")
class ARGatePlusFusion(BaseFusion):
    def __init__(self, dim, num_streams):   
        super().__init__(dim, num_streams)

        self.gate = nn.Sequential(
            nn.Linear(dim * num_streams, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_streams)
        )
        nn.init.uniform_(self.gate[3].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.gate[3].bias)

        # 2. Monotonic Target Network (f_k): Learns targets from auxiliary losses.
        # It must be structurally constrained to be monotonic non-increasing.
        # A standard way to enforce this in a mini-net is keeping weights negative.
        self.target_net = nn.Sequential(
            nn.Linear(num_streams, 16),
            nn.Tanh(), # Tanh helps bound intermediate representations nicely
            nn.Linear(16, num_streams),
            nn.Sigmoid() # Keeps targets strictly between [0, 1]
        )

    def enforce_monotonicity(self):
        """
        Enforces a monotonic non-increasing relationship by clamping the 
        weights of the target network to be non-positive (<= 0). 
        Call this after optimizer.step() in your training loop.
        """
        with torch.no_grad():
            for layer in self.target_net:
                if isinstance(layer, nn.Linear):
                    layer.weight.clamp_(max=0.0)

    def forward(self, features_list, aux_losses=None):
        """
        Args:
            features_list (list[Tensor]): List of feature tensors per stream.
            aux_losses (Tensor, optional): Tensor of shape (B, num_streams) containing 
                                          the individual losses for each auxiliary path.
                                          Required during training for target learning.
        """
        # 1. Stack and concatenate features
        stacked = torch.stack(features_list, dim=1) # (B, num_streams, dim)
        combined = torch.cat(features_list, dim=1)   # (B, dim * num_streams)
        
        # 2. Extract fusion weights from the main model
        logits = self.gate(combined)
        # Softmax ensures sum(weights) = 1 across streams
        weights = torch.softmax(logits, dim=1) # (B, num_streams)
        
        # 3. Apply weighted sum for final sensor fusion
        out = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        
        # 4. If training and aux_losses are provided, compute the Monotonic Targets
        fusion_targets = None
        if aux_losses is not None:
            # f_k(Loss_aux) -> mapping errors to ideal trust targets
            fusion_targets = self.target_net(aux_losses) # (B, num_streams)

        return out, weights, fusion_targets