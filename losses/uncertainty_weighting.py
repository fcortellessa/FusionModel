"""
Multi-task uncertainty weighting for loss balancing.

This implements the uncertainty weighting approach from:
"Multi-Task Learning Using Uncertainty to Weigh Losses" by Kendall et al.
https://arxiv.org/abs/1802.07335

The learned log-variances act as uncertainty estimates for each task (rotation and translation),
automatically weighting the losses during training.
"""

import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """
    Multi-task uncertainty weighting module for combining rotation and translation losses.
    
    Learns log-variances (log_var_rot, log_var_trans) as parameters, which are used to 
    weight the respective loss components. Higher variance = lower weight (less confident).
    
    Args:
        init_log_var_rot: Initial log-variance for rotation loss (default: 0.0)
        init_log_var_trans: Initial log-variance for translation loss (default: 0.0)
    
    Example:
        >>> uncertainty = UncertaintyWeighting()
        >>> loss_rot = compute_rotation_loss(...)
        >>> loss_trans = compute_translation_loss(...)
        >>> total_loss = uncertainty(loss_rot, loss_trans)
        >>> # The model will learn to adjust log_var_rot and log_var_trans
    """
    
    def __init__(self, init_log_var_rot=0.0, init_log_var_trans=0.0):
        super().__init__()
        self.log_var_rot = nn.Parameter(
            torch.tensor(float(init_log_var_rot), dtype=torch.float32)
        )
        self.log_var_trans = nn.Parameter(
            torch.tensor(float(init_log_var_trans), dtype=torch.float32)
        )
    
    def forward(self, loss_rot, loss_trans):
        """
        Compute weighted loss combining rotation and translation components.
        
        Args:
            loss_rot: Scalar tensor for rotation loss
            loss_trans: Scalar tensor for translation loss
            
        Returns:
            Scalar tensor: Weighted total loss
            
        Formula:
            L_total = exp(-log_var_rot) * loss_rot + log_var_rot 
                    + exp(-log_var_trans) * loss_trans + log_var_trans
        """
        weighted_rot = torch.exp(-self.log_var_rot) * loss_rot + self.log_var_rot
        weighted_trans = torch.exp(-self.log_var_trans) * loss_trans + self.log_var_trans
        return weighted_rot + weighted_trans
