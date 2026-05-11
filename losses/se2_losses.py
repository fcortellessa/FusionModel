"""
Model Output: 
- Translation in (y, z) coordinates
- 2D Unit Vector $(\sin, \cos)$ for rotation in the plane (around x-axis)

Losses:
- MSE Loss for translation
- Geodesic Loss for rotation, computed as the angle between predicted and target unit vectors:
$$L_{rot} = \arccos(\text{clamp}(\hat{r} \cdot r_{target}, -1 + \epsilon, 1 - \epsilon))$$
This loss directly measures the angular difference in the plane, ensuring that the model learns to predict the correct orientation regardless of the quadrant.
"""

import torch
import torch.nn.functional as F

def axis_angle_to_quaternion(axis_angle):
    """
    Convert axis-angle representation (Lie algebra so(3)) to quaternion (SO(3)).
    Uses the exponential map: exp(θ * v) where v is unit axis, θ is angle.
    
    Input: axis_angle of shape (batch_size, 3) representing rotation vector
    Output: quaternion of shape (batch_size, 4) as [x, y, z, w]
    """
    # Compute the angle (norm of the axis-angle vector)
    theta = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # (B, 1)
    
    # Avoid division by zero
    theta_safe = torch.clamp(theta, min=1e-7)
    
    # Compute the unit axis
    axis = axis_angle / theta_safe  # (B, 3)
    
    # Half angle (for quaternion representation)
    half_theta = theta / 2.0
    
    # Quaternion: [sin(θ/2) * axis, cos(θ/2)]
    xyz = torch.sin(half_theta) * axis  # (B, 3)
    w = torch.cos(half_theta)  # (B, 1)
    
    # Concatenate to form quaternion [x, y, z, w]
    quat = torch.cat([xyz, w], dim=-1)  # (B, 4)
    
    return quat


def mse_loss(pred_trans, target_trans):
    """
    Standard Mean Squared Error for (y, z) coordinates.
    Returns a scalar tensor.
    """
    return F.mse_loss(pred_trans, target_trans)

def geodesic_loss_2d_unit_vector(pred_vec, target_vec, eps=1e-7):
    """
    Computes the geodesic loss between two 2D unit vectors representing rotation in the plane.
    The loss is the angle between the predicted and target unit vectors.
    Input: pred_vec and target_vec are tensors of shape (batch_size, 2) representing (sin, cos).
    """
    # Ensure unit vectors
    pred_vec = F.normalize(pred_vec, p=2, dim=-1)
    target_vec = F.normalize(target_vec, p=2, dim=-1)

    # Compute dot product
    dot_product = torch.sum(pred_vec * target_vec, dim=-1)

    # Clamp to avoid numerical issues with arccos
    dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)

    # Geodesic loss is the angle between the two vectors
    loss = torch.acos(dot_product)
    
    return loss.mean()

def normalized_mse_loss_2d_unit_vector(pred_vec, target_vec):
    """
    Normalized MSE loss: L2-normalize both vectors, then apply MSE.
    This is often easier to optimize than arccos and works well for unit vector targets.
    Input: pred_vec and target_vec are tensors of shape (batch_size, 2) representing (sin, cos).
    """
    pred_vec = F.normalize(pred_vec, p=2, dim=-1)
    target_vec = F.normalize(target_vec, p=2, dim=-1)
    return F.mse_loss(pred_vec, target_vec)

def wrapped_angle_loss(pred_angle, target_angle, eps=1e-7):
    """
    Loss for direct angle prediction with wraparound at ±π.
    Computes the shortest angular distance between two angles.
    Both inputs are normalized to [-π, π] before computing distance.
    Input: pred_angle and target_angle are tensors of shape (batch_size, 1) or (batch_size,) in radians.
    """
    # Ensure 1D
    if pred_angle.dim() > 1:
        pred_angle = pred_angle.squeeze(-1)
    if target_angle.dim() > 1:
        target_angle = target_angle.squeeze(-1)
    
    # Normalize both angles to [-π, π]
    # Use atan2(sin(x), cos(x)) to normalize
    pred_angle_norm = torch.atan2(torch.sin(pred_angle), torch.cos(pred_angle))
    target_angle_norm = torch.atan2(torch.sin(target_angle), torch.cos(target_angle))
    
    # Compute wrapped difference
    diff = pred_angle_norm - target_angle_norm
    # Normalize diff to [-π, π]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    
    # L1 loss on wrapped angle
    return torch.abs(diff).mean()

def total_loss_2D_vector(pred_trans, target_trans, pred_vec, target_vec, lambda_rot=200.0):
    """
    Combines translation and rotation losses into a total loss.
    Args:
- pred_trans: Predicted translation tensor of shape (batch_size, 2)
- target_trans: Target translation tensor of shape (batch_size, 2)
- pred_vec: Predicted rotation unit vector tensor of shape (batch_size, 2)
- target_vec: Target rotation unit vector tensor of shape (batch_size, 2)
- lambda_rot: Weighting factor for the rotation loss
    """
    loss_trans = mse_loss(pred_trans, target_trans)
    loss_rot = geodesic_loss_2d_unit_vector(pred_vec, target_vec)
    return loss_trans + lambda_rot * loss_rot


