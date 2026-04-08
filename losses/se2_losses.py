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

def total_loss_2D_vector(pred_trans, target_trans, pred_vec, target_vec, lambda_rot=1.0):
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


