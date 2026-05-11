"""
Loss functions for training the model. This includes:
- MSE loss for translation (y, z)
- Lie Algebra Log-Euclidian Loss for rotation (quaternions)
The total loss is a weighted combination of these components, with scaling to balance their contributions.
The test loss function computes absolute errors for evaluation purposes, providing insights into the model's performance in terms of translation and rotation accuracy.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss_translation(pred_trans, target_trans):
    """
    Standard Mean Squared Error for (y, z) coordinates.
    Returns a scalar tensor.
    """
    return F.mse_loss(pred_trans, target_trans)


def lie_algebra_loss_quaternion(pred_q, target_q, eps=1e-7):
    """
    Computes the L2 norm of the relative rotation in the Lie Algebra (so3).
    Input: Quaternions in [w, x, y, z] formatnsl
    """
    # 1. Ensure unit quaternions (critical for log-map)
    pred_q = F.normalize(pred_q, p=2, dim=-1)
    target_q = F.normalize(target_q, p=2, dim=-1)

    # 2. Compute relative rotation: Δq = target_inv * pred
    # Conjugate of target [w, -x, -y, -z]
    target_inv = target_q.clone()
    target_inv[:, 1:] *= -1
    
    # Quaternion multiplication (Hamilton product)
    w1, x1, y1, z1 = target_inv.unbind(dim=-1)
    w2, x2, y2, z2 = pred_q.unbind(dim=-1)
    
    res_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    res_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    res_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    res_z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    rel_q = torch.stack([res_w, res_x, res_y, res_z], dim=-1)

    # 3. Log-Map to Tangent Space (so3)
    # Extract scalar (w) and vector (v)
    # We use the absolute value of w to handle the antipodal property (q = -q)
    w = rel_q[:, 0:1]
    v = rel_q[:, 1:]
    
    # Check if w is negative to pick the shortest path on the sphere
    v = torch.where(w < 0, -v, v)
    w = torch.abs(w)
    
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    
    # Angle theta = 2 * atan2(||v||, w)
    theta = 2.0 * torch.atan2(v_norm, w)
    
    # Numerical stability: if theta is small, scale -> 2.0 (Taylor expansion)
    # otherwise scale -> theta / sin(theta/2)
    # Since sin(theta/2) is exactly v_norm for unit quaternions:
    scale = torch.where(
        v_norm > eps,
        theta / v_norm,
        2.0 * torch.ones_like(v_norm)
    )
    
    # The rotation vector (Lie Algebra element)
    rotation_vector = scale * v
    
    # Return the mean L2 norm of the rotation vectors
    return torch.mean(torch.norm(rotation_vector, p=2, dim=-1))


class LieAlgebraRotationLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_q, target_q):
        """
        Calculates L2 loss in the Lie Algebra so(3).
        Input: [N, 4] quaternions (w, x, y, z)
        """
        # 1. Normalize to ensure they stay on the unit sphere
        pred_q = F.normalize(pred_q, p=2, dim=-1)
        target_q = F.normalize(target_q, p=2, dim=-1)

        # 2. Compute Relative Quaternion: Δq = target_inv * pred
        # Conjugate of target is [w, -x, -y, -z]
        w1, x1, y1, z1 = target_q[:, 0], -target_q[:, 1], -target_q[:, 2], -target_q[:, 3]
        w2, x2, y2, z2 = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2], pred_q[:, 3]

        # Hamiltonian Product
        res_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        res_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        res_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        res_z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        rel_q = torch.stack([res_w, res_x, res_y, res_z], dim=-1)

        # 3. Shortest Path (Antipodal Property)
        # If w < 0, the rotation is > 180 degrees. 
        # We flip the quaternion to represent the same rotation via the shorter arc.
        rel_q = torch.where(rel_q[:, 0:1] < 0, -rel_q, rel_q)

        # 4. Log-Map: q -> rotation vector (ω)
        # Extract scalar (w) and vector (v)
        w = rel_q[:, 0:1].clamp(-1.0 + self.eps, 1.0 - self.eps)
        v = rel_q[:, 1:]
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        
        # θ = 2 * atan2(||v||, w)
        theta = 2.0 * torch.atan2(v_norm, w)
        
        # Numerical stability for small angles:
        # As theta -> 0, theta/sin(theta/2) -> 2.0
        # In quaternions, sin(theta/2) is exactly v_norm.
        scale = torch.where(
            v_norm > self.eps,
            theta / v_norm,
            2.0 * torch.ones_like(v_norm)
        )
        
        # The rotation vector in so(3)
        omega = scale * v
        
        # 5. Loss is the L2 norm of the rotation vector
        return torch.mean(torch.norm(omega, p=2, dim=-1))
    

class PoseLossManager(nn.Module):
    """
    Combines translation and rotation losses with learned weighting. 
    This allows the model to balance the two tasks dynamically during training.
    """
    def __init__(self):
        super().__init__()
        self.rot_loss_fn = LieAlgebraRotationLoss()
        
        # Trainable log-variances (initialized to 0.0 -> weight of 1.0)
        # These allow the model to learn the relative 'noise' of each task.
        self.log_var_trans = nn.Parameter(torch.zeros(1))
        self.log_var_rot = nn.Parameter(torch.zeros(1))

    def forward(self, pred_t, target_t, pred_q, target_q):
        # Calculate raw losses
        l_trans = F.mse_loss(pred_t, target_t)
        l_rot = self.rot_loss_fn(pred_q, target_q)

        # Apply learned weighting: Loss = (1 / exp(log_var)) * L + log_var
        # This prevents the weights from going to zero and balances scales.
        weighted_trans = l_trans * torch.exp(-self.log_var_trans) + self.log_var_trans
        weighted_rot = l_rot * torch.exp(-self.log_var_rot) + self.log_var_rot
        
        total_loss = weighted_trans + weighted_rot
        
        return total_loss, l_trans, l_rot



# --- Example usage in a training loop ---
def compute_total_loss(pred_t, target_t, pred_q, target_q, w_t=1.0, w_q=0.5):
    l_mse = mse_loss_translation(pred_t, target_t)
    l_lie = lie_algebra_loss_quaternion(pred_q, target_q)
    
    # Combined loss
    total_loss = (w_t * l_mse) + (w_q * l_lie)
    
    return total_loss, l_mse, l_lie


# --- SE(3) Direct Optimization via Lie Algebra ---
def quaternion_to_rotation_vector(quat, eps=1e-7):
    """
    Convert quaternion to rotation vector (axis-angle) using exponential map.
    Input: quat tensor shape (B, 4) in [x, y, z, w] format
    Output: rotation_vector shape (B, 3)
    """
    # Normalize
    quat = F.normalize(quat, p=2, dim=-1)
    
    # Extract components
    x, y, z, w = quat[:, 0:1], quat[:, 1:2], quat[:, 2:3], quat[:, 3:4]
    
    # Log map: compute the axis-angle representation
    # For quaternion q = [sin(θ/2) * axis, cos(θ/2)]
    # The rotation vector is θ * axis
    
    # Clamp w to valid range for numerical stability
    w = torch.clamp(w, -1.0 + eps, 1.0 - eps)
    
    # θ = 2 * acos(w)
    theta = 2.0 * torch.acos(w)
    
    # sin(θ/2) = ||[x, y, z]||
    sin_half_theta = torch.sqrt(x**2 + y**2 + z**2)
    
    # Numerically stable scale computation
    # As theta -> 0, scale -> 2.0 (from Taylor expansion)
    # Otherwise scale = theta / sin_half_theta
    scale = torch.where(
        sin_half_theta > eps,
        theta / sin_half_theta,
        2.0 * torch.ones_like(sin_half_theta)
    )
    
    rotation_vector = scale * torch.cat([x, y, z], dim=-1)
    return rotation_vector


def rotation_vector_to_quaternion(rot_vec, eps=1e-7):
    """
    Convert rotation vector (axis-angle) to quaternion via exponential map.
    Input: rot_vec tensor shape (B, 3)
    Output: quat tensor shape (B, 4) in [x, y, z, w] format
    """
    # θ = ||ω||
    theta = torch.norm(rot_vec, p=2, dim=-1, keepdim=True)
    
    # Normalized axis
    # For small angles, use Taylor expansion to avoid division by zero
    scale = torch.where(
        theta > eps,
        torch.sin(theta / 2.0) / theta,
        0.5 * torch.ones_like(theta)
    )
    
    xyz = scale * rot_vec
    w = torch.cos(theta / 2.0)
    
    quat = torch.cat([xyz, w], dim=-1)
    return quat


class SE3DirectLoss(nn.Module):
    """
    Direct SE(3) loss in Lie algebra space.
    Optimizes the complete SE(3) pose as a unified 6D vector,
    where the distance is measured in the tangent space of SE(3).
    
    The loss treats translation and rotation symmetrically in the Lie algebra,
    allowing the optimization to directly optimize SE(3) geodesics.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_se3, target_se3):
        """
        Calculates L2 loss for SE(3) Lie algebra elements.
        
        Args:
            pred_se3: [B, 6] predicted SE(3) in Lie algebra (3 translation + 3 rotation)
            target_se3: [B, 6] target SE(3) in Lie algebra (3 translation + 3 rotation)
        
        Returns:
            loss: scalar tensor representing the SE(3) geodesic distance
        """
        # Compute relative SE(3) element in the Lie algebra
        # δ_se3 = target^-1 * pred (in Lie algebra approximation)
        # For small perturbations, this is approximately: target - pred
        delta_se3 = pred_se3 - target_se3
        
        # Compute the norm (geodesic distance in SE(3))
        # This treats translation and rotation uniformly in the Lie algebra
        se3_norm = torch.norm(delta_se3, p=2, dim=-1)
        
        # Return mean L2 norm
        return torch.mean(se3_norm)


def se3_direct_loss(pred_se3, target_se3, eps=1e-7):
    """
    Functional interface for SE(3) direct loss.
    
    Args:
        pred_se3: [B, 6] predicted SE(3) in Lie algebra
        target_se3: [B, 6] target SE(3) in Lie algebra
    
    Returns:
        loss: scalar tensor
    """
    delta_se3 = pred_se3 - target_se3
    se3_norm = torch.norm(delta_se3, p=2, dim=-1)
    return torch.mean(se3_norm)

