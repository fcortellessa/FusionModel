import torch
import torch.nn.functional as F

def quaternion_geodesic_distance(q1, q2, eps=1e-7):
    """
    Compute geodesic distance between two quaternions.
    Returns the mean smallest rotation angle (in radians) on SO(3).
    """
    # Normalize quaternions to unit norm
    q1 = F.normalize(q1, p=2, dim=-1, eps=eps)
    q2 = F.normalize(q2, p=2, dim=-1, eps=eps)

    # Dot product and clamp for stability
    dot = torch.sum(q1 * q2, dim=-1)
    dot = torch.clamp(torch.abs(dot), max=1.0 - eps)

    # Geodesic angle = 2 * acos(|q1·q2|)
    angle = 2.0 * torch.acos(dot)
    return angle.mean()

def total_loss(target_pose, pred_pose, loss_weight=1.0, rot_scaling_factor=200.0):
    """
    Compute total loss from predicted vs. target poses.
    
    Pose vector: [x, y, z, qx, qy, qz, qw]
    Returns: (total_loss, loss_q, loss_y, loss_z)
    """
    # Slice translation (y, z are indices 1, 2) and rotation (3:7)
    trn_target = target_pose[:, 1:3] 
    trn_pred = pred_pose[:, 1:3]
    
    quat_target = target_pose[:, 3:]
    quat_pred = pred_pose[:, 3:]

    # 1. Rotation loss (scaled by rot_scaling_factor)
    loss_q = quaternion_geodesic_distance(quat_target, quat_pred) * rot_scaling_factor

    # 2. Translation losses (y and z separately for tracking)
    loss_y = F.mse_loss(trn_target[:, 0], trn_pred[:, 0])
    loss_z = F.mse_loss(trn_target[:, 1], trn_pred[:, 1])

    # 3. Combine losses
    total = (loss_y + loss_z + loss_q) * loss_weight

    return total, loss_q, loss_y, loss_z

def test_loss(target_pose, pred_pose):
    """
    Compute mean absolute errors (MAE) for evaluation.
    """
    # Scale to mm ?? -> hasn't it been normalised and lost the original scale?
    trn_target = target_pose[:, 1:3] * 1#000.0
    trn_pred = pred_pose[:, 1:3] * 1#000.0
    
    # Translation MAE
    diff_y = torch.abs(trn_target[:, 0] - trn_pred[:, 0]).mean()
    diff_z = torch.abs(trn_target[:, 1] - trn_pred[:, 1]).mean()

    # Rotation Error (Radians)
    diff_q = quaternion_geodesic_distance(target_pose[:, 3:], pred_pose[:, 3:])

    return diff_y.item(), diff_z.item(), diff_q.item()