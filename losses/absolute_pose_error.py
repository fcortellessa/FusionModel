# absolute_pose_error.py

import torch
import torch.nn.functional as F
import tqdm

def absolute_pose_error(pred, target):
    """
    Calculates Translation Error (Euclidean) and Rotation Error (Angular).
    
    Args:
        pred: (N, 7) tensor [x, y, z, qw, qx, qy, qz]
        target: (N, 7) tensor [x, y, z, qw, qx, qy, qz]
        
    Returns:
        trans_error: Mean Euclidean distance
        rot_error: Mean angular error in degrees
    """
    # 1. Translation Error (Euclidean Distance)
    # sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
    trans_error = torch.norm(pred[:, :3] - target[:, :3], dim=1)
    
    # 2. Rotation Error (Angular Distance)
    # The dot product of two unit quaternions is the cos(theta/2)
    q1 = pred[:, 3:]
    q2 = target[:, 3:]
    
    # Ensure quaternions are normalized
    q1 = q1 / torch.norm(q1, dim=1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=1, keepdim=True)
    
    # Clamp to avoid NaNs in acos due to floating point precision
    d_abs = torch.abs(torch.sum(q1 * q2, dim=1))
    d_abs = torch.clamp(d_abs, 0.0, 1.0)
    
    # Angle formula: theta = 2 * acos(|q1 dot q2|)
    rot_error_rad = 2 * torch.acos(d_abs)
    rot_error_deg = torch.rad2deg(rot_error_rad)
    
    return trans_error.mean(), rot_error_deg.mean()

def absolute_pose_error_x_rot(pred, target):
    """
    Calculates Translation Error and Rotation Error specifically around the X-axis.
    
    Args:
        pred: (N, 7) tensor [x, y, z, qx, qy, qz, qw]
        target: (N, 7) tensor [x, y, z, qx, qy, qz, qw]
    """
    # Euclidean Distance for translation error
    trans_error = torch.norm(pred[:, 1:3] - target[:, 1:3], dim=1)
    y_error = torch.abs(pred[:, 1] - target[:, 1])
    z_error = torch.abs(pred[:, 2] - target[:, 2])


    def get_x_rotation(q):
        q = F.normalize(q, p=2, dim=1, eps=1e-7)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Used formula: atan2(2(wx + yz), 1 - 2(x^2 + y^2))
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        return torch.atan2(sinr_cosp, cosr_cosp)

    roll_pred = get_x_rotation(pred[:, 3:])
    roll_target = get_x_rotation(target[:, 3:])

    # Calculate shortest distance between angles (handling wrapping at pi)
    rot_x_error_rad = torch.abs(roll_pred - roll_target)
    rot_x_error_rad = torch.min(rot_x_error_rad, 2 * torch.pi - rot_x_error_rad)
    
    rot_x_error_deg = torch.rad2deg(rot_x_error_rad)

    return y_error.mean(), z_error.mean(), trans_error.mean(), rot_x_error_deg.mean()
