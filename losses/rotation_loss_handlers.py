"""
Rotation parametrisation-specific loss and metric computation.
Encapsulates the logic for different rotation representations (2D unit vector, angle, lie algebra, se3_direct, quaternion).
"""

import torch
from losses.se2_losses import geodesic_loss_2d_unit_vector, normalized_mse_loss_2d_unit_vector, wrapped_angle_loss, quaternion_to_2d_unit_vector, axis_angle_to_quaternion
from losses.se3_losses import se3_direct_loss, rotation_vector_to_quaternion, quaternion_to_rotation_vector
from losses.absolute_pose_error import absolute_pose_error_x_rot


class RotationLossHandler:
    """Base class for rotation parametrisation loss handling."""
    
    def compute_loss(self, pose_pred, pose, uncertainty_module=None):
        """
        Compute loss and loss components (loss_q, loss_y, loss_z).
        Returns: (total_loss, loss_q, loss_y, loss_z, loss_trans)
        """
        raise NotImplementedError
    
    def compute_metrics(self, pose_pred, pose):
        """
        Compute absolute error metrics for logging.
        Returns: (abs_trans_y, abs_trans_z, abs_trans, abs_rot_x)
        """
        raise NotImplementedError

    def compute_rotation_errors_degrees(self, pose_pred, pose):
        """Return one roll-angle error in degrees per example for percentiles."""
        raise NotImplementedError

    def compute_rotation_errors_degrees_signed(self, pose_pred, pose):
        """Return signed roll-angle error in degrees per example (preserving sign)."""
        raise NotImplementedError


def _roll_error_degrees(pred_roll_vec, target_roll_vec):
    """Compute shortest signed-angle distance for sine/cosine roll vectors."""
    pred_angle = torch.atan2(pred_roll_vec[:, 0], pred_roll_vec[:, 1])
    target_angle = torch.atan2(target_roll_vec[:, 0], target_roll_vec[:, 1])
    difference = torch.abs(pred_angle - target_angle)
    difference = torch.atan2(torch.sin(difference), torch.cos(difference))
    return torch.rad2deg(torch.abs(difference))


def _roll_error_degrees_signed(pred_roll_vec, target_roll_vec):
    """Compute shortest signed-angle distance (preserving sign) for sine/cosine roll vectors.

    Returns signed error in degrees where positive means prediction is ahead of target.
    """
    pred_angle = torch.atan2(pred_roll_vec[:, 0], pred_roll_vec[:, 1])
    target_angle = torch.atan2(target_roll_vec[:, 0], target_roll_vec[:, 1])
    difference = pred_angle - target_angle
    # Wrap to [-pi, pi]
    difference = torch.atan2(torch.sin(difference), torch.cos(difference))
    return torch.rad2deg(difference)


class TwoDUnitVectorHandler(RotationLossHandler):
    """Handler for 2D unit vector rotation parametrisation."""
    
    def __init__(self, rot_scaling_factor=2000.0, rotation_loss_type="geodesic"):
        self.rot_scaling_factor = rot_scaling_factor
        self.rotation_loss_type = rotation_loss_type
    
    def compute_loss(self, pose_pred, pose, uncertainty_module=None):
        pred_trans = pose_pred["trans"][:, 1:3]
        target_trans = pose[:, 1:3]
        pred_vec = pose_pred["rotation"]
        target_vec = quaternion_to_2d_unit_vector(pose[:, 3:])
        
        loss_y = torch.nn.functional.mse_loss(pred_trans[:, 0], target_trans[:, 0])
        loss_z = torch.nn.functional.mse_loss(pred_trans[:, 1], target_trans[:, 1])
        
        # Use selected rotation loss type
        if self.rotation_loss_type == "normalized_mse":
            loss_q = normalized_mse_loss_2d_unit_vector(pred_vec, target_vec)
        else:  # default to geodesic
            loss_q = geodesic_loss_2d_unit_vector(pred_vec, target_vec)
        
        loss_q_scaled = loss_q * self.rot_scaling_factor
        loss_trans = loss_y + loss_z
        base_loss = loss_q_scaled + loss_y + loss_z
        
        if uncertainty_module is not None:
            loss = uncertainty_module(loss_q_scaled, loss_trans)
        else:
            loss = base_loss
        
        return loss, loss_q, loss_y, loss_z, loss_trans
    
    def compute_metrics(self, pose_pred, pose):
        pred_trans = pose_pred["trans"][:, 1:3]
        target_trans = pose[:, 1:3]
        pred_vec = pose_pred["rotation"]
        target_vec = quaternion_to_2d_unit_vector(pose[:, 3:])
        
        abs_trans_y = torch.abs(pred_trans[:, 0] - target_trans[:, 0]).mean()
        abs_trans_z = torch.abs(pred_trans[:, 1] - target_trans[:, 1]).mean()
        abs_trans = torch.norm(pred_trans - target_trans, dim=1).mean()
        
        pred_angle = torch.atan2(pred_vec[:, 0], pred_vec[:, 1])
        target_angle = torch.atan2(target_vec[:, 0], target_vec[:, 1])
        rot_diff = torch.abs(pred_angle - target_angle)
        rot_diff = torch.min(rot_diff, 2 * torch.pi - rot_diff)
        abs_rot_x = torch.rad2deg(rot_diff).mean()
        
        return abs_trans_y, abs_trans_z, abs_trans, abs_rot_x

    def compute_rotation_errors_degrees(self, pose_pred, pose):
        return _roll_error_degrees(
            pose_pred["rotation"], quaternion_to_2d_unit_vector(pose[:, 3:])
        )

    def compute_rotation_errors_degrees_signed(self, pose_pred, pose):
        return _roll_error_degrees_signed(
            pose_pred["rotation"], quaternion_to_2d_unit_vector(pose[:, 3:])
        )


class AngleHandler(RotationLossHandler):
    """Handler for angle (wrapped) rotation parametrisation."""
    
    def __init__(self, rot_scaling_factor=200.0):
        self.rot_scaling_factor = rot_scaling_factor
    
    def compute_loss(self, pose_pred, pose, uncertainty_module=None):
        pred_trans = pose_pred["trans"][:, 1:3]
        target_trans = pose[:, 1:3]
        pred_angle = pose_pred["rotation"]  # shape: [B, 1]
        
        # Extract target roll from quaternion
        target_roll = quaternion_to_2d_unit_vector(pose[:, 3:])  # [B, 2] as (sin, cos)
        target_angle = torch.atan2(target_roll[:, 0], target_roll[:, 1])  # [B]
        target_angle = target_angle.unsqueeze(-1)  # [B, 1]
        
        loss_y = torch.nn.functional.mse_loss(pred_trans[:, 0], target_trans[:, 0])
        loss_z = torch.nn.functional.mse_loss(pred_trans[:, 1], target_trans[:, 1])
        loss_q = wrapped_angle_loss(pred_angle, target_angle)
        loss_q_scaled = loss_q * self.rot_scaling_factor
        loss_trans = loss_y + loss_z
        base_loss = loss_q_scaled + loss_y + loss_z
        
        if uncertainty_module is not None:
            loss = uncertainty_module(loss_q_scaled, loss_trans)
        else:
            loss = base_loss
        
        return loss, loss_q, loss_y, loss_z, loss_trans
    
    def compute_metrics(self, pose_pred, pose):
        pred_trans = pose_pred["trans"][:, 1:3]
        target_trans = pose[:, 1:3]
        pred_angle = pose_pred["rotation"]
        
        abs_trans_y = torch.abs(pred_trans[:, 0] - target_trans[:, 0]).mean()
        abs_trans_z = torch.abs(pred_trans[:, 1] - target_trans[:, 1]).mean()
        abs_trans = torch.norm(pred_trans - target_trans, dim=1).mean()
        
        # Extract target roll from quaternion
        target_roll = quaternion_to_2d_unit_vector(pose[:, 3:])
        target_angle = torch.atan2(target_roll[:, 0], target_roll[:, 1]).unsqueeze(-1)
        
        # Normalize angles for metric computation
        pred_angle_norm = torch.atan2(torch.sin(pred_angle), torch.cos(pred_angle))
        target_angle_norm = torch.atan2(torch.sin(target_angle), torch.cos(target_angle))
        rot_diff = torch.abs(pred_angle_norm - target_angle_norm)
        rot_diff = torch.atan2(torch.sin(rot_diff), torch.cos(rot_diff))
        abs_rot_x = torch.rad2deg(torch.abs(rot_diff)).mean()
        
        return abs_trans_y, abs_trans_z, abs_trans, abs_rot_x

    def compute_rotation_errors_degrees(self, pose_pred, pose):
        pred_angle = pose_pred["rotation"].squeeze(-1)
        target_roll = quaternion_to_2d_unit_vector(pose[:, 3:])
        target_angle = torch.atan2(target_roll[:, 0], target_roll[:, 1])
        difference = torch.atan2(
            torch.sin(pred_angle - target_angle), torch.cos(pred_angle - target_angle)
        )
        return torch.rad2deg(torch.abs(difference))

    def compute_rotation_errors_degrees_signed(self, pose_pred, pose):
        pred_angle = pose_pred["rotation"].squeeze(-1)
        target_roll = quaternion_to_2d_unit_vector(pose[:, 3:])
        target_angle = torch.atan2(target_roll[:, 0], target_roll[:, 1])
        difference = torch.atan2(
            torch.sin(pred_angle - target_angle), torch.cos(pred_angle - target_angle)
        )
        return torch.rad2deg(difference)


class LieAlgebraHandler(RotationLossHandler):
    """Handler for Lie algebra (axis-angle) rotation parametrisation."""
    
    def __init__(self, rot_scaling_factor=200.0):
        self.rot_scaling_factor = rot_scaling_factor
    
    def compute_loss(self, pose_pred, pose, uncertainty_module=None):
        pred_trans = pose_pred["trans"][:, 1:3]
        target_trans = pose[:, 1:3]
        pred_axis_angle = pose_pred["rotation"]  # shape: [B, 3]
        
        # Convert predicted axis-angle to quaternion via exponential map
        pred_quat = axis_angle_to_quaternion(pred_axis_angle)
        target_quat = pose[:, 3:]  # ground truth quaternion [x,y,z,w]
        
        loss_y = torch.nn.functional.mse_loss(pred_trans[:, 0], target_trans[:, 0])
        loss_z = torch.nn.functional.mse_loss(pred_trans[:, 1], target_trans[:, 1])
        
        # Normalize quaternions
        pred_quat_norm = torch.nn.functional.normalize(pred_quat, p=2, dim=1, eps=1e-7)
        target_quat_norm = torch.nn.functional.normalize(target_quat, p=2, dim=1, eps=1e-7)
        
        # Geodesic distance: 2 * arccos(|dot product|)
        dot_prod = torch.sum(pred_quat_norm * target_quat_norm, dim=1)
        dot_prod = torch.clamp(torch.abs(dot_prod), 0.0, 1.0 - 1e-7)
        loss_q = 2.0 * torch.acos(dot_prod).mean()
        loss_q_scaled = loss_q * self.rot_scaling_factor
        loss_trans = loss_y + loss_z
        base_loss = loss_q_scaled + loss_y + loss_z
        
        if uncertainty_module is not None:
            loss = uncertainty_module(loss_q_scaled, loss_trans)
        else:
            loss = base_loss
        
        return loss, loss_q, loss_y, loss_z, loss_trans
    
    def compute_metrics(self, pose_pred, pose):
        pred_trans = pose_pred["trans"][:, 1:3]
        target_trans = pose[:, 1:3]
        pred_axis_angle = pose_pred["rotation"]
        
        # Convert to quaternion for metric computation
        pred_quat = axis_angle_to_quaternion(pred_axis_angle)
        target_quat = pose[:, 3:]
        pred_quat_norm = torch.nn.functional.normalize(pred_quat, p=2, dim=1, eps=1e-7)
        target_quat_norm = torch.nn.functional.normalize(target_quat, p=2, dim=1, eps=1e-7)
        
        abs_trans_y = torch.abs(pred_trans[:, 0] - target_trans[:, 0]).mean()
        abs_trans_z = torch.abs(pred_trans[:, 1] - target_trans[:, 1]).mean()
        abs_trans = torch.norm(pred_trans - target_trans, dim=1).mean()
        
        # Extract roll for metric comparison
        pred_roll_vec = quaternion_to_2d_unit_vector(pred_quat_norm)
        target_roll_vec = quaternion_to_2d_unit_vector(target_quat_norm)
        pred_angle = torch.atan2(pred_roll_vec[:, 0], pred_roll_vec[:, 1])
        target_angle = torch.atan2(target_roll_vec[:, 0], target_roll_vec[:, 1])
        rot_diff = torch.abs(pred_angle - target_angle)
        rot_diff = torch.atan2(torch.sin(rot_diff), torch.cos(rot_diff))
        abs_rot_x = torch.rad2deg(torch.abs(rot_diff)).mean()
        
        return abs_trans_y, abs_trans_z, abs_trans, abs_rot_x

    def compute_rotation_errors_degrees(self, pose_pred, pose):
        pred_quat = torch.nn.functional.normalize(
            axis_angle_to_quaternion(pose_pred["rotation"]), p=2, dim=1, eps=1e-7
        )
        target_quat = torch.nn.functional.normalize(pose[:, 3:], p=2, dim=1, eps=1e-7)
        return _roll_error_degrees(
            quaternion_to_2d_unit_vector(pred_quat), quaternion_to_2d_unit_vector(target_quat)
        )

    def compute_rotation_errors_degrees_signed(self, pose_pred, pose):
        pred_quat = torch.nn.functional.normalize(
            axis_angle_to_quaternion(pose_pred["rotation"]), p=2, dim=1, eps=1e-7
        )
        target_quat = torch.nn.functional.normalize(pose[:, 3:], p=2, dim=1, eps=1e-7)
        return _roll_error_degrees_signed(
            quaternion_to_2d_unit_vector(pred_quat), quaternion_to_2d_unit_vector(target_quat)
        )


class SE3DirectHandler(RotationLossHandler):
    """Handler for SE(3) direct optimization in Lie algebra space."""
    
    def compute_loss(self, pose_pred, pose, uncertainty_module=None):
        pred_se3 = pose_pred["se3"]  # shape: [B, 6]
        
        # Target: extract translation and convert rotation to rotation vector
        target_trans = pose[:, 0:3]  # [x, y, z]
        target_quat = pose[:, 3:]   # [qx, qy, qz, qw]
        target_rot_vec = quaternion_to_rotation_vector(target_quat)
        target_se3 = torch.cat([target_trans, target_rot_vec], dim=-1)  # [tx, ty, tz, ωx, ωy, ωz]
        
        # SE(3) direct loss - treats the complete pose uniformly
        loss = se3_direct_loss(pred_se3, target_se3)
        
        # For logging purposes, compute components
        loss_y = torch.nn.functional.mse_loss(pred_se3[:, 1:2], target_se3[:, 1:2])
        loss_z = torch.nn.functional.mse_loss(pred_se3[:, 2:3], target_se3[:, 2:3])
        loss_q = torch.norm(pred_se3[:, 3:6] - target_se3[:, 3:6], p=2, dim=-1).mean()
        loss_trans = torch.norm(pred_se3[:, 0:3] - target_se3[:, 0:3], p=2, dim=-1).mean()
        
        # Note: SE3 doesn't use uncertainty_module in the same way
        return loss, loss_q, loss_y, loss_z, loss_trans
    
    def compute_metrics(self, pose_pred, pose):
        pred_se3 = pose_pred["se3"]
        
        target_trans = pose[:, 0:3]
        target_quat = pose[:, 3:]
        target_rot_vec = quaternion_to_rotation_vector(target_quat)
        target_se3 = torch.cat([target_trans, target_rot_vec], dim=-1)
        
        abs_trans = torch.norm(pred_se3[:, 0:3] - target_se3[:, 0:3], dim=1).mean()
        abs_trans_y = torch.abs(pred_se3[:, 1] - target_se3[:, 1]).mean()
        abs_trans_z = torch.abs(pred_se3[:, 2] - target_se3[:, 2]).mean()
        
        # For rotation error, convert predicted rotation vector to quaternion and compare
        pred_rot_vec = pred_se3[:, 3:6]
        pred_quat = rotation_vector_to_quaternion(pred_rot_vec)
        pred_quat_norm = torch.nn.functional.normalize(pred_quat, p=2, dim=1, eps=1e-7)
        target_quat_norm = torch.nn.functional.normalize(target_quat, p=2, dim=1, eps=1e-7)
        
        # Extract roll for metric comparison
        pred_roll_vec = quaternion_to_2d_unit_vector(pred_quat_norm)
        target_roll_vec = quaternion_to_2d_unit_vector(target_quat_norm)
        pred_angle = torch.atan2(pred_roll_vec[:, 0], pred_roll_vec[:, 1])
        target_angle = torch.atan2(target_roll_vec[:, 0], target_roll_vec[:, 1])
        rot_diff = torch.abs(pred_angle - target_angle)
        rot_diff = torch.atan2(torch.sin(rot_diff), torch.cos(rot_diff))
        abs_rot_x = torch.rad2deg(torch.abs(rot_diff)).mean()
        
        return abs_trans_y, abs_trans_z, abs_trans, abs_rot_x

    def compute_rotation_errors_degrees(self, pose_pred, pose):
        pred_quat = torch.nn.functional.normalize(
            rotation_vector_to_quaternion(pose_pred["se3"][:, 3:6]), p=2, dim=1, eps=1e-7
        )
        target_quat = torch.nn.functional.normalize(pose[:, 3:], p=2, dim=1, eps=1e-7)
        return _roll_error_degrees(
            quaternion_to_2d_unit_vector(pred_quat), quaternion_to_2d_unit_vector(target_quat)
        )

    def compute_rotation_errors_degrees_signed(self, pose_pred, pose):
        pred_quat = torch.nn.functional.normalize(
            rotation_vector_to_quaternion(pose_pred["se3"][:, 3:6]), p=2, dim=1, eps=1e-7
        )
        target_quat = torch.nn.functional.normalize(pose[:, 3:], p=2, dim=1, eps=1e-7)
        return _roll_error_degrees_signed(
            quaternion_to_2d_unit_vector(pred_quat), quaternion_to_2d_unit_vector(target_quat)
        )


class QuaternionHandler(RotationLossHandler):
    """Handler for default quaternion rotation parametrisation."""
    
    def __init__(self, criterion):
        self.criterion = criterion
    
    def compute_loss(self, pose_pred, pose, uncertainty_module=None):
        base_loss, loss_q, loss_y, loss_z = self.criterion(pose, pose_pred["pose"])
        loss_trans = loss_y + loss_z
        if uncertainty_module is not None:
            loss = uncertainty_module(loss_q, loss_trans)
        else:
            loss = base_loss
        return loss, loss_q, loss_y, loss_z, loss_trans
    
    def compute_metrics(self, pose_pred, pose):
        abs_trans_y, abs_trans_z, abs_trans, abs_rot_x = absolute_pose_error_x_rot(pose_pred["pose"].detach(), pose)
        return abs_trans_y, abs_trans_z, abs_trans, abs_rot_x

    def compute_rotation_errors_degrees(self, pose_pred, pose):
        return _roll_error_degrees(
            quaternion_to_2d_unit_vector(pose_pred["rotation"]),
            quaternion_to_2d_unit_vector(pose[:, 3:]),
        )

    def compute_rotation_errors_degrees_signed(self, pose_pred, pose):
        return _roll_error_degrees_signed(
            quaternion_to_2d_unit_vector(pose_pred["rotation"]),
            quaternion_to_2d_unit_vector(pose[:, 3:]),
        )


def get_rotation_handler(rotation_parametrisation, criterion=None, rotation_loss_type="geodesic",
                          rot_scaling_factor=None):
    """Factory function to get the appropriate rotation loss handler.

    rot_scaling_factor: optional override for the handler's own hardcoded default
    (see "Unit balancing" in CLAUDE.md). None (unset, the default) preserves every
    existing config's behavior exactly -- each handler falls back to its own
    default (200.0, or 2000.0 for 2D_unit_vector) unchanged. The quaternion case
    doesn't take this directly (SE3DirectHandler doesn't use rot_scaling_factor at
    all); a configured value for it is instead baked into the passed-in `criterion`
    itself -- see TrainingConfigBuilder.build_loss_function.
    """
    kwargs = {} if rot_scaling_factor is None else {"rot_scaling_factor": rot_scaling_factor}
    if rotation_parametrisation == "2D_unit_vector":
        return TwoDUnitVectorHandler(rotation_loss_type=rotation_loss_type, **kwargs)
    elif rotation_parametrisation == "angle":
        return AngleHandler(**kwargs)
    elif rotation_parametrisation == "lie_algebra":
        return LieAlgebraHandler(**kwargs)
    elif rotation_parametrisation == "se3_direct":
        return SE3DirectHandler()
    else:
        return QuaternionHandler(criterion)


