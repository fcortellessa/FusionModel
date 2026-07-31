"""
Evaluate and compare roll error from quaternion vs 2D vector models.
Usage:
  python scripts/evaluate_roll_error.py --checkpoint path/to/checkpoint.pth --config configs/gated.yml
"""

import os
import sys
import argparse
import torch
import json
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from base_model.pose_estimation_model import PoseEstimationNet
from data_processing.dataloader_preprocessed import get_dataloaders
from losses.absolute_pose_error import absolute_pose_error_x_rot
from scripts.train import quaternion_to_2d_unit_vector
from utils.utils import load_config


def extract_roll_from_quat(quat):
    """
    Extract roll (x-rotation) from quaternion.
    Input: quat tensor of shape (B, 4) as [x, y, z, w]
    Output: roll angle in degrees
    """
    import torch.nn.functional as F
    
    q = F.normalize(quat, p=2, dim=1, eps=1e-7)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    return roll


def evaluate_roll(checkpoint_path, config_path, data_dir="Data/Data_split"):
    """
    Evaluate roll error from a trained model.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = PoseEstimationNet(
        model_name=cfg["model"]["backbone"],
        streams=cfg["model"]["streams"],
        fusion=cfg["model"]["fusion"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        rotation_parametrisation=cfg["model"]["head_type"],
        activation=cfg["model"]["activation"],
        rotation_layers=[cfg["model"]["mlp_quat_1"], cfg["model"]["mlp_quat_2"], cfg["model"]["mlp_quat_3"]],
        translation_layers=[cfg["model"]["mlp_trans_1"], cfg["model"]["mlp_trans_2"], cfg["model"]["mlp_trans_3"]],
        dropout_rot=cfg["training"]["dropout_rot"],
        dropout_trans=cfg["training"]["dropout_trans"],
        use_batchnorm=cfg["training"]["use_batchnorm"] 
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Model head_type: {cfg['model']['head_type']}")
    
    # Load data
    _, val_loader, _, _ = get_dataloaders(
        root_dir=data_dir,
        batch_size=16,
        shuffle=False
    )
    
    # Evaluate
    all_roll_errors = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            pose_target = batch["pose"]
            pose_pred, _ = model(batch)
            
            head_type = cfg["model"]["head_type"]
            
            if head_type == "2D_unit_vector":
                # Direct roll error from 2D predictions
                pred_vec = pose_pred["rotation"]
                target_vec = quaternion_to_2d_unit_vector(pose_target[:, 3:])
                
                pred_angle = torch.atan2(pred_vec[:, 0], pred_vec[:, 1])
                target_angle = torch.atan2(target_vec[:, 0], target_vec[:, 1])
                rot_diff = torch.abs(pred_angle - target_angle)
                rot_diff = torch.min(rot_diff, 2 * torch.pi - rot_diff)
                roll_error = torch.rad2deg(rot_diff)
                
            else:  # quaternion
                # Extract roll from quaternion predictions
                pred_roll = extract_roll_from_quat(pose_pred["rotation"])
                target_roll = extract_roll_from_quat(pose_target[:, 3:])
                
                roll_diff = torch.abs(pred_roll - target_roll)
                roll_diff = torch.min(roll_diff, 2 * torch.pi - roll_diff)
                roll_error = torch.rad2deg(roll_diff)
            
            all_roll_errors.append(roll_error.cpu().numpy())
    
    import numpy as np
    all_roll_errors = np.concatenate(all_roll_errors)
    mean_roll_error = all_roll_errors.mean()
    std_roll_error = all_roll_errors.std()
    median_roll_error = np.median(all_roll_errors)
    
    print(f"\n=== Roll Error Summary ===")
    print(f"Model type: {head_type}")
    print(f"Mean roll error: {mean_roll_error:.2f}°")
    print(f"Std roll error:  {std_roll_error:.2f}°")
    print(f"Median roll error: {median_roll_error:.2f}°")
    print(f"Min roll error: {all_roll_errors.min():.2f}°")
    print(f"Max roll error: {all_roll_errors.max():.2f}°")
    print(f"90th percentile: {np.percentile(all_roll_errors, 90):.2f}°")
    
    return {
        "mean": mean_roll_error,
        "std": std_roll_error,
        "median": median_roll_error,
        "min": all_roll_errors.min(),
        "max": all_roll_errors.max(),
        "p90": np.percentile(all_roll_errors, 90),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate roll error from a trained model checkpoint"
    )
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint (.pth file)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file used for training")
    parser.add_argument("--data_dir", type=str, default="Data/Data_split",
                        help="Path to data directory")
    
    args = parser.parse_args()
    
    results = evaluate_roll(args.checkpoint, args.config, args.data_dir)
