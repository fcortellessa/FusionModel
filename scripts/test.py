# test script

# usage: 
# python .\scripts\test.py --config runs\20260511_152356_bs16_gated\config.yml 
# --checkpoint runs\20260511_152356_bs16_gated\checkpoint_best.pth --results_dir runs\20260511_152356_bs16_gated\results
import argparse
import json
import time
import os
import sys
import warnings

# Prevent albumentations from trying to fetch version info and suppress pin_memory warning 
os.environ.setdefault("ALBUMENTATIONS_CHECK_FOR_UPDATES", "0")
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator is found.*")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from base_model.pose_estimation_model import PoseEstimationNet
from data_processing.dataloader_preprocessed import get_test_dataloader
from utils.utils import load_checkpoint, load_config, get_optimizer
from utils.train_utils import _build_named_param_groups


def _normalize_quaternions(quaternions):
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions / np.clip(norms, 1e-8, None)


def _quaternion_roll(quaternions):
    quaternions = _normalize_quaternions(quaternions)
    x = quaternions[:, 0]
    y = quaternions[:, 1]
    z = quaternions[:, 2]
    w = quaternions[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return np.arctan2(sinr_cosp, cosr_cosp)


def _quaternion_geodesic_error_deg(pred_quat, target_quat):
    pred_quat = _normalize_quaternions(pred_quat)
    target_quat = _normalize_quaternions(target_quat)
    dot = np.abs(np.sum(pred_quat * target_quat, axis=1))
    dot = np.clip(dot, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


config_path = "configs/gated.yml"
root_data_path = "Data/Data_split/"

def infer_and_evaluate(config, checkpoint_model, test_data, results_directory, device):
    model = PoseEstimationNet(
        model_name=config["model"]["backbone"],
        streams=config["model"]["streams"],
        fusion=config["model"]["fusion"],
        freeze_backbone=config["model"]["freeze_backbone"],
        rotation_parametrisation=config["model"]["head_type"],
        activation=config["model"]["activation"],
        rotation_layers=[config["model"]["mlp_quat_1"], config["model"]["mlp_quat_2"], config["model"]["mlp_quat_3"]],
        translation_layers=[config["model"]["mlp_trans_1"], config["model"]["mlp_trans_2"],config["model"]["mlp_trans_3"]],
        dropout_rot=config["training"]["dropout_rot"],
        dropout_trans=config["training"]["dropout_trans"],
        use_batchnorm=config["training"]["use_batchnorm"]
    )
    # load state dict
    test_loader = get_test_dataloader(
        root_dir=root_data_path,
        batch_size=config["training"]["batch_size"],
        shuffle=False # Typically False for testing
    )
    # Create a dummy optimizer for checkpoint loading (optimizer state not needed for inference)
    optimizer = get_optimizer(model_or_params=model, optimizer_name=config["training"]["optimizer"], lr=config["training"]["lr"])
    load_checkpoint(model, optimizer, model_path, device="cpu", load_optimizer=False)
    # load_checkpoint(model, checkpoint_path=model_path, device="cpu", load_optimizer=False)
    print("Model loaded successfully!")

    # run inference
    model.eval()
    
    # Store raw results efficiently across batches
    pred_translations = []
    pred_rotations = []
    pred_poses = []
    gt_poses = []
    
    with torch.no_grad():
        for sample in test_loader:
            inputs = {k: v for k, v in sample.items() if k != "pose"}
            target = sample["pose"] # Assumed shape [B, D]

            model_out, fusion_weights = model(inputs)
            trans = model_out["trans"]       # [B, 3]
            rotation = model_out["rotation"] # [B, 3] or [B, 4]
            
            pred_translations.append(trans.cpu().numpy())
            pred_rotations.append(rotation.cpu().numpy())
            pred_poses.append(model_out["pose"].cpu().numpy())
            gt_poses.append(target.cpu().numpy())

    # Flatten collected batches into single arrays
    pred_translations = np.concatenate(pred_translations, axis=0)
    pred_rotations = np.concatenate(pred_rotations, axis=0)
    pred_poses = np.concatenate(pred_poses, axis=0)
    gt_poses = np.concatenate(gt_poses, axis=0)
    
    # Pose tensors in this project are [x, y, z, qx, qy, qz, qw].
    gt_translations = gt_poses[:, :3]
    gt_rotations = gt_poses[:, 3:7]
    pred_rotations = pred_poses[:, 3:7]
    
    num_samples = gt_poses.shape[0]

    # Compute errors (fixing the data flow issue in original script)
    val_abs_trans_vec = np.abs(pred_translations - gt_translations)
    val_abs_trans = np.linalg.norm(val_abs_trans_vec, axis=1) # Vector magnitude
    val_abs_x = val_abs_trans_vec[:, 0]
    val_abs_y = val_abs_trans_vec[:, 1]
    val_abs_z = val_abs_trans_vec[:, 2]
    
    pred_roll = _quaternion_roll(pred_rotations)
    gt_roll = _quaternion_roll(gt_rotations)
    roll_diff = np.abs(pred_roll - gt_roll)
    roll_diff = np.minimum(roll_diff, 2.0 * np.pi - roll_diff)
    val_abs_rot_x = np.degrees(roll_diff)
    val_abs_rot_z = _quaternion_geodesic_error_deg(pred_rotations, gt_rotations)

    # Assemble results into a clean dictionary
    # Storing raw numpy arrays here will fail with DataFrame, convert to simple lists or keep as flat arrays
    csv_results = {
        "sample_id": np.arange(num_samples),
        # Store individual components for easy plotting/analysis
        "pred_trans_x": pred_translations[:, 0],
        "pred_trans_y": pred_translations[:, 1],
        "pred_trans_z": pred_translations[:, 2],
        "gt_trans_x": gt_translations[:, 0],
        "gt_trans_y": gt_translations[:, 1],
        "gt_trans_z": gt_translations[:, 2],
        # Mandatory columns needed by the original metrics code
        "val_abs_trans": val_abs_trans,
        "val_abs_x": val_abs_x,
        "val_abs_y": val_abs_y,
        "val_abs_z": val_abs_z,
        "val_abs_rot_x": val_abs_rot_x,
        "val_abs_rot_z": val_abs_rot_z,
    }

    # Save results to CSV
    df = pd.DataFrame(csv_results)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    csv_name = f"results_{timestamp}.csv"
    df.to_csv(os.path.join(results_dir, csv_name), index=False)
    print(f"Saved inference results to {os.path.join(results_dir, csv_name)}")

    # Compute metrics (best, worst, mean, variance) and save to a text file
    metrics_name = f"metrics_{timestamp}.txt"
    with open(os.path.join(results_dir, metrics_name), "w") as f:
        if "val_abs_trans" in df.columns:
            f.write(f"Final Val Translation Error: {df['val_abs_trans'].iloc[-1]:.6f}\n")
            f.write(f"Mean Val Translation Error: {df['val_abs_trans'].mean():.6f}\n")
            f.write(f"Variance Val Translation Error: {df['val_abs_trans'].var():.6f}\n")
            f.write(f"Best Val Translation Error: {df['val_abs_trans'].min():.6f}\n")
            f.write(f"Worst Val Translation Error: {df['val_abs_trans'].max():.6f}\n")
        if "val_abs_rot_x" in df.columns:
            f.write(f"Final Val Rotation Error around x (roll): {df['val_abs_rot_x'].iloc[-1]:.6f}\n")
            f.write(f"Mean Val Rotation Error around x (roll): {df['val_abs_rot_x'].mean():.6f}\n")
            f.write(f"Variance Val Rotation Error around x (roll): {df['val_abs_rot_x'].var():.6f}\n")
            f.write(f"Best Val Rotation Error around x (roll): {df['val_abs_rot_x'].min():.6f}\n")
            f.write(f"Worst Val Rotation Error around x (roll): {df['val_abs_rot_x'].max():.6f}\n")
        if "val_abs_y" in df.columns:
            f.write(f"Final Val Translation Error in y: {df['val_abs_y'].iloc[-1]:.6f}\n")
            f.write(f"Mean Val Translation Error in y: {df['val_abs_y'].mean():.6f}\n")
            f.write(f"Variance Val Translation Error in y: {df['val_abs_y'].var():.6f}\n")
            f.write(f"Best Val Translation Error in y: {df['val_abs_y'].min():.6f}\n")
            f.write(f"Worst Val Translation Error in y: {df['val_abs_y'].max():.6f}\n")
        if "val_abs_z" in df.columns:
            f.write(f"Final Val Translation Error in z: {df['val_abs_z'].iloc[-1]:.6f}\n")
            f.write(f"Mean Val Translation Error in z: {df['val_abs_z'].mean():.6f}\n")
            f.write(f"Variance Val Translation Error in z: {df['val_abs_z'].var():.6f}\n")
            f.write(f"Best Val Translation Error in z: {df['val_abs_z'].min():.6f}\n")
            f.write(f"Worst Val Translation Error in z: {df['val_abs_z'].max():.6f}\n")
        if "val_abs_rot_z" in df.columns:
            f.write(f"Final Val Rotation Error (geodesic): {df['val_abs_rot_z'].iloc[-1]:.6f}\n")
            f.write(f"Mean Val Rotation Error (geodesic): {df['val_abs_rot_z'].mean():.6f}\n")
            f.write(f"Variance Val Rotation Error (geodesic): {df['val_abs_rot_z'].var():.6f}\n")
            f.write(f"Best Val Rotation Error (geodesic): {df['val_abs_rot_z'].min():.6f}\n")
            f.write(f"Worst Val Rotation Error (geodesic): {df['val_abs_rot_z'].max():.6f}\n")
    print(f"Saved metrics to {os.path.join(results_dir, metrics_name)}")


def plot_results(results_directory, csv_results, config_path):
    print("Generating visualization plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Pose Estimation Test Results\nRun: {os.path.basename(config_path)}", fontsize=16)

    # Plot 1: Histogram of Y-Translation Errors
    ax = axs[0, 0]
    ax.hist(csv_results['val_abs_y'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(csv_results['val_abs_y'].mean(), color='red', linestyle='--', label=f'Mean: {csv_results["val_abs_y"].mean():.4f}')
    ax.set_title("Y-Translation Error Distribution")
    ax.set_xlabel("Absolute Error (meters)")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Plot 2: Histogram of Z-Translation Errors
    ax = axs[0, 1]
    ax.hist(csv_results['val_abs_z'], bins=30, color='#de8f05', edgecolor='black', alpha=0.7)
    ax.axvline(csv_results['val_abs_z'].mean(), color='red', linestyle='--', label=f'Mean: {csv_results["val_abs_z"].mean():.4f}')
    ax.set_title("Z-Translation Error Distribution")
    ax.set_xlabel("Absolute Error (meters)")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Plot 3: Histogram of Rotation Errors (Axis X)
    ax = axs[1, 0]
    ax.hist(csv_results['val_abs_rot_x'], bins=30, color='#029e73', edgecolor='black', alpha=0.7)
    ax.axvline(csv_results['val_abs_rot_x'].mean(), color='red', linestyle='--', label=f'Mean: {csv_results["val_abs_rot_x"].mean():.4f}')
    ax.set_title("Rotation Error Distribution (Axis X)")
    ax.set_xlabel("Absolute Error (degrees/radians)")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Plot 4: Boxplot of Translation Errors by Axis (Per-component analysis) and Rotation Error (geodesic)
    data_to_plot = [csv_results['val_abs_x'], csv_results['val_abs_y'], csv_results['val_abs_z'], csv_results['val_abs_rot_x']]
    
    ax_trans = axs[1, 1]
    ax_rot = ax_trans.twinx()
    ax_rot.grid(False) 

    # Plot Translation errors on the primary left axis
    bp_trans = ax_trans.boxplot([csv_results['val_abs_y'], csv_results['val_abs_z']], positions=[1, 2], widths=0.5, patch_artist=True, notch=True)
    bp_rot = ax_rot.boxplot([csv_results['val_abs_rot_x']], positions=[3], widths=0.5, patch_artist=True, notch=True)

    # --- Styling ---
    # Colors for: [Translation Y, Translation Z, Rotation X]
    colors = ['#3498db', '#de8f05', '#029e73']
    
    # Apply colors and styles to Translation boxes
    for patch, color in zip(bp_trans['boxes'], colors[0:2]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Apply color and style to Rotation box
    for patch in bp_rot['boxes']:
        patch.set_facecolor(colors[2])
        patch.set_alpha(0.6)

    # Uniform styling across all whiskers, caps, medians, and fliers
    for bp in [bp_trans, bp_rot]:
        for whisker in bp['whiskers']:
            whisker.set(color='#7f8c8d', linewidth=1.5, linestyle=":")
        for cap in bp['caps']:
            cap.set(color='#7f8c8d', linewidth=2)
        for median in bp['medians']:
            median.set(color='red', linewidth=2)
        for flier in bp['fliers']:
            flier.set(marker='o', color='#e74c3c', alpha=0.5)

    # --- Axis Limits & Labels ---
    # Set the x-axis limits to frame our 3 positions nicely
    ax_trans.set_xlim(0.5, 3.5)
    ax_trans.set_xticks([1, 2, 3])
    ax_trans.set_xticklabels(['Error Y', 'Error Z', 'Rot Error X'])
    
    # Title and separate Y labels
    ax_trans.set_title("Component Error Breakdown")
    ax_trans.set_ylabel("Translation Error (meters)", color='#3498db')
    ax_rot.set_ylabel("Rotation Error (degrees)", color='#029e73')
    
    # Optional: Color the axis tick labels to match the data
    ax_trans.tick_params(axis='y', labelcolor='#3498db')
    ax_rot.tick_params(axis='y', labelcolor='#029e73')

    # --- Save & Finalize ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to not cut off title

    # Save the figure
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    plot_name = f"summary_plots_{timestamp}.png"
    plt.savefig(os.path.join(results_directory, plot_name), dpi=150)
    print(f"Saved visualization plots to {os.path.join(results_directory, plot_name)}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default=None, help="Path to a structured run folder containing config, checkpoints, etc.")
    # folder or each config and checkpoint specified separately
    parser.add_argument("--config", type=str, default=config_path, help="Path to the training config used for the run")
    parser.add_argument("--run_dir", type=str, default=None, help="Specific run directory to load instead of the latest run")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path to load")
    parser.add_argument("--results_dir", type=str, default=None, help="Override the directory where test CSVs are saved")
    
    args = parser.parse_args()

    # --run_folder input
    if args.run_folder is not None:
        manifest_path = os.path.join(args.run_folder, "manifest.json")
        
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            # Fallback layout mapping based on your structure
            manifest = {
                "config_snapshot_path": os.path.join(args.run_folder, "config.yml"),
                "checkpoint_path": os.path.join(args.run_folder, "checkpoint_best.pth"),
                "results_dir": os.path.join(args.run_folder, "results"),
            }

    # separate inputs for config and checkpoint
    elif args.run_dir is not None:
        manifest_path = os.path.join(args.run_dir, "manifest.json")
        legacy_manifest_path = os.path.join(args.run_dir, "run_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        elif os.path.exists(legacy_manifest_path):
            with open(legacy_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {
                "config_snapshot_path": os.path.join(args.run_dir, "config.yml"),
                "checkpoint_path": os.path.join(args.run_dir, "checkpoint_best.pth"),
                "run_dir": args.run_dir,
                "results_dir": os.path.join(args.run_dir, "results"),
            }

            if not os.path.exists(manifest["config_snapshot_path"]):
                manifest["config_snapshot_path"] = os.path.join(args.run_dir, "config.resolved.yml")
            if not os.path.exists(manifest["checkpoint_path"]):
                manifest["checkpoint_path"] = os.path.join(args.run_dir, "checkpoints", "best.pth")
    else:
        if args.checkpoint is None:
            raise ValueError("Provide either --run_folder, --run_dir, or an explicit --checkpoint.")
        manifest = {
            "config_snapshot_path": args.config,
            "checkpoint_path": args.checkpoint,
            "results_dir": "results",
        }

    # Execute inference and evaluation
    cfg_path = manifest.get("config_snapshot_path") or args.config
    cfg = load_config(cfg_path)
    model_path = args.checkpoint or manifest["checkpoint_path"]
    results_dir = args.results_dir or manifest.get("results_dir") or os.path.join(os.path.dirname(model_path), "..", "results")
    
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    infer_and_evaluate(cfg, model_path, None, results_dir, device)
    
    plot_results(results_dir, pd.read_csv(os.path.join(results_dir, os.listdir(results_dir)[-1])), cfg_path)
