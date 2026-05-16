# test script
import time
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from base_model.pose_estimation_model import PoseEstimationNet
from data_processing.dataloader_preprocessed import get_test_dataloader
from utils.utils import load_checkpoint, load_config



model_path = "/home/MA_Flavia/Documents/FusionModel/checkpoints/checkpoint_concat_lr0.001_batch32_20260425_2042h.pth"
config_path = "configs/default.yml"
root_data_path = "Data/Data_split/"


def _quat_to_euler_xyz_deg(quat_xyzw):
    """
    Convert quaternions [x, y, z, w] to Euler angles (roll, pitch, yaw) in degrees.
    """
    q = quat_xyzw / np.clip(np.linalg.norm(quat_xyzw, axis=1, keepdims=True), 1e-12, None)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.rad2deg(np.stack([roll, pitch, yaw], axis=1))


def _angle_abs_diff_deg(a_deg, b_deg):
    """Shortest absolute angular distance in degrees."""
    return np.abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def save_evaluation_plots(pred_trans_all, pred_rot_all, gt_pose_all, results_dir, timestamp):
    gt_trans_all = gt_pose_all[:, :3]
    gt_rot_all = gt_pose_all[:, 3:3 + pred_rot_all.shape[1]]

    abs_trans = np.abs(pred_trans_all - gt_trans_all)
    use_xyz_rotation_errors = pred_rot_all.shape[1] == 4

    if use_xyz_rotation_errors:
        pred_euler_xyz = _quat_to_euler_xyz_deg(pred_rot_all)
        gt_euler_xyz = _quat_to_euler_xyz_deg(gt_rot_all)
        abs_rot_xyz = _angle_abs_diff_deg(pred_euler_xyz, gt_euler_xyz)
    else:
        abs_rot_xyz = np.abs(pred_rot_all - gt_rot_all)

    trans_errors = np.linalg.norm(pred_trans_all - gt_trans_all, axis=1)
    rot_errors = np.linalg.norm(pred_rot_all - gt_rot_all, axis=1)

    # Summary metrics
    summary = {
        "num_samples": int(len(trans_errors)),
        "trans_error_mean": float(np.mean(trans_errors)),
        "trans_error_median": float(np.median(trans_errors)),
        "trans_error_std": float(np.std(trans_errors)),
        "trans_abs_x_mae": float(np.mean(abs_trans[:, 0])),
        "trans_abs_y_mae": float(np.mean(abs_trans[:, 1])),
        "trans_abs_z_mae": float(np.mean(abs_trans[:, 2])),
        "rot_error_mean": float(np.mean(rot_errors)),
        "rot_error_median": float(np.median(rot_errors)),
        "rot_error_std": float(np.std(rot_errors)),
    }
    if use_xyz_rotation_errors:
        summary["rot_abs_x_deg_mae"] = float(np.mean(abs_rot_xyz[:, 0]))
        summary["rot_abs_y_deg_mae"] = float(np.mean(abs_rot_xyz[:, 1]))
        summary["rot_abs_z_deg_mae"] = float(np.mean(abs_rot_xyz[:, 2]))
    else:
        for i in range(abs_rot_xyz.shape[1]):
            summary[f"rot_abs_c{i}_mae"] = float(np.mean(abs_rot_xyz[:, i]))

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(results_dir, f"summary_{timestamp}.csv"), index=False)

    # 1) Per-sample error curves
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sample_idx = np.arange(len(trans_errors))
    ax1.plot(sample_idx, trans_errors, label="Translation L2 error", linewidth=1.5)
    ax1.plot(sample_idx, rot_errors, label="Rotation L2 error", linewidth=1.5)
    ax1.set_title("Per-sample prediction errors")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("L2 error")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(results_dir, f"errors_curve_{timestamp}.png"), dpi=180)
    plt.close(fig1)

    # 2) Error histograms
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(trans_errors, bins=30, alpha=0.8)
    axes[0].set_title("Translation error distribution")
    axes[0].set_xlabel("L2 error")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(rot_errors, bins=30, alpha=0.8, color="tab:orange")
    axes[1].set_title("Rotation error distribution")
    axes[1].set_xlabel("L2 error")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(results_dir, f"errors_hist_{timestamp}.png"), dpi=180)
    plt.close(fig2)

    # 3) Predicted vs GT translation components
    labels = ["x", "y", "z"]
    fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, ax in enumerate(axes):
        gt = gt_trans_all[:, i]
        pred = pred_trans_all[:, i]
        ax.scatter(gt, pred, s=12, alpha=0.65)
        min_v = min(gt.min(), pred.min())
        max_v = max(gt.max(), pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
        ax.set_title(f"Translation {labels[i]}: pred vs gt")
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Prediction")
        ax.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(results_dir, f"translation_pred_vs_gt_{timestamp}.png"), dpi=180)
    plt.close(fig3)

    # 4) Absolute translation component errors per sample
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    sample_idx = np.arange(abs_trans.shape[0])
    for i, lbl in enumerate(["|Δx|", "|Δy|", "|Δz|"]):
        ax4.plot(sample_idx, abs_trans[:, i], linewidth=1.1, label=lbl)
    ax4.set_title("Absolute translation errors per sample")
    ax4.set_xlabel("Sample index")
    ax4.set_ylabel("Absolute error")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(results_dir, f"abs_translation_curve_{timestamp}.png"), dpi=180)
    plt.close(fig4)

    # 5) Absolute rotation component errors per sample
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    if use_xyz_rotation_errors:
        rot_labels = ["|Δroll_x|", "|Δpitch_y|", "|Δyaw_z|"]
    else:
        rot_labels = [f"|Δr{i}|" for i in range(abs_rot_xyz.shape[1])]
    sample_idx = np.arange(abs_rot_xyz.shape[0])
    for i, lbl in enumerate(rot_labels):
        ax5.plot(sample_idx, abs_rot_xyz[:, i], linewidth=1.1, label=lbl)
    ax5.set_title("Absolute rotation errors per sample")
    ax5.set_xlabel("Sample index")
    ax5.set_ylabel("Absolute error (deg)" if use_xyz_rotation_errors else "Absolute error")
    ax5.grid(True, alpha=0.3)
    ax5.legend(ncol=max(1, min(4, len(rot_labels))))
    fig5.tight_layout()
    fig5.savefig(os.path.join(results_dir, f"abs_rotation_curve_{timestamp}.png"), dpi=180)
    plt.close(fig5)

    # 6) Absolute error distributions by component
    fig6, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].boxplot([abs_trans[:, 0], abs_trans[:, 1], abs_trans[:, 2]], labels=["|Δx|", "|Δy|", "|Δz|"])
    axes[0].set_title("Absolute translation error distribution")
    axes[0].set_ylabel("Absolute error")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].boxplot([abs_rot_xyz[:, i] for i in range(abs_rot_xyz.shape[1])], labels=rot_labels)
    axes[1].set_title("Absolute rotation error distribution")
    axes[1].set_ylabel("Absolute error (deg)" if use_xyz_rotation_errors else "Absolute error")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig6.tight_layout()
    fig6.savefig(os.path.join(results_dir, f"abs_errors_boxplot_{timestamp}.png"), dpi=180)
    plt.close(fig6)

    print(
        f"[INFO] Summary metrics saved. Mean translation error={summary['trans_error_mean']:.6f}, "
        f"mean rotation error={summary['rot_error_mean']:.6f}"
    )


if __name__ == "__main__":
    cfg = load_config(config_path)
    model = PoseEstimationNet(
        model_name=cfg["model"]["backbone"],
        streams=cfg["model"]["streams"],
        fusion=cfg["model"]["fusion"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        rotation_parametrisation=cfg["model"]["head_type"],
        activation=cfg["model"]["activation"],
        rotation_layers=[cfg["model"]["mlp_quat_1"], cfg["model"]["mlp_quat_2"], cfg["model"]["mlp_quat_3"]],
        translation_layers=[cfg["model"]["mlp_trans_1"], cfg["model"]["mlp_trans_2"],cfg["model"]["mlp_trans_3"]],
        dropout_rot=cfg["training"]["dropout_rot"],
        dropout_trans=cfg["training"]["dropout_trans"],
        use_batchnorm=cfg["training"]["use_batchnorm"]
    )
    # load state dict
    test_loader = get_test_dataloader(
        root_dir=root_data_path,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )
    load_checkpoint(model, checkpoint_path=model_path, device="cpu", load_optimizer=False)
    print("Model loaded successfully!")

    # load test data
    # test_data = torch.load(test_data_path)

    # run inference
    model.eval()
    csv_results = {
        "sample_id": [],
        "pred_translation": [],
        "pred_rotation": [],
        "pred_quaternion": [],
        "gt_pose": []
    }
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            inputs = {k: v for k, v in sample.items() if k != "pose"}
            target = sample["pose"]

            pred_dict, _ = model(inputs)
            pred_trans = pred_dict["trans"]
            pred_rot = pred_dict["rotation"]
            pred_pose = pred_dict["pose"]
            # Store results for CSV export
            csv_results["sample_id"].append(i)
            csv_results["pred_translation"].append(pred_trans.cpu().numpy())
            csv_results["pred_rotation"].append(pred_rot.cpu().numpy())
            csv_results["pred_quaternion"].append(pred_pose.cpu().numpy())
            csv_results["gt_pose"].append(target.cpu().numpy())

    # Save results to CSV
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

    df = pd.DataFrame(csv_results)
    df.to_csv(f"{results_dir}/results_{timestamp}.csv", index=False)

    # Create arrays for meaningful evaluation plots
    pred_trans_all = np.concatenate(csv_results["pred_translation"], axis=0)
    pred_rot_all = np.concatenate(csv_results["pred_rotation"], axis=0)
    gt_pose_all = np.concatenate(csv_results["gt_pose"], axis=0)

    save_evaluation_plots(
        pred_trans_all=pred_trans_all,
        pred_rot_all=pred_rot_all,
        gt_pose_all=gt_pose_all,
        results_dir=results_dir,
        timestamp=timestamp,
    )
