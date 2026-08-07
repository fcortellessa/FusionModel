#!/usr/bin/env python3
"""
Evaluate an Optuna study's winning trial against the held-out test split, and
report it next to the val-based score Optuna actually optimized against.

Why this exists: optuna_common.py's objective (score_final) only ever reads
val_abs_* metrics -- nothing in the phased Optuna pipeline (phase1a/1b/2/3)
touches Data/Data_split/test. That means a study can pick a "winner" whose val
score looks great without anyone checking whether it holds up on data the
model never influenced training-time decisions with. This script closes that
gap as a separate, deliberate step you run after a study finishes (or at any
point during it, on any completed trial).

The test-side score uses the exact same formula as optuna_common.score_final
(val_abs_rot_x in degrees + L2 norm of (abs_y, abs_z) in mm) so the two numbers
are directly comparable -- just swap "val" for "test" in your head.

Usage (auto-pick the study's best completed trial):
    python scripts/evaluate/specific_eval/eval_optuna_winner_on_test.py \\
        --study-dir runs/optuna_phase1b_head_architecture_grid10

Usage (a specific trial number instead of the best one):
    python scripts/evaluate/specific_eval/eval_optuna_winner_on_test.py \\
        --study-dir runs/optuna_phase1b_head_architecture_grid10 --trial 7

Usage (bypass the study dir entirely, point at any config/checkpoint pair):
    python scripts/evaluate/specific_eval/eval_optuna_winner_on_test.py \\
        --config path/to/config.yml --checkpoint path/to/checkpoint_best.pth

Output: prints a val-vs-test comparison to stdout, writes per-sample
predictions to <output-dir>/test_predictions.csv and a summary to
<output-dir>/summary.json (output-dir defaults to <trial_dir>/test_eval/).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator is found.*")
os.environ.setdefault("ALBUMENTATIONS_CHECK_FOR_UPDATES", "0")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import torch
import yaml

from base_model.pose_estimation_model import PoseEstimationNet
from data_processing.dataloader_preprocessed import (
    _load_or_compute_norm_stats,
    get_test_dataloader,
    tactile_collate_fn,
)
from data_processing.preprocess_data import ensure_test_split_current
from utils.normalization import canonicalize_translation_norm_stats
from utils.utils import get_optimizer, load_checkpoint, resolve_data_root


# ---------------------------------------------------------------------------
# Locating the trial to evaluate
# ---------------------------------------------------------------------------

def find_trial(study_dir: str, trial_number: int | None):
    """Return (trial_dir, config_path, checkpoint_path, recorded_val_score) for
    either the requested trial number or, if None, the study's best completed
    trial per trials.csv (the file optuna_common.run_study always writes)."""
    trials_csv = os.path.join(study_dir, "trials.csv")
    if not os.path.exists(trials_csv):
        raise FileNotFoundError(
            f"No trials.csv found at {trials_csv}. Point --study-dir at an "
            f"Optuna output-dir that has already run at least one trial, or "
            f"use --config/--checkpoint directly instead."
        )
    trials_df = pd.read_csv(trials_csv)
    complete = trials_df[trials_df["state"] == "COMPLETE"]
    if complete.empty:
        raise RuntimeError(f"No COMPLETE trials recorded in {trials_csv}")

    if trial_number is None:
        row = complete.loc[complete["value"].idxmin()]
    else:
        matches = complete[complete["number"] == trial_number]
        if matches.empty:
            raise RuntimeError(f"Trial {trial_number} not found (or not COMPLETE) in {trials_csv}")
        row = matches.iloc[0]

    number = int(row["number"])
    val_score = float(row["value"])
    trial_dir = os.path.join(study_dir, f"trial_{number:04d}")
    config_path = os.path.join(trial_dir, "config.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Trial {number}'s config not found at {config_path}")

    checkpoint_candidates = glob.glob(os.path.join(trial_dir, "runs", "*", "checkpoint_best.pth"))
    if not checkpoint_candidates:
        raise FileNotFoundError(f"No checkpoint_best.pth found under {trial_dir}/runs/*/")
    checkpoint_path = checkpoint_candidates[0]

    print(f"Selected trial {number} (val score={val_score:.4f} from trials.csv)")
    return trial_dir, config_path, checkpoint_path, val_score


# ---------------------------------------------------------------------------
# Model / inference
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    return PoseEstimationNet(
        model_name=config["model"]["backbone"],
        streams=config["model"]["streams"],
        fusion=config["model"]["fusion"],
        freeze_backbone=config["model"]["freeze_backbone"],
        rotation_parametrisation=config["model"]["head_type"],
        activation=config["model"]["activation"],
        rotation_layers=[config["model"]["mlp_quat_1"], config["model"]["mlp_quat_2"], config["model"]["mlp_quat_3"]],
        translation_layers=[config["model"]["mlp_trans_1"], config["model"]["mlp_trans_2"], config["model"]["mlp_trans_3"]],
        dropout_rot=config["training"]["dropout_rot"],
        dropout_trans=config["training"]["dropout_trans"],
        use_batchnorm=config["training"]["use_batchnorm"],
        dual_fusion=bool(config["model"].get("dual_fusion", False)),
        translation_head_mode=config["model"].get("translation_head_mode", "separate"),
        spatial_grid_size=config["model"].get("spatial_grid_size", (10, 10)),
    )


def resolve_translation_norm_stats(config: dict, data_root: str) -> dict:
    """Load the *actual* mean/std used at training time (cached alongside the
    split by preprocessing), not config['data']['translation_norm_stats'] --
    these Optuna-generated configs never carry that key, and
    canonicalize_translation_norm_stats() would silently fall back to the
    module-level placeholder mean=0/std=1 (a no-op "unnormalization"), which
    would badly understate translation error. Pull the real stats from
    normalization_stats.json instead, exactly as training/validation did."""
    train_raw_dir = os.path.join(data_root, "Data_split", "train")
    stats_path = os.path.join(data_root, "Data_split", "normalization_stats.json")
    raw_stats = _load_or_compute_norm_stats(stats_path, train_raw_dir)
    return canonicalize_translation_norm_stats(raw_stats)


def roll_x_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
    return np.degrees(np.arctan2(sinr_cosp, cosr_cosp))


def run_inference(model, test_loader, stats: dict, device: str) -> pd.DataFrame:
    model.eval()
    records = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pose = batch["pose"]
            model_out, _ = model({k: v for k, v in batch.items() if k != "pose"})
            pred_pose = model_out["pose"].clone()

            pred_pose[:, 1] = pred_pose[:, 1] * stats["std_y"] + stats["mean_y"]
            pred_pose[:, 2] = pred_pose[:, 2] * stats["std_z"] + stats["mean_z"]
            gt_pose = pose.clone()
            gt_pose[:, 1] = gt_pose[:, 1] * stats["std_y"] + stats["mean_y"]
            gt_pose[:, 2] = gt_pose[:, 2] * stats["std_z"] + stats["mean_z"]

            abs_y = (pred_pose[:, 1] - gt_pose[:, 1]).abs().cpu().numpy()
            abs_z = (pred_pose[:, 2] - gt_pose[:, 2]).abs().cpu().numpy()

            pred_q = pred_pose[:, 3:].cpu().numpy()
            gt_q = gt_pose[:, 3:].cpu().numpy()
            pred_q = pred_q / np.clip(np.linalg.norm(pred_q, axis=1, keepdims=True), 1e-8, None)
            gt_q = gt_q / np.clip(np.linalg.norm(gt_q, axis=1, keepdims=True), 1e-8, None)

            diff = np.abs(roll_x_deg(pred_q) - roll_x_deg(gt_q))
            rot_x_deg = np.minimum(diff, 360.0 - diff)

            for y, z, r in zip(abs_y, abs_z, rot_x_deg):
                records.append({"abs_y": float(y), "abs_z": float(z), "abs_rot_x": float(r)})

    return pd.DataFrame.from_records(records)


def combined_score(df: pd.DataFrame) -> float:
    """Same formula as optuna_common.score_final: val_abs_rot_x (deg) + L2 norm
    of (abs_y, abs_z) in mm, just computed over the whole test set at once
    instead of averaged over the last N logged epochs (there's only one
    checkpoint being evaluated here, so no epoch-averaging applies)."""
    trans_l2_mm = ((df["abs_y"].mean() * 1000) ** 2 + (df["abs_z"].mean() * 1000) ** 2) ** 0.5
    return float(df["abs_rot_x"].mean() + trans_l2_mm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-dir", type=str, default=None, help="Optuna output-dir, e.g. runs/optuna_phase1b_head_architecture_grid10")
    parser.add_argument("--trial", type=int, default=None, help="Specific trial number to evaluate (default: study's best COMPLETE trial)")
    parser.add_argument("--config", type=str, default=None, help="Explicit config.yml (bypasses --study-dir)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint (bypasses --study-dir)")
    parser.add_argument("--val-score", type=float, default=None, help="Val score to compare against, when using --config/--checkpoint directly (no trials.csv to read it from)")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to write test_predictions.csv/summary.json (default: <trial_dir>/test_eval/, or ./test_eval/ in direct mode)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gap-warn-pct", type=float, default=20.0, help="Flag a warning if test score is worse than val score by more than this percent")
    args = parser.parse_args()

    if args.config or args.checkpoint:
        if not (args.config and args.checkpoint):
            parser.error("--config and --checkpoint must be given together")
        config_path, checkpoint_path = args.config, args.checkpoint
        val_score = args.val_score
        output_dir = args.output_dir or "test_eval"
    else:
        if not args.study_dir:
            parser.error("Provide either --study-dir, or --config together with --checkpoint")
        trial_dir, config_path, checkpoint_path, val_score = find_trial(args.study_dir, args.trial)
        output_dir = args.output_dir or os.path.join(trial_dir, "test_eval")

    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Config:     {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {device}")

    data_root = resolve_data_root(config, REPO_ROOT)
    ensure_test_split_current(config, data_root)
    split_root = os.path.join(data_root, "Data_split")

    stats = resolve_translation_norm_stats(config, data_root)
    print(f"Translation norm stats (from normalization_stats.json): {stats}")

    test_loader = get_test_dataloader(root_dir=split_root, batch_size=args.batch_size, shuffle=False)

    model = build_model(config)
    optimizer = get_optimizer(model_or_params=model, optimizer_name=config["training"]["optimizer"], lr=config["training"]["lr"])
    load_checkpoint(model, optimizer, checkpoint_path, device="cpu", load_optimizer=False)
    model = model.to(device)

    df = run_inference(model, test_loader, stats, device)
    test_score = combined_score(df)

    predictions_path = os.path.join(output_dir, "test_predictions.csv")
    df.to_csv(predictions_path, index=False)

    summary = {
        "config_path": os.path.abspath(config_path),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "num_test_samples": len(df),
        "test_score": test_score,
        "test_abs_rot_x_deg_mean": float(df["abs_rot_x"].mean()),
        "test_abs_y_mm_mean": float(df["abs_y"].mean() * 1000),
        "test_abs_z_mm_mean": float(df["abs_z"].mean() * 1000),
        "val_score": val_score,
    }
    if val_score is not None:
        gap = test_score - val_score
        gap_pct = 100.0 * gap / val_score if val_score else float("nan")
        summary["val_test_gap"] = gap
        summary["val_test_gap_pct"] = gap_pct

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Test set (n={len(df)}): rot_x={summary['test_abs_rot_x_deg_mean']:.4f} deg, "
          f"y={summary['test_abs_y_mm_mean']:.4f} mm, z={summary['test_abs_z_mm_mean']:.4f} mm")
    print(f"Test score (same formula as Optuna's val objective): {test_score:.4f}")
    if val_score is not None:
        print(f"Val score (recorded during the study):              {val_score:.4f}")
        print(f"Gap (test - val): {gap:+.4f}  ({gap_pct:+.1f}%)")
        if gap_pct > args.gap_warn_pct:
            print(f"\n⚠️  Test score is more than {args.gap_warn_pct:.0f}% worse than the val score Optuna "
                  f"picked this trial on -- val is likely not representative of held-out performance.")
        elif gap_pct < -args.gap_warn_pct:
            print(f"\nNote: test score is more than {args.gap_warn_pct:.0f}% BETTER than val -- unusual; "
                  f"worth double-checking the test split isn't trivially easier than val by chance.")
        else:
            print(f"\nTest and val are within {args.gap_warn_pct:.0f}% of each other -- no large val/test gap.")
    print("=" * 72)
    print(f"\nWrote per-sample predictions to {predictions_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
