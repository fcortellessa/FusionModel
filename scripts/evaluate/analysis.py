#!/usr/bin/env python3
"""
Universal Training & Ablation Analysis Pipeline.
Supports both individual run diagnostic reporting and multi-run directory benchmarking.
"""

import os
import glob
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from pathlib import Path

def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flattens nested configuration blocks, mapping lists to hashable tuples."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            tuple_val = tuple(flatten_dict(x) if isinstance(x, dict) else x for x in v)
            items.append((new_key, tuple_val))
        else:
            items.append((new_key, v))
    return dict(items)


def format_metadata_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return int(value)
    return value


def get_run_metadata_text(frame):
    metadata_fields = ["fusion", "head_type", "batch_size"]
    parts = []
    for field in metadata_fields:
        if field not in frame.columns:
            continue
        series = frame[field].dropna()
        if series.empty:
            continue
        value = format_metadata_value(series.iloc[0])
        if value is not None:
            parts.append(f"{field}={value}")
    return " | ".join(parts) if parts else None


def add_run_metadata(fig, frame):
    metadata_text = get_run_metadata_text(frame)
    if metadata_text:
        fig.text(0.01, 0.01, metadata_text, ha="left", va="bottom", fontsize=8, alpha=0.75)


def plot_train_val(df, ax, epoch_col, train_col, val_col, title, ylabel, train_label=None, val_label=None, train_color="#1f77b4", val_color="#ff7f0e"):
    if train_col in df.columns:
        ax.plot(df[epoch_col], df[train_col], color=train_color, label=train_label or f"Train {train_col}")
    if val_col in df.columns:
        ax.plot(df[epoch_col], df[val_col], color=val_color, label=val_label or f"Val {val_col}")
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)


def maybe_parse_fusion_weights(value):
    if pd.isna(value) or value == "":
        return None
    try:
        parsed = literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        return parsed
    return None


def analyze_single_run(df, output_dir: Path):
    """Generates exhaustive diagnostic plots for an isolated execution run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. TRAIN VS VAL LOSS
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_train_val(df, ax, "epoch", "train_loss", "val_loss", "Train vs Validation Loss Convergence", "Loss Value")
    if "train_loss" in df.columns:
        best_idx = df["train_loss"].idxmin()
        ax.annotate(f"Best Train: {df.loc[best_idx, 'train_loss']:.3f}", 
                    (df.loc[best_idx, "epoch"], df.loc[best_idx, "train_loss"]),
                    xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->", alpha=0.5))
    ax.set_xlabel("Epoch")
    add_run_metadata(fig, df)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_convergence.png", dpi=150)
    plt.close()

    # 2. OVERALL TRANSLATION AND ROTATION ERRORS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plot_train_val(df, ax1, "epoch", "train_abs_trans", "val_abs_trans", "Total Translation Error Magnitude", "Error [m or mm]")
    plot_train_val(df, ax2, "epoch", "train_abs_rot_x", "val_abs_rot_x", "Orientation Error Profile (X-axis)", "Error [deg]",
                   train_color="#e377c2", val_color="#d62728")
    ax2.set_xlabel("Epoch")
    add_run_metadata(fig, df)
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_errors_total.png", dpi=150)
    plt.close()

    # 3. PER-AXIS TRANSLATION ANALYSIS
    if any(col in df.columns for col in ["train_abs_y", "val_abs_y", "train_abs_z", "val_abs_z"]):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        plot_train_val(df, ax1, "epoch", "train_abs_y", "val_abs_y", "Y-Axis Alignment Error", "Abs Error [m or mm]")
        plot_train_val(df, ax2, "epoch", "train_abs_z", "val_abs_z", "Z-Axis Alignment Error", "Abs Error [m or mm]", 
                       train_color="#2ca02c", val_color="#9467bd")
        ax2.set_xlabel("Epoch")
        add_run_metadata(fig, df)
        plt.tight_layout()
        plt.savefig(output_dir / "trajectory_errors_axes.png", dpi=150)
        plt.close()

    # Dedicated Plot: ROT_X Train vs Val (Same color, Train lower alpha)
    if "train_abs_rot_x" in df.columns or "val_abs_rot_x" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        if "train_abs_rot_x" in df.columns:
            ax.plot(df["epoch"], df["train_abs_rot_x"], color="#d62728", alpha=0.4, linewidth=2, label="Train Rotation X")
        if "val_abs_rot_x" in df.columns:
            ax.plot(df["epoch"], df["val_abs_rot_x"], color="#d62728", alpha=1.0, linewidth=2, label="Val Rotation X")
        ax.set_title("Rotation X: Train vs Validation Performance", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error [deg]")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)
        add_run_metadata(fig, df)
        plt.tight_layout()
        plt.savefig(output_dir / "metric_compare_rot_x.png", dpi=150)
        plt.close()

    # Dedicated Plot: ABS_Y Train vs Val (Same color, Train lower alpha)
    if "train_abs_y" in df.columns or "val_abs_y" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        if "train_abs_y" in df.columns:
            ax.plot(df["epoch"], df["train_abs_y"], color="#1f77b4", alpha=0.4, linewidth=2, label="Train Abs Y")
        if "val_abs_y" in df.columns:
            ax.plot(df["epoch"], df["val_abs_y"], color="#1f77b4", alpha=1.0, linewidth=2, label="Val Abs Y")
        ax.set_title("Y-Axis Translation: Train vs Validation Performance", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error [m or mm]")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)
        add_run_metadata(fig, df)
        plt.tight_layout()
        plt.savefig(output_dir / "metric_compare_abs_y.png", dpi=150)
        plt.close()

    # Dedicated Plot: ABS_Z Train vs Val (Same color, Train lower alpha)
    if "train_abs_z" in df.columns or "val_abs_z" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        if "train_abs_z" in df.columns:
            ax.plot(df["epoch"], df["train_abs_z"], color="#2ca02c", alpha=0.4, linewidth=2, label="Train Abs Z")
        if "val_abs_z" in df.columns:
            ax.plot(df["epoch"], df["val_abs_z"], color="#2ca02c", alpha=1.0, linewidth=2, label="Val Abs Z")
        ax.set_title("Z-Axis Translation: Train vs Validation Performance", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error [m or mm]")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)
        add_run_metadata(fig, df)
        plt.tight_layout()
        plt.savefig(output_dir / "metric_compare_abs_z.png", dpi=150)
        plt.close()

    # 4. TASK UNCERTAINTY DYNAMIC WEIGHTS
    weight_cols = [col for col in ["uncertainty_weight_rot", "uncertainty_weight_trans"] if col in df.columns]
    if weight_cols:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for col in weight_cols:
            ax.plot(df["epoch"], df[col], label=col.replace("uncertainty_weight_", "").upper() + " Task Weight", linewidth=2)
        ax.set_title("Homoscedastic Multi-Task Loss Weight Balancing", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dynamic Loss Scaling Modifier Weights")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        add_run_metadata(fig, df)
        plt.tight_layout()
        plt.savefig(output_dir / "homoscedastic_uncertainty_weights.png", dpi=150)
        plt.close()

    # 5. GENERALIZATION LOSS RATIO PROFILE
    if "train_loss" in df.columns and "val_loss" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ratio = df["val_loss"] / df["train_loss"]
        ax.plot(df["epoch"], ratio, color="#7f7f7f", linewidth=2, label=r"Generalization Index Ratio ($\frac{\mathcal{L}_{val}}{\mathcal{L}_{train}}$)")
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Ideal Equivalence Line')
        ax.set_title("Generalization Gap Index Profile", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Ratio Scale Metric")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        add_run_metadata(fig, df)
        plt.tight_layout()
        plt.savefig(output_dir / "generalization_loss_ratio.png", dpi=150)
        plt.close()

    print(f"Diagnostics safely generated inside directory: {output_dir}")


def analyze_folder_ablation(runs_dir: Path, output_dir: Path):
    """Finds all child runs, isolates configuration edits, and plots comparative curves."""
    metrics_files = glob.glob(os.path.join(str(runs_dir), "*/metrics.csv"))
    if not metrics_files:
        print(f"Error: No run configurations holding metrics.csv discovered inside {runs_dir}")
        return

    all_histories = []
    all_configs = {}

    for path in metrics_files:
        run_name = os.path.basename(os.path.dirname(path))
        config_path = Path(os.path.dirname(path)) / "config.yml"
        
        flat_config = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    flat_config = flatten_dict(yaml.safe_load(f) or {})
            except Exception as e:
                print(f"Warning: Configuration extraction failure on run '{run_name}': {e}")
        all_configs[run_name] = flat_config

        try:
            run_df = pd.read_csv(path)
            if not run_df.empty:
                run_df["run_name"] = run_name
                all_histories.append(run_df)
        except Exception as e:
            print(f"Error reading metrics matrix for run folder '{run_name}': {e}")

    if not all_histories:
        return

    config_df = pd.DataFrame.from_dict(all_configs, orient='index')
    ignore_keys = ["data.augmentation.seed", "training.uncertainty_weighting.init_log_var_rot", "training.uncertainty_weighting.init_log_var_trans", "seed"]
    config_df = config_df.drop(columns=[c for c in ignore_keys if c in config_df.columns], errors='ignore')
    
    unique_counts = config_df.nunique()
    ablated_keys = unique_counts[unique_counts > 1].index.tolist()

    # Form labels dynamically based on what was changed
    variant_mapping = {}
    for run_name, cfg in all_configs.items():
        if ablated_keys:
            variant_mapping[run_name] = " | ".join([f"{k}={cfg.get(k, 'N/A')}" for k in ablated_keys])
        else:
            variant_mapping[run_name] = run_name

    master_history = pd.concat(all_histories, ignore_index=True)
    master_history["variant"] = master_history["run_name"].map(variant_mapping)

    output_dir.mkdir(parents=True, exist_ok=True)
    variants = master_history['variant'].unique()
    cmap = plt.get_cmap('tab10')
    colors = {v: cmap(i % 10) for i, v in enumerate(variants)}

    metrics_to_benchmark = [col for col in ['train_loss', 'epoch_time_sec', 'val_abs_rot_x', 'val_abs_y', 'val_abs_z'] if col in master_history.columns]

    for metric in metrics_to_benchmark:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for variant in variants:
            v_df = master_history[master_history['variant'] == variant].dropna(subset=[metric])
            if v_df.empty:
                continue
            
            stats = v_df.groupby('epoch')[metric].agg(['mean', 'std']).reset_index()
            lbl = variant if len(variant) <= 55 else variant[:52] + "..."
            
            ax.plot(stats['epoch'], stats['mean'], label=lbl, color=colors[variant], linewidth=2)
            if 'std' in stats.columns and stats['std'].notna().any():
                ax.fill_between(stats['epoch'], stats['mean'] - stats['std'], stats['mean'] + stats['std'], color=colors[variant], alpha=0.15)

        if "loss" in metric:
            unit_str = "Value"
        elif "time" in metric:
            unit_str = "Seconds"
        elif "rot_x" in metric:
            unit_str = "deg"
        else:
            unit_str = "m or mm"

        ax.set_title(f"Ablation Performance Metric: Comparison of {metric}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Scale Measure [{unit_str}]")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        
        plt.savefig(output_dir / f"ablation_comparison_{metric}.png", dpi=150)
        plt.close()
        
    print(f"Multi-run ablation bench plots saved inside target folder: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Run & Multi-Run Performance Evaluator Engine")
    parser.add_argument("--path", type=str, required=True, help="Target path: Can be an explicit path to a single metrics.csv file, a run folder, or a parent multi-run directory.")
    parser.add_argument("--nested", action="store_true", help="If set and --path is an ablation folder, runs diagnostic analysis for each sub-run folder sequentially.")
    args = parser.parse_args()

    target_path = Path(args.path)

    if target_path.is_file() and target_path.suffix == '.csv':
        df = pd.read_csv(target_path)
        analyze_single_run(df, output_dir=target_path.parent / "diagnostics")
    elif target_path.is_dir():
        metrics_check = target_path / "metrics.csv"
        if metrics_check.exists():
            df = pd.read_csv(metrics_check)
            analyze_single_run(df, output_dir=target_path / "diagnostics")
        else:
            # First, complete the core multi-run comparison logic
            analyze_folder_ablation(target_path, output_dir=target_path / "results" / "plots")
            
            # If the user toggled --nested, loop over individual run folders inside
            if args.nested:
                print("\n--- Starting Nested Subfolder Diagnostics Reporting ---")
                metrics_files = glob.glob(os.path.join(str(target_path), "*/metrics.csv"))
                for path_str in metrics_files:
                    csv_path = Path(path_str)
                    run_folder_name = csv_path.parent.name
                    print(f"Processing isolated diagnostics for run variant: {run_folder_name}")
                    
                    sub_df = pd.read_csv(csv_path)
                    if not sub_df.empty:
                        analyze_single_run(sub_df, output_dir=csv_path.parent / "diagnostics")
    else:
        print(f"Error: Specified target input reference route does not exist: {target_path}")