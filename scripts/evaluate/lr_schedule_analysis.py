#!/usr/bin/env python3
"""
LR Schedule Effect Analysis Pipeline
Compares global vs per-head learning rate schedules and their impact on:
1. Convergence Behavior (val loss, per-task errors over epochs)
2. Learning Dynamics (per-head parameter updates, gradient flow)
3. Task-Specific Adaptation (rotation vs translation error divergence)
4. Stability Across Seeds (mean/std of convergence metrics)
"""

import os
import glob
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def extract_lr_schedule_config(run_name: str, cfg: dict) -> Tuple[str, str, int]:
    """
    Extract: (schedule_type, scope, seed)

    Returns:
        (schedule_type, scope, seed)
        e.g., ("reduce_on_plateau", "global", 46)
        e.g., ("warmup_cosine", "per_head", 47)
    """
    seed = cfg.get("training.seed", "?")

    # Determine schedule type
    lr_schedule_enabled = cfg.get("training.lr_schedule.enabled", False)
    if not lr_schedule_enabled:
        return ("none", "none", seed)

    schedule_type = cfg.get("training.lr_schedule.type", "none")

    # Determine scope (global vs per-head)
    head_lr_decay_enabled = cfg.get("training.head_lr_decay.enabled", False)
    scope = "per_head" if head_lr_decay_enabled else "global"

    return (schedule_type, scope, seed)


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flattens nested configuration blocks."""
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


def compute_task_divergence(row: pd.Series) -> float:
    """
    Compute divergence between rotation and translation error trends.
    Measures if one task is learning faster than the other.
    Returns the absolute difference in normalized errors.
    """
    rot_cols = [c for c in row.index if 'rot' in c.lower() and 'abs' in c.lower()]
    trans_cols = [c for c in row.index if ('_y' in c.lower() or '_z' in c.lower()) and 'abs' in c.lower()]

    if not rot_cols or not trans_cols:
        return np.nan

    rot_val = row[rot_cols[0]] if pd.notna(row[rot_cols[0]]) else np.nan
    trans_val = row[trans_cols[0]] if pd.notna(row[trans_cols[0]]) else np.nan

    if pd.isna(rot_val) or pd.isna(trans_val):
        return np.nan

    # Normalize to [0, 1] range for comparison (rough normalization)
    # Rotation: typically 0-180 degrees, translation: typically 0-0.1 meters
    normalized_rot = min(rot_val / 180.0, 1.0)
    normalized_trans = min(trans_val / 0.1, 1.0)

    divergence = abs(normalized_rot - normalized_trans)
    return divergence


def load_and_prepare_data(runs_dir: Path) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Load all metrics.csv files and extract LR schedule configuration.

    Returns:
        (master_df, run_configs)
    """
    metrics_files = glob.glob(os.path.join(str(runs_dir), "*/metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.csv files found in {runs_dir}")

    all_dfs = []
    run_configs = {}

    for path in metrics_files:
        run_name = os.path.basename(os.path.dirname(path))
        config_path = Path(os.path.dirname(path)) / "config.yml"

        cfg = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg = flatten_dict(yaml.safe_load(f) or {})
            except Exception as e:
                print(f"Warning: Failed to load config for {run_name}: {e}")

        run_configs[run_name] = cfg

        try:
            df = pd.read_csv(path)
            if not df.empty:
                df["run_name"] = run_name
                all_dfs.append(df)
        except Exception as e:
            print(f"Error reading metrics for {run_name}: {e}")

    if not all_dfs:
        raise ValueError("No valid metrics files loaded")

    master_df = pd.concat(all_dfs, ignore_index=True)

    # Add LR schedule classification
    master_df["schedule_type"] = master_df["run_name"].apply(
        lambda rn: extract_lr_schedule_config(rn, run_configs.get(rn, {}))[0]
    )
    master_df["schedule_scope"] = master_df["run_name"].apply(
        lambda rn: extract_lr_schedule_config(rn, run_configs.get(rn, {}))[1]
    )
    master_df["seed"] = master_df["run_name"].apply(
        lambda rn: extract_lr_schedule_config(rn, run_configs.get(rn, {}))[2]
    )

    # Create a composite schedule label for easier grouping
    master_df["schedule_label"] = master_df.apply(
        lambda row: f"{row['schedule_type'].upper()}\n({row['schedule_scope'].upper()})" if row['schedule_type'] != 'none' else "NO SCHEDULE",
        axis=1
    )

    # Compute task divergence for each epoch
    master_df["task_divergence"] = master_df.apply(compute_task_divergence, axis=1)

    return master_df, run_configs


def plot_convergence_comparison(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Compare validation loss convergence across LR schedule strategies.
    Separate subplots for each schedule type, colored by scope (global vs per-head).
    Creates both Plotly (interactive) and matplotlib (static) versions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted([s for s in master_df["schedule_type"].unique() if s != "none"])
    if not schedule_types:
        print("Warning: No LR schedules found to compare")
        return

    # ===== MATPLOTLIB STATIC PLOT =====
    fig, axes = plt.subplots(1, len(schedule_types), figsize=(6 * len(schedule_types), 5))
    if len(schedule_types) == 1:
        axes = [axes]

    colors = {"global": "#1f77b4", "per_head": "#ff7f0e"}

    for idx, sched_type in enumerate(schedule_types):
        ax = axes[idx]

        # Plot both global and per-head variants
        for scope in ["global", "per_head"]:
            subset = master_df[
                (master_df["schedule_type"] == sched_type) &
                (master_df["schedule_scope"] == scope)
            ]

            if subset.empty:
                continue

            # Group by epoch and compute mean/std loss
            if "val_loss" in subset.columns:
                loss_by_epoch = subset.groupby("epoch")["val_loss"].agg(['mean', 'std']).reset_index()

                ax.plot(loss_by_epoch["epoch"], loss_by_epoch["mean"],
                       label=f"{scope.title()} Scope", linewidth=2.5, marker='o',
                       markersize=4, color=colors[scope])

                if 'std' in loss_by_epoch.columns and loss_by_epoch['std'].notna().any():
                    ax.fill_between(
                        loss_by_epoch["epoch"],
                        loss_by_epoch["mean"] - loss_by_epoch["std"],
                        loss_by_epoch["mean"] + loss_by_epoch["std"],
                        alpha=0.2, color=colors[scope]
                    )

        ax.set_title(f"Val Loss: {sched_type.upper()}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f"convergence_comparison.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()

    # ===== PLOTLY INTERACTIVE VERSION =====
    if PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=1, cols=len(schedule_types),
            subplot_titles=[f"{s.upper()}" for s in schedule_types],
            specs=[[{"secondary_y": False}] * len(schedule_types)]
        )

        for idx, sched_type in enumerate(schedule_types):
            for scope in ["global", "per_head"]:
                subset = master_df[
                    (master_df["schedule_type"] == sched_type) &
                    (master_df["schedule_scope"] == scope)
                ]

                if subset.empty:
                    continue

                if "val_loss" in subset.columns:
                    loss_by_epoch = subset.groupby("epoch")["val_loss"].agg(['mean', 'std']).reset_index().sort_values("epoch")

                    fig.add_trace(
                        go.Scatter(
                            x=loss_by_epoch["epoch"],
                            y=loss_by_epoch["mean"],
                            mode='lines+markers',
                            name=f"{scope.title()} {sched_type.upper()}",
                            line=dict(color=colors[scope], width=2.5),
                            error_y=dict(
                                type='data',
                                array=loss_by_epoch['std'],
                                visible=True
                            ),
                            showlegend=(idx==0)
                        ),
                        row=1, col=idx+1
                    )

        fig.update_layout(
            title="Validation Loss Convergence: Global vs Per-Head LR Schedules",
            height=500,
            template="plotly_white",
            hovermode='x unified'
        )
        fig.update_yaxes(title_text="Loss Value", row=1, col=1)
        fig.update_xaxes(title_text="Epoch")

        fig.write_html(str(output_dir / "convergence_comparison_interactive.html"), include_plotlyjs='cdn')
        print(f"✓ Interactive Plotly convergence plot saved: {output_dir / 'convergence_comparison_interactive.html'}")

    print(f"✓ Convergence comparison plots saved to {output_dir}")


def plot_task_specific_errors(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Compare rotation vs translation error trajectories across LR schedules.
    Goal: see if per-head scheduling helps one task converge faster than the other.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted([s for s in master_df["schedule_type"].unique() if s != "none"])
    if not schedule_types:
        return

    colors = {"global": "#1f77b4", "per_head": "#ff7f0e"}

    # ===== ROTATION ERROR =====
    fig, axes = plt.subplots(1, len(schedule_types), figsize=(6 * len(schedule_types), 5))
    if len(schedule_types) == 1:
        axes = [axes]

    for idx, sched_type in enumerate(schedule_types):
        ax = axes[idx]

        for scope in ["global", "per_head"]:
            subset = master_df[
                (master_df["schedule_type"] == sched_type) &
                (master_df["schedule_scope"] == scope)
            ]

            if subset.empty or "val_abs_rot_x" not in subset.columns:
                continue

            rot_by_epoch = subset.groupby("epoch")["val_abs_rot_x"].agg(['mean', 'std']).reset_index()

            ax.plot(rot_by_epoch["epoch"], rot_by_epoch["mean"],
                   label=f"{scope.title()}", linewidth=2.5, marker='o',
                   markersize=4, color=colors[scope])

            if 'std' in rot_by_epoch.columns and rot_by_epoch['std'].notna().any():
                ax.fill_between(
                    rot_by_epoch["epoch"],
                    rot_by_epoch["mean"] - rot_by_epoch["std"],
                    rot_by_epoch["mean"] + rot_by_epoch["std"],
                    alpha=0.2, color=colors[scope]
                )

        ax.set_title(f"Rotation Error: {sched_type.upper()}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Absolute Error [deg]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f"task_errors_rotation.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()

    # ===== TRANSLATION ERROR (Y + Z) =====
    fig, axes = plt.subplots(2, len(schedule_types), figsize=(6 * len(schedule_types), 8))
    if len(schedule_types) == 1:
        axes = axes.reshape(-1, 1)

    for idx, sched_type in enumerate(schedule_types):
        # Y axis
        ax_y = axes[0, idx]

        for scope in ["global", "per_head"]:
            subset = master_df[
                (master_df["schedule_type"] == sched_type) &
                (master_df["schedule_scope"] == scope)
            ]

            if subset.empty or "val_abs_y" not in subset.columns:
                continue

            trans_by_epoch = subset.groupby("epoch")["val_abs_y"].agg(['mean', 'std']).reset_index()

            ax_y.plot(trans_by_epoch["epoch"], trans_by_epoch["mean"],
                     label=f"{scope.title()}", linewidth=2.5, marker='o',
                     markersize=4, color=colors[scope])

            if 'std' in trans_by_epoch.columns and trans_by_epoch['std'].notna().any():
                ax_y.fill_between(
                    trans_by_epoch["epoch"],
                    trans_by_epoch["mean"] - trans_by_epoch["std"],
                    trans_by_epoch["mean"] + trans_by_epoch["std"],
                    alpha=0.2, color=colors[scope]
                )

        ax_y.set_title(f"Y Translation Error: {sched_type.upper()}", fontsize=11, fontweight='bold')
        ax_y.set_ylabel("Absolute Error [m]")
        ax_y.grid(True, alpha=0.3)
        ax_y.legend(fontsize=9)

        # Z axis
        ax_z = axes[1, idx]

        for scope in ["global", "per_head"]:
            subset = master_df[
                (master_df["schedule_type"] == sched_type) &
                (master_df["schedule_scope"] == scope)
            ]

            if subset.empty or "val_abs_z" not in subset.columns:
                continue

            trans_by_epoch = subset.groupby("epoch")["val_abs_z"].agg(['mean', 'std']).reset_index()

            ax_z.plot(trans_by_epoch["epoch"], trans_by_epoch["mean"],
                     label=f"{scope.title()}", linewidth=2.5, marker='o',
                     markersize=4, color=colors[scope])

            if 'std' in trans_by_epoch.columns and trans_by_epoch['std'].notna().any():
                ax_z.fill_between(
                    trans_by_epoch["epoch"],
                    trans_by_epoch["mean"] - trans_by_epoch["std"],
                    trans_by_epoch["mean"] + trans_by_epoch["std"],
                    alpha=0.2, color=colors[scope]
                )

        ax_z.set_title(f"Z Translation Error: {sched_type.upper()}", fontsize=11, fontweight='bold')
        ax_z.set_xlabel("Epoch")
        ax_z.set_ylabel("Absolute Error [m]")
        ax_z.grid(True, alpha=0.3)
        ax_z.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f"task_errors_translation.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Task-specific error plots saved to {output_dir}")


def plot_task_divergence_over_training(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Track the divergence between rotation and translation error trends.
    Hypothesis: per-head scheduling should allow tasks to adapt independently,
    reducing this divergence (more specialized learning).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted([s for s in master_df["schedule_type"].unique() if s != "none"])
    if not schedule_types:
        return

    colors = {"global": "#1f77b4", "per_head": "#ff7f0e"}

    fig, axes = plt.subplots(1, len(schedule_types), figsize=(6 * len(schedule_types), 5))
    if len(schedule_types) == 1:
        axes = [axes]

    for idx, sched_type in enumerate(schedule_types):
        ax = axes[idx]

        for scope in ["global", "per_head"]:
            subset = master_df[
                (master_df["schedule_type"] == sched_type) &
                (master_df["schedule_scope"] == scope)
            ]

            if subset.empty:
                continue

            # Group by epoch and compute mean divergence
            div_by_epoch = subset.groupby("epoch")["task_divergence"].agg(['mean', 'std']).reset_index()

            ax.plot(div_by_epoch["epoch"], div_by_epoch["mean"],
                   label=f"{scope.title()}", linewidth=2.5, marker='o',
                   markersize=4, color=colors[scope])

            if 'std' in div_by_epoch.columns and div_by_epoch['std'].notna().any():
                ax.fill_between(
                    div_by_epoch["epoch"],
                    div_by_epoch["mean"] - div_by_epoch["std"],
                    div_by_epoch["mean"] + div_by_epoch["std"],
                    alpha=0.2, color=colors[scope]
                )

        ax.set_title(f"Task Divergence: {sched_type.upper()}\n(Lower → Tasks Align)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rot/Trans Error Divergence")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f"task_divergence_over_training.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()

    # ===== PLOTLY INTERACTIVE VERSION =====
    if PLOTLY_AVAILABLE:
        fig = go.Figure()

        for sched_type in schedule_types:
            for scope in ["global", "per_head"]:
                subset = master_df[
                    (master_df["schedule_type"] == sched_type) &
                    (master_df["schedule_scope"] == scope)
                ]

                if subset.empty:
                    continue

                div_by_epoch = subset.groupby("epoch")["task_divergence"].agg(['mean', 'std']).reset_index().sort_values("epoch")

                # Use dash style to differentiate schedule types
                dash_style = 'solid' if sched_type == 'reduce_on_plateau' else 'dash'

                fig.add_trace(go.Scatter(
                    x=div_by_epoch["epoch"],
                    y=div_by_epoch["mean"],
                    mode='lines+markers',
                    name=f"{sched_type.upper()} ({scope.title()})",
                    line=dict(color=colors[scope], width=2.5, dash=dash_style),
                    error_y=dict(
                        type='data',
                        array=div_by_epoch['std'],
                        visible=True
                    )
                ))

        fig.update_layout(
            title="Task Divergence Over Training<br><sub>Lower → Rotation & Translation Errors Align Better</sub>",
            xaxis_title="Epoch",
            yaxis_title="Task Divergence (Normalized Error Difference)",
            template="plotly_white",
            height=600,
            width=1000,
            hovermode='x unified'
        )
        fig.update_yaxes(range=[0, 1])

        fig.write_html(str(output_dir / "task_divergence_interactive.html"), include_plotlyjs='cdn')
        print(f"✓ Interactive Plotly divergence plot saved: {output_dir / 'task_divergence_interactive.html'}")

    print(f"✓ Task divergence plots saved to {output_dir}")


def plot_learning_stability_across_seeds(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Compare variance in convergence across seeds for each schedule strategy.
    Higher variance → less stable learning.
    Hypothesis: per-head scheduling may have higher variance if head-specific rates diverge.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted([s for s in master_df["schedule_type"].unique() if s != "none"])
    if not schedule_types:
        return

    # Compute final epoch loss variance per (schedule_type, scope)
    summary_data = []
    for sched_type in schedule_types:
        for scope in ["global", "per_head"]:
            subset = master_df[
                (master_df["schedule_type"] == sched_type) &
                (master_df["schedule_scope"] == scope)
            ]

            if subset.empty or "val_loss" not in subset.columns:
                continue

            # Get final 5 epochs
            final_epochs = subset[subset["epoch"] >= subset["epoch"].max() - 4]

            if not final_epochs.empty:
                loss_std_across_seeds = final_epochs.groupby("seed")["val_loss"].mean().std()
                loss_mean_across_seeds = final_epochs.groupby("seed")["val_loss"].mean().mean()

                summary_data.append({
                    "Schedule Type": sched_type.upper(),
                    "Scope": scope.title(),
                    "Mean Final Loss": loss_mean_across_seeds,
                    "Std Across Seeds": loss_std_across_seeds,
                    "CV (Coefficient of Variation)": loss_std_across_seeds / loss_mean_across_seeds if loss_mean_across_seeds > 0 else 0
                })

    if not summary_data:
        print("Warning: No stability data to plot")
        return

    summary_df = pd.DataFrame(summary_data)

    # ===== MATPLOTLIB BAR PLOT =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Absolute Std Across Seeds
    for sched_type in schedule_types:
        data = summary_df[summary_df["Schedule Type"] == sched_type]
        x_pos = np.arange(len(data))
        ax1.bar(x_pos + (schedule_types.index(sched_type) * 0.35), data["Std Across Seeds"],
               width=0.35, label=sched_type.upper(), alpha=0.8)

    ax1.set_xlabel("Schedule Scope")
    ax1.set_ylabel("Std Dev of Final Loss (Across Seeds)")
    ax1.set_title("Learning Stability: Loss Variance Across Seeds\n(Lower → More Stable)",
                 fontsize=11, fontweight='bold')
    ax1.set_xticks(np.arange(len(["global", "per_head"])))
    ax1.set_xticklabels(["Global", "Per-Head"])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=9)

    # Plot 2: Coefficient of Variation
    for sched_type in schedule_types:
        data = summary_df[summary_df["Schedule Type"] == sched_type]
        x_pos = np.arange(len(data))
        ax2.bar(x_pos + (schedule_types.index(sched_type) * 0.35), data["CV (Coefficient of Variation)"],
               width=0.35, label=sched_type.upper(), alpha=0.8)

    ax2.set_xlabel("Schedule Scope")
    ax2.set_ylabel("Coefficient of Variation")
    ax2.set_title("Learning Stability: Normalized Variance\n(Lower → More Stable Relative to Mean)",
                 fontsize=11, fontweight='bold')
    ax2.set_xticks(np.arange(len(["global", "per_head"])))
    ax2.set_xticklabels(["Global", "Per-Head"])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f"stability_across_seeds.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Stability plots saved to {output_dir}")
    print("\n=== Stability Summary ===")
    print(summary_df.to_string(index=False))


def generate_summary_report(master_df: pd.DataFrame, output_dir: Path):
    """
    Generate a comprehensive text summary comparing LR schedule strategies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted([s for s in master_df["schedule_type"].unique() if s != "none"])

    with open(output_dir / "lr_schedule_summary.txt", "w") as f:
        f.write("=" * 90 + "\n")
        f.write("LEARNING RATE SCHEDULE COMPARISON REPORT\n")
        f.write("=" * 90 + "\n\n")

        for sched_type in schedule_types:
            for scope in ["global", "per_head"]:
                subset = master_df[
                    (master_df["schedule_type"] == sched_type) &
                    (master_df["schedule_scope"] == scope)
                ]

                if subset.empty:
                    continue

                final_epochs = subset[subset["epoch"] >= subset["epoch"].max() - 4]

                f.write(f"\n{'='*70}\n")
                f.write(f"SCHEDULE: {sched_type.upper()} | SCOPE: {scope.upper()}\n")
                f.write(f"{'='*70}\n")

                if "val_loss" in final_epochs.columns:
                    val_loss_mean = final_epochs["val_loss"].mean()
                    val_loss_std = final_epochs["val_loss"].std()
                    f.write(f"  Val Loss (Last 5 Epochs):           {val_loss_mean:.5f} ± {val_loss_std:.5f}\n")

                if "val_abs_rot_x" in final_epochs.columns:
                    rot_mean = final_epochs["val_abs_rot_x"].mean()
                    rot_std = final_epochs["val_abs_rot_x"].std()
                    f.write(f"  Val Abs Rotation X (deg):          {rot_mean:.4f} ± {rot_std:.4f}\n")

                if "val_abs_y" in final_epochs.columns:
                    y_mean = final_epochs["val_abs_y"].mean()
                    y_std = final_epochs["val_abs_y"].std()
                    f.write(f"  Val Abs Y Translation (m):         {y_mean:.5f} ± {y_std:.5f}\n")

                if "val_abs_z" in final_epochs.columns:
                    z_mean = final_epochs["val_abs_z"].mean()
                    z_std = final_epochs["val_abs_z"].std()
                    f.write(f"  Val Abs Z Translation (m):         {z_mean:.5f} ± {z_std:.5f}\n")

                if "task_divergence" in subset.columns:
                    div_mean = subset["task_divergence"].mean()
                    div_final = final_epochs["task_divergence"].mean()
                    f.write(f"  Task Divergence (Mean/Final):      {div_mean:.4f} / {div_final:.4f}\n")

                # Convergence rate (loss drop from epoch 1 to final)
                first_epoch = subset[subset["epoch"] == subset["epoch"].min()]
                if not first_epoch.empty and "val_loss" in first_epoch.columns:
                    first_loss = first_epoch["val_loss"].mean()
                    final_loss = final_epochs["val_loss"].mean()
                    improvement = ((first_loss - final_loss) / first_loss * 100) if first_loss > 0 else 0
                    f.write(f"  Convergence (Loss Improvement):    {improvement:.2f}%\n")

                f.write(f"\n  Seeds Used: {sorted(subset['seed'].unique())}\n")
                f.write(f"  Num Runs: {len(subset['run_name'].unique())}\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 90 + "\n")
        f.write("""
LR SCHEDULE TYPES:
  - "none" → No learning rate scheduling (constant LR throughout training)
  - "reduce_on_plateau" → LR reduced when val loss plateaus (adaptive to stagnation)
  - "warmup_cosine" → Cosine annealing with optional warmup (smooth LR decay)

SCHEDULE SCOPE:
  - "global" → Single LR schedule applied to all parameters equally
  - "per_head" → Independent LR schedules for base backbone vs. rotation/translation heads

  Hypothesis: per-head scheduling may help because rotation and translation have
  different learning dynamics and different gradient magnitudes.

CONVERGENCE COMPARISON:
  - Higher loss improvement % → Faster/better convergence
  - Lower final loss → Better end performance
  - Lower std → More consistent performance across seeds

TASK DIVERGENCE:
  - Measures how much rotation and translation errors differ during training
  - Lower divergence → Tasks learning in sync (good for stability)
  - Per-head scheduling may reduce divergence if heads can adapt independently

STABILITY ACROSS SEEDS:
  - Coefficient of Variation (CV) = Std / Mean
  - CV < 0.05 → Very stable (robust across seeds)
  - CV < 0.10 → Stable
  - CV > 0.15 → Unstable (high variance across seeds)
""")

    print(f"✓ Summary report saved to {output_dir / 'lr_schedule_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="LR Schedule Effect Analysis Pipeline"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to runs directory containing multiple run folders with metrics.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Output directory for plots and reports"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "svg"],
        default="png",
        help="Output format for plots"
    )

    args = parser.parse_args()

    runs_dir = Path(args.path)
    output_dir = Path(args.output_dir)

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    print(f"Loading data from {runs_dir}...")
    master_df, run_configs = load_and_prepare_data(runs_dir)

    print(f"Loaded {len(master_df)} total metric rows from {len(run_configs)} runs")
    print(f"LR schedule types found: {sorted([s for s in master_df['schedule_type'].unique() if s != 'none'])}")

    print("\nGenerating analysis plots...")

    # All plots
    plot_convergence_comparison(master_df, output_dir / "schedules", fmt=args.format)
    plot_task_specific_errors(master_df, output_dir / "task_errors", fmt=args.format)
    plot_task_divergence_over_training(master_df, output_dir / "task_divergence", fmt=args.format)
    plot_learning_stability_across_seeds(master_df, output_dir / "stability", fmt=args.format)

    # Report
    generate_summary_report(master_df, output_dir)

    print(f"\n✓ All analysis complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
