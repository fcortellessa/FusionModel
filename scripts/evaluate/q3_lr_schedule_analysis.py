#!/usr/bin/env python3
"""
Q3: Learning Rate Schedule Analysis Pipeline
Compares global vs per-head learning rate schedules
Analyzes: convergence speed, task-specific dynamics, bias learning, robustness
"""

import os
import glob
import argparse
import re
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def save_figure(fig, output_dir: Path, filename_base: str, fmt=None):
    """
    Save matplotlib figure in specified formats.

    Args:
        fig: matplotlib figure object
        output_dir: directory to save to
        filename_base: filename without extension
        fmt: format(s) to save in - string, list of strings, or None
             Options: "png", "svg", "plotly_html"
    """
    plt.tight_layout()

    # Normalize fmt to list
    if fmt is None:
        formats = ["png", "svg"]
    elif isinstance(fmt, str):
        formats = [fmt]
    elif isinstance(fmt, list):
        formats = [f for f in fmt if f in ["png", "svg"]]
    else:
        formats = ["png"]

    saved_paths = {}

    for file_format in formats:
        if file_format == "png":
            path = output_dir / f"{filename_base}.png"
            fig.savefig(str(path), dpi=150, bbox_inches='tight', format='png')
            saved_paths['png'] = path
        elif file_format == "svg":
            path = output_dir / f"{filename_base}.svg"
            fig.savefig(str(path), dpi=150, bbox_inches='tight', format='svg')
            saved_paths['svg'] = path

    return saved_paths


def save_plotly_figure(fig, output_dir: Path, filename_base: str):
    """Save Plotly figure as interactive HTML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename_base}_interactive.html"
    fig.write_html(str(path), include_plotlyjs='cdn')
    return path


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


def extract_schedule_config(run_name: str, cfg: dict) -> Tuple[str, int, bool]:
    """
    Extract: (schedule_type, seed, uncertainty_enabled)

    Returns:
        (schedule_type, seed, uncertainty_enabled)
        e.g., ("perHead_decayLROnPlateau", 43, True)
    """
    seed = cfg.get("training.seed", "?")

    # `cfg` is flattened, so nested LR-schedule config lives under
    # "training.lr_schedule.<key>" / "training.head_lr_decay.<key>" --
    # "training.lr_schedule" itself is never a literal key.
    schedule_enabled = cfg.get("training.lr_schedule.enabled", False)
    schedule_type = cfg.get("training.lr_schedule.type", None)
    per_head_enabled = bool(cfg.get("training.head_lr_decay.enabled", False))

    if schedule_enabled and schedule_type == "reduce_on_plateau":
        lr_schedule = "perHead_decayLROnPlateau" if per_head_enabled else "global_decayLROnPlateau"
    elif schedule_enabled and schedule_type == "warmup_cosine":
        lr_schedule = "perHead_cosine_lr_schedule" if per_head_enabled else "global_warmup_cosine"
    elif schedule_enabled and schedule_type is not None:
        lr_schedule = f"{'perHead' if per_head_enabled else 'global'}_{schedule_type}"
    else:
        # Fallback: infer from run_name (only useful if run folders are
        # actually named after their variant, which q3_global_vs_perHead_LR
        # run folders currently are not).
        if "perHead" in run_name or "per_head" in run_name:
            lr_schedule = "perHead_decayLROnPlateau"
        elif "global_warmup" in run_name or "warmup_cosine" in run_name:
            lr_schedule = "global_warmup_cosine"
        elif "decay" in run_name.lower():
            lr_schedule = "global_decayLROnPlateau"
        else:
            lr_schedule = "unknown"

    uncertainty_enabled = cfg.get("training.task_weighting.enabled", False)

    return (lr_schedule, seed, uncertainty_enabled)


def load_and_prepare_data(runs_dir: Path) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Load all metrics.csv files and extract schedule configuration.

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

    # Add schedule configuration
    master_df["schedule_type"] = master_df["run_name"].apply(
        lambda rn: extract_schedule_config(rn, run_configs.get(rn, {}))[0]
    )
    master_df["seed"] = master_df["run_name"].apply(
        lambda rn: extract_schedule_config(rn, run_configs.get(rn, {}))[1]
    )
    master_df["uncertainty_enabled"] = master_df["run_name"].apply(
        lambda rn: extract_schedule_config(rn, run_configs.get(rn, {}))[2]
    )

    return master_df, run_configs


def plot_final_accuracy_scatter(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Plot 1: Final accuracy by LR schedule
    Scatter: rotation (loss_q) vs translation (loss_trans), plus the same
    scatter re-run against the Y-only and Z-only translation loss
    separately, since y/z can have different units/scales/behavior and a
    combined metric can mask a per-axis effect.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted(master_df["schedule_type"].unique())
    colors = {
        'global_decayLROnPlateau': '#1f77b4',
        'perHead_decayLROnPlateau': '#ff7f0e',
        'global_warmup_cosine': '#2ca02c',
        'global_warmup_cosine+perHead_decayLROnPlateau': '#d62728',
        'unknown': '#888888'
    }

    # (translation loss column, axis label, filename suffix)
    trans_metrics = [
        ("val_loss_trans", "Translation", "trans"),
        ("val_loss_y", "Y Translation", "y"),
        ("val_loss_z", "Z Translation", "z"),
    ]

    for trans_col, trans_label, suffix in trans_metrics:
        if "val_loss_q" not in master_df.columns or trans_col not in master_df.columns:
            print(f"Warning: val_loss_q or {trans_col} not found, skipping scatter plot")
            continue

        # Get final epochs, then average per run_name so each seed contributes
        # exactly one point instead of one point per epoch (5x overplotting
        # per seed otherwise).
        final_epochs = master_df[master_df["epoch"] >= master_df.groupby("run_name")["epoch"].transform("max") - 4]
        per_run = final_epochs.groupby(["run_name", "schedule_type"]).agg(
            val_loss_q=("val_loss_q", "mean"),
            trans_loss=(trans_col, "mean"),
            seed=("seed", "first"),
            uncertainty_enabled=("uncertainty_enabled", "first"),
        ).reset_index()

        filename_suffix = "" if suffix == "trans" else f"_{suffix}"

        # ===== MATPLOTLIB SCATTER =====
        fig, ax = plt.subplots(figsize=(12, 9))

        for schedule_type in schedule_types:
            sched_df = per_run[per_run["schedule_type"] == schedule_type]

            if sched_df.empty:
                continue

            rot_vals = sched_df["val_loss_q"].values
            trans_vals = sched_df["trans_loss"].values
            seeds = sched_df["seed"].values
            uncertainty = sched_df["uncertainty_enabled"].values

            color = colors.get(schedule_type, '#000000')

            # Plot points with uncertainty info in marker style
            for i, (rot, trans, seed, unc) in enumerate(zip(rot_vals, trans_vals, seeds, uncertainty)):
                marker = 'o' if not unc else 's'  # circle = no uncertainty, square = uncertainty
                ax.scatter(trans, rot, s=150, alpha=0.7, color=color, marker=marker,
                          edgecolors='black', linewidth=0.5)

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = []
        for schedule_type in schedule_types:
            if schedule_type in colors:
                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=colors[schedule_type], markersize=10,
                                             label=schedule_type.replace('_', ' ')))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                     markersize=10, label='No Uncertainty'))
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                                     markersize=10, label='Uncertainty Enabled'))

        ax.legend(handles=legend_elements, fontsize=10, loc='best')

        ax.set_xlabel(f"Validation Loss - {trans_label} ({trans_col})", fontsize=12, fontweight='bold')
        ax.set_ylabel("Validation Loss - Rotation (val_loss_q)", fontsize=12, fontweight='bold')
        ax.set_title(f"Final Accuracy by LR Schedule\n(Circle=No Uncertainty | Square=Uncertainty Enabled)",
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        save_figure(fig, output_dir, f"final_accuracy_by_lr_schedule{filename_suffix}", fmt)
        plt.close()

        # ===== PLOTLY INTERACTIVE =====
        if PLOTLY_AVAILABLE and fmt and 'plotly_html' in fmt:
            fig = go.Figure()

            for schedule_type in schedule_types:
                sched_df = per_run[per_run["schedule_type"] == schedule_type]

                if sched_df.empty:
                    continue

                rot_vals = sched_df["val_loss_q"].values
                trans_vals = sched_df["trans_loss"].values
                seeds = sched_df["seed"].values
                uncertainty = sched_df["uncertainty_enabled"].values

                color = colors.get(schedule_type, '#000000')

                fig.add_trace(go.Scatter(
                    x=trans_vals,
                    y=rot_vals,
                    mode='markers',
                    name=schedule_type.replace('_', ' '),
                    marker=dict(size=12, opacity=0.7, color=color),
                    text=[f"Seed: {s}, Uncertainty: {u}" for s, u in zip(seeds, uncertainty)],
                    hovertemplate=f'<b>%{{fullData.name}}</b><br>{trans_label} Loss: %{{x:.4f}}<br>Rot Loss: %{{y:.4f}}<br>%{{text}}<extra></extra>'
                ))

            fig.update_layout(
                title=f"Final Accuracy by LR Schedule (Interactive) - {trans_label}",
                xaxis_title=f"Validation Loss - {trans_label}",
                yaxis_title="Validation Loss - Rotation",
                template="plotly_white",
                height=700,
                width=1000,
                hovermode='closest'
            )

            save_plotly_figure(fig, output_dir, f"final_accuracy_by_lr_schedule{filename_suffix}")

    print(f"✓ Final accuracy scatter plots (trans/y/z) saved to {output_dir}")


def plot_convergence_loss_q(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Plot 2a: Loss_Q (rotation) convergence over epochs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "val_loss_q" not in master_df.columns or "train_loss_q" not in master_df.columns:
        print("Warning: train_loss_q or val_loss_q not found, skipping loss_q plot")
        return

    schedule_types = sorted(master_df["schedule_type"].unique())
    colors = {
        'global_decayLROnPlateau': '#1f77b4',
        'perHead_decayLROnPlateau': '#ff7f0e',
        'global_warmup_cosine': '#2ca02c',
        'unknown': '#888888'
    }

    def _draw_ax(ax, col, split_label, log_scale):
        # Log-scale variant skips std shading: mean-std can dip to <=0
        # near a plateau, which is undefined on a log axis.
        for schedule_type in schedule_types:
            sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")
            by_epoch = sched_df.groupby("epoch")[col].agg(['mean', 'std']).reset_index()

            color = colors.get(schedule_type, '#000000')
            ax.plot(by_epoch["epoch"], by_epoch["mean"],
                    label=schedule_type.replace('_', ' '), linewidth=2.5, color=color)
            if not log_scale and by_epoch['std'].notna().any():
                ax.fill_between(by_epoch["epoch"],
                                 by_epoch["mean"] - by_epoch["std"],
                                 by_epoch["mean"] + by_epoch["std"],
                                 alpha=0.15, color=color)

        ax.set_title(f"{split_label}{' (log scale)' if log_scale else ''}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value" + (" [log]" if log_scale else ""))
        if log_scale:
            ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both' if log_scale else 'major')

    # ===== MATPLOTLIB =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_ax(axes[0], "train_loss_q", "Training Loss - Rotation (Loss_Q)", log_scale=False)
    _draw_ax(axes[1], "val_loss_q", "Validation Loss - Rotation (Loss_Q)", log_scale=False)
    save_figure(fig, output_dir, "convergence_loss_q_by_schedule", fmt)
    plt.close()

    # Log-y variant: early-training loss is often orders of magnitude
    # above the plateau, compressing late-epoch differences on a linear
    # axis. Log scale makes where each schedule actually converges legible.
    fig_log, axes_log = plt.subplots(1, 2, figsize=(14, 5))
    _draw_ax(axes_log[0], "train_loss_q", "Training Loss - Rotation (Loss_Q)", log_scale=True)
    _draw_ax(axes_log[1], "val_loss_q", "Validation Loss - Rotation (Loss_Q)", log_scale=True)
    save_figure(fig_log, output_dir, "convergence_loss_q_by_schedule_log", fmt)
    plt.close()

    # ===== PLOTLY INTERACTIVE =====
    if PLOTLY_AVAILABLE and fmt and 'plotly_html' in fmt:
        for log_scale in (False, True):
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Training Loss - Rotation", "Validation Loss - Rotation"),
                specs=[[{}, {}]]
            )

            for schedule_type in schedule_types:
                sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")
                by_epoch_train = sched_df.groupby("epoch")["train_loss_q"].mean().reset_index()
                by_epoch_val = sched_df.groupby("epoch")["val_loss_q"].mean().reset_index()

                color = colors.get(schedule_type, '#000000')
                label = schedule_type.replace('_', ' ')

                fig.add_trace(
                    go.Scatter(x=by_epoch_train["epoch"], y=by_epoch_train["train_loss_q"],
                              mode='lines', name=f"{label} (train)", line=dict(color=color)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=by_epoch_val["epoch"], y=by_epoch_val["val_loss_q"],
                              mode='lines', name=f"{label} (val)", line=dict(color=color, dash='dash')),
                    row=1, col=2
                )

            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Loss Value", type="log" if log_scale else "linear", row=1, col=1)
            fig.update_yaxes(title_text="Loss Value", type="log" if log_scale else "linear", row=1, col=2)
            fig.update_layout(template="plotly_white", height=500, width=1200, hovermode='x unified')

            save_plotly_figure(fig, output_dir, "convergence_loss_q_by_schedule_log" if log_scale else "convergence_loss_q_by_schedule")

    print(f"✓ Loss_Q convergence plots (linear + log-scale) saved to {output_dir}")


def plot_convergence_loss_trans(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Plot 2b: Loss_Trans (translation) convergence over epochs, plus the same
    convergence plot for the Y-only and Z-only translation loss separately,
    since y/z can have different units/scales/behavior and a combined
    metric can mask a per-axis effect.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted(master_df["schedule_type"].unique())
    colors = {
        'global_decayLROnPlateau': '#1f77b4',
        'perHead_decayLROnPlateau': '#ff7f0e',
        'global_warmup_cosine': '#2ca02c',
        'unknown': '#888888'
    }

    def _draw_ax(ax, col, split_label, log_scale):
        for schedule_type in schedule_types:
            sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")
            by_epoch = sched_df.groupby("epoch")[col].agg(['mean', 'std']).reset_index()

            color = colors.get(schedule_type, '#000000')
            ax.plot(by_epoch["epoch"], by_epoch["mean"],
                    label=schedule_type.replace('_', ' '), linewidth=2.5, color=color)
            if not log_scale and by_epoch['std'].notna().any():
                ax.fill_between(by_epoch["epoch"],
                                 by_epoch["mean"] - by_epoch["std"],
                                 by_epoch["mean"] + by_epoch["std"],
                                 alpha=0.15, color=color)

        ax.set_title(f"{split_label}{' (log scale)' if log_scale else ''}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value" + (" [log]" if log_scale else ""))
        if log_scale:
            ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both' if log_scale else 'major')

    # (train col, val col, axis label, filename suffix)
    trans_metrics = [
        ("train_loss_trans", "val_loss_trans", "Translation", "trans"),
        ("loss_y", "val_loss_y", "Y Translation", "y"),
        ("loss_z", "val_loss_z", "Z Translation", "z"),
    ]

    for train_col, val_col, trans_label, suffix in trans_metrics:
        if val_col not in master_df.columns or train_col not in master_df.columns:
            print(f"Warning: {train_col} or {val_col} not found, skipping {trans_label.lower()} loss plot")
            continue

        # ===== MATPLOTLIB =====
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        _draw_ax(axes[0], train_col, f"Training Loss - {trans_label}", log_scale=False)
        _draw_ax(axes[1], val_col, f"Validation Loss - {trans_label}", log_scale=False)
        save_figure(fig, output_dir, f"convergence_loss_{suffix}_by_schedule", fmt)
        plt.close()

        fig_log, axes_log = plt.subplots(1, 2, figsize=(14, 5))
        _draw_ax(axes_log[0], train_col, f"Training Loss - {trans_label}", log_scale=True)
        _draw_ax(axes_log[1], val_col, f"Validation Loss - {trans_label}", log_scale=True)
        save_figure(fig_log, output_dir, f"convergence_loss_{suffix}_by_schedule_log", fmt)
        plt.close()

    print(f"✓ Loss_Trans convergence plots (trans/y/z, linear + log-scale) saved to {output_dir}")


def plot_signed_errors(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Plot 3: Signed errors over epochs (rotation + translation)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted(master_df["schedule_type"].unique())
    colors = {
        'global_decayLROnPlateau': '#1f77b4',
        'perHead_decayLROnPlateau': '#ff7f0e',
        'global_warmup_cosine': '#2ca02c',
        'unknown': '#888888'
    }

    # Plot rotation signed errors
    signed_rot_cols = [c for c in master_df.columns if 'signed' in c.lower() and 'rot' in c.lower()]
    if signed_rot_cols:
        fig, ax = plt.subplots(figsize=(12, 6))

        for schedule_type in schedule_types:
            sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")
            by_epoch = sched_df.groupby("epoch")[signed_rot_cols[0]].agg(['mean', 'std']).reset_index()

            color = colors.get(schedule_type, '#000000')
            ax.plot(by_epoch["epoch"], by_epoch["mean"],
                   label=schedule_type.replace('_', ' '), linewidth=2.5, marker='o', markersize=3, color=color)
            if 'std' in by_epoch.columns and by_epoch['std'].notna().any():
                ax.fill_between(by_epoch["epoch"],
                               by_epoch["mean"] - by_epoch["std"],
                               by_epoch["mean"] + by_epoch["std"],
                               alpha=0.15, color=color)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title("Signed Rotation Error Over Training\n(Positive=Overprediction | Negative=Underprediction)",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Signed Error")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        save_figure(fig, output_dir, "signed_errors_rotation_by_schedule", fmt)
        plt.close()

    # Plot translation signed errors
    signed_trans_cols = [c for c in master_df.columns if 'signed' in c.lower() and ('_y' in c.lower() or '_z' in c.lower())]
    if signed_trans_cols:
        fig, axes = plt.subplots(1, len(signed_trans_cols), figsize=(6 * len(signed_trans_cols), 5))
        if len(signed_trans_cols) == 1:
            axes = [axes]

        for idx, col in enumerate(signed_trans_cols):
            ax = axes[idx]

            for schedule_type in schedule_types:
                sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")
                by_epoch = sched_df.groupby("epoch")[col].agg(['mean', 'std']).reset_index()

                color = colors.get(schedule_type, '#000000')
                ax.plot(by_epoch["epoch"], by_epoch["mean"],
                       label=schedule_type.replace('_', ' '), linewidth=2.5, marker='o', markersize=3, color=color)
                if 'std' in by_epoch.columns and by_epoch['std'].notna().any():
                    ax.fill_between(by_epoch["epoch"],
                                   by_epoch["mean"] - by_epoch["std"],
                                   by_epoch["mean"] + by_epoch["std"],
                                   alpha=0.15, color=color)

            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_title(f"Signed {col.replace('signed_', '').upper()} Error",
                        fontsize=11, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Signed Error")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, output_dir, "signed_errors_translation_by_schedule", fmt)
        plt.close()

    print(f"✓ Signed error plots saved to {output_dir}")


def plot_absolute_errors(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Plot 5: Absolute errors over epochs (rotation + Y + Z translation)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted(master_df["schedule_type"].unique())
    colors = {
        'global_decayLROnPlateau': '#1f77b4',
        'perHead_decayLROnPlateau': '#ff7f0e',
        'global_warmup_cosine': '#2ca02c',
        'unknown': '#888888'
    }

    # Rotation absolute errors
    if "val_abs_rot_x" in master_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Train
        ax_train = axes[0]
        for schedule_type in schedule_types:
            sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")
            by_epoch = sched_df.groupby("epoch")["train_abs_rot_x"].agg(['mean', 'std']).reset_index()

            color = colors.get(schedule_type, '#000000')
            ax_train.plot(by_epoch["epoch"], by_epoch["mean"],
                         label=schedule_type.replace('_', ' '), linewidth=2.5, color=color)
            if 'std' in by_epoch.columns and by_epoch['std'].notna().any():
                ax_train.fill_between(by_epoch["epoch"],
                                     by_epoch["mean"] - by_epoch["std"],
                                     by_epoch["mean"] + by_epoch["std"],
                                     alpha=0.15, color=color)

        ax_train.set_title("Training Absolute Rotation Error", fontsize=11, fontweight='bold')
        ax_train.set_xlabel("Epoch")
        ax_train.set_ylabel("Error [deg]")
        ax_train.legend(fontsize=9)
        ax_train.grid(True, alpha=0.3)

        # Val
        ax_val = axes[1]
        for schedule_type in schedule_types:
            sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")
            by_epoch = sched_df.groupby("epoch")["val_abs_rot_x"].agg(['mean', 'std']).reset_index()

            color = colors.get(schedule_type, '#000000')
            ax_val.plot(by_epoch["epoch"], by_epoch["mean"],
                       label=schedule_type.replace('_', ' '), linewidth=2.5, color=color)
            if 'std' in by_epoch.columns and by_epoch['std'].notna().any():
                ax_val.fill_between(by_epoch["epoch"],
                                   by_epoch["mean"] - by_epoch["std"],
                                   by_epoch["mean"] + by_epoch["std"],
                                   alpha=0.15, color=color)

        ax_val.set_title("Validation Absolute Rotation Error", fontsize=11, fontweight='bold')
        ax_val.set_xlabel("Epoch")
        ax_val.set_ylabel("Error [deg]")
        ax_val.legend(fontsize=9)
        ax_val.grid(True, alpha=0.3)

        save_figure(fig, output_dir, "absolute_errors_rotation_by_schedule", fmt)
        plt.close()

    # Translation absolute errors
    if "val_abs_y" in master_df.columns and "val_abs_z" in master_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for idx, (col_prefix, row_idx) in enumerate([("train_abs_y", 0), ("train_abs_z", 1), ("val_abs_y", 0), ("val_abs_z", 1)]):
            if col_prefix.startswith("train"):
                ax = axes[row_idx, 0]
                is_train = True
            else:
                ax = axes[row_idx, 1]
                is_train = False

            for schedule_type in schedule_types:
                sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")
                by_epoch = sched_df.groupby("epoch")[col_prefix].agg(['mean', 'std']).reset_index()

                color = colors.get(schedule_type, '#000000')
                ax.plot(by_epoch["epoch"], by_epoch["mean"],
                       label=schedule_type.replace('_', ' '), linewidth=2.5, color=color)
                if 'std' in by_epoch.columns and by_epoch['std'].notna().any():
                    ax.fill_between(by_epoch["epoch"],
                                   by_epoch["mean"] - by_epoch["std"],
                                   by_epoch["mean"] + by_epoch["std"],
                                   alpha=0.15, color=color)

            axis_name = col_prefix.split('_')[-1].upper()
            split_name = "Train" if is_train else "Val"
            ax.set_title(f"{split_name} Absolute {axis_name} Translation Error", fontsize=11, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Error [m]")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, output_dir, "absolute_errors_translation_by_schedule", fmt)
        plt.close()

    print(f"✓ Absolute error plots saved to {output_dir}")


def plot_task_decoupling(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Plot 6: Task decoupling - loss ratio dynamics over training, plus the
    same ratio plot against the Y-only and Z-only translation loss
    separately, since y/z can have different units/scales/behavior and a
    combined metric can mask a per-axis effect.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted(master_df["schedule_type"].unique())
    colors = {
        'global_decayLROnPlateau': '#1f77b4',
        'perHead_decayLROnPlateau': '#ff7f0e',
        'global_warmup_cosine': '#2ca02c',
        'unknown': '#888888'
    }

    # (translation loss column, axis label, filename suffix)
    trans_metrics = [
        ("val_loss_trans", "Loss_Trans", "trans"),
        ("val_loss_y", "Loss_Y", "y"),
        ("val_loss_z", "Loss_Z", "z"),
    ]

    for trans_col, trans_label, suffix in trans_metrics:
        if "val_loss_q" not in master_df.columns or trans_col not in master_df.columns:
            print(f"Warning: Cannot compute loss ratio without loss_q and {trans_col}")
            continue

        filename_suffix = "" if suffix == "trans" else f"_{suffix}"

        fig, ax = plt.subplots(figsize=(12, 6))

        for schedule_type in schedule_types:
            sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")

            # Compute loss ratio
            sched_df = sched_df.copy()
            sched_df['loss_ratio'] = sched_df['val_loss_q'] / (sched_df[trans_col] + 1e-6)

            by_epoch = sched_df.groupby("epoch")['loss_ratio'].agg(['mean', 'std']).reset_index()

            color = colors.get(schedule_type, '#000000')
            ax.plot(by_epoch["epoch"], by_epoch["mean"],
                   label=schedule_type.replace('_', ' '), linewidth=2.5, marker='o', markersize=3, color=color)
            if 'std' in by_epoch.columns and by_epoch['std'].notna().any():
                ax.fill_between(by_epoch["epoch"],
                               by_epoch["mean"] - by_epoch["std"],
                               by_epoch["mean"] + by_epoch["std"],
                               alpha=0.15, color=color)

        ax.set_title(f"Task Decoupling: Loss_Q / {trans_label} Ratio Over Training\n(Global schedule forces ratio constant; Per-head allows variation)",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Loss Ratio (Loss_Q / {trans_label})")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        save_figure(fig, output_dir, f"task_decoupling_analysis{filename_suffix}", fmt)
        plt.close()

    print(f"✓ Task decoupling plots (trans/y/z) saved to {output_dir}")


def plot_learning_rate_schedules(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Plot 4: Learning rate schedules over training epochs
    Shows how LR evolves for different schedules
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted(master_df["schedule_type"].unique())

    # Check if we have LR data logged
    lr_cols = [c for c in master_df.columns if 'learning_rate' in c.lower() or 'lr' in c.lower()]
    if not lr_cols:
        print("Warning: No learning rate columns found in metrics. Skipping LR schedule plot.")
        print("         To visualize LR schedules, add LR logging to train.py:")
        print("         - log_dict['learning_rate'] = optimizer.param_groups[0]['lr']")
        print("         - Or log per-group: log_dict['lr_backbone'], log_dict['lr_head_rot'], etc.")
        return

    colors = {
        'global_decayLROnPlateau': '#1f77b4',
        'perHead_decayLROnPlateau': '#ff7f0e',
        'global_warmup_cosine': '#2ca02c',
        'global_warmup_cosine+perHead_decayLROnPlateau': '#d62728',
        'unknown': '#888888'
    }

    # ===== MATPLOTLIB =====
    fig, ax = plt.subplots(figsize=(14, 7))

    for schedule_type in schedule_types:
        sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")

        # Use first available LR column
        lr_col = lr_cols[0]
        by_epoch = sched_df.groupby("epoch")[lr_col].agg(['mean', 'std']).reset_index()

        color = colors.get(schedule_type, '#000000')
        ax.plot(by_epoch["epoch"], by_epoch["mean"],
               label=schedule_type.replace('_', ' '), linewidth=2.5, marker='o', markersize=4, color=color)

        if 'std' in by_epoch.columns and by_epoch['std'].notna().any():
            ax.fill_between(by_epoch["epoch"],
                           by_epoch["mean"] - by_epoch["std"],
                           by_epoch["mean"] + by_epoch["std"],
                           alpha=0.15, color=color)

    ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
    ax.set_ylabel("Learning Rate (log scale)", fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_title(f"Learning Rate Schedules Over Training\n(Using: {lr_col})",
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, which='both')

    save_figure(fig, output_dir, "learning_rate_schedules", fmt)
    plt.close()

    # ===== PLOTLY INTERACTIVE =====
    if PLOTLY_AVAILABLE and fmt and 'plotly_html' in fmt:
        fig = go.Figure()

        for schedule_type in schedule_types:
            sched_df = master_df[master_df["schedule_type"] == schedule_type].sort_values("epoch")
            by_epoch = sched_df.groupby("epoch")[lr_col].mean().reset_index()

            color = colors.get(schedule_type, '#000000')

            fig.add_trace(go.Scatter(
                x=by_epoch["epoch"],
                y=by_epoch[lr_col],
                mode='lines+markers',
                name=schedule_type.replace('_', ' '),
                line=dict(color=color, width=2.5),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title=f"Learning Rate Schedules Over Training (Interactive)",
            xaxis_title="Epoch",
            yaxis_title="Learning Rate (log scale)",
            yaxis_type="log",
            template="plotly_white",
            height=600,
            width=1200,
            hovermode='x unified'
        )

        save_plotly_figure(fig, output_dir, "learning_rate_schedules")

    print(f"✓ Learning rate schedule plot saved to {output_dir}")


def plot_best_epoch_tracking(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Plot 7: Best epoch tracking by schedule
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "val_loss" not in master_df.columns:
        print("Warning: val_loss not found, skipping best epoch plot")
        return

    # Find best epoch per run
    best_epochs = []
    for run_name in master_df["run_name"].unique():
        run_df = master_df[master_df["run_name"] == run_name]
        best_epoch = run_df.loc[run_df["val_loss"].idxmin(), "epoch"]
        schedule_type = run_df.iloc[0]["schedule_type"]
        best_epochs.append({'schedule_type': schedule_type, 'best_epoch': best_epoch})

    best_df = pd.DataFrame(best_epochs)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    schedule_types = sorted(best_df["schedule_type"].unique())
    means = [best_df[best_df["schedule_type"] == st]["best_epoch"].mean() for st in schedule_types]
    stds = [best_df[best_df["schedule_type"] == st]["best_epoch"].std() for st in schedule_types]

    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax.bar(range(len(schedule_types)), means, yerr=stds, capsize=5,
           color=colors_list[:len(schedule_types)], alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_xticks(range(len(schedule_types)))
    ax.set_xticklabels([s.replace('_', '\n') for s in schedule_types], fontsize=10)
    ax.set_ylabel("Best Epoch", fontsize=12, fontweight='bold')
    ax.set_title("Best Epoch (Where val_loss is Minimum) by LR Schedule\n(Lower = Faster Convergence)",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    save_figure(fig, output_dir, "best_epoch_tracking", fmt)
    plt.close()

    print(f"✓ Best epoch plot saved to {output_dir}")


def generate_summary_report(master_df: pd.DataFrame, output_dir: Path):
    """Generate text summary report comparing schedules."""
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_types = sorted(master_df["schedule_type"].unique())

    with open(output_dir / "lr_schedule_summary.txt", "w") as f:
        f.write("=" * 90 + "\n")
        f.write("LEARNING RATE SCHEDULE COMPARISON REPORT\n")
        f.write("=" * 90 + "\n\n")

        for schedule_type in schedule_types:
            sched_df = master_df[master_df["schedule_type"] == schedule_type]
            final_epochs = sched_df[sched_df["epoch"] >= sched_df.groupby("run_name")["epoch"].transform("max") - 4]

            f.write(f"\n{'='*80}\n")
            f.write(f"SCHEDULE: {schedule_type.upper()}\n")
            f.write(f"{'='*80}\n")

            if "val_loss" in final_epochs.columns:
                val_loss_mean = final_epochs["val_loss"].mean()
                val_loss_std = final_epochs["val_loss"].std()
                f.write(f"  Combined Val Loss (Last 5 Epochs):  {val_loss_mean:.5f} ± {val_loss_std:.5f}\n")

            if "val_loss_q" in final_epochs.columns:
                rot_loss_mean = final_epochs["val_loss_q"].mean()
                rot_loss_std = final_epochs["val_loss_q"].std()
                f.write(f"  Val Loss - Rotation (loss_q):        {rot_loss_mean:.5f} ± {rot_loss_std:.5f}\n")

            if "val_loss_trans" in final_epochs.columns:
                trans_loss_mean = final_epochs["val_loss_trans"].mean()
                trans_loss_std = final_epochs["val_loss_trans"].std()
                f.write(f"  Val Loss - Translation (loss_trans): {trans_loss_mean:.5f} ± {trans_loss_std:.5f}\n")

            if "val_abs_rot_x" in final_epochs.columns:
                rot_err_mean = final_epochs["val_abs_rot_x"].mean()
                rot_err_std = final_epochs["val_abs_rot_x"].std()
                f.write(f"  Val Abs Rotation Error (deg):       {rot_err_mean:.4f} ± {rot_err_std:.4f}\n")

            if "val_abs_y" in final_epochs.columns and "val_abs_z" in final_epochs.columns:
                y_err_mean = final_epochs["val_abs_y"].mean()
                z_err_mean = final_epochs["val_abs_z"].mean()
                f.write(f"  Val Abs Y Translation (m):          {y_err_mean:.5f}\n")
                f.write(f"  Val Abs Z Translation (m):          {z_err_mean:.5f}\n")

            f.write(f"\n  Seeds Used: {sorted(sched_df['seed'].unique())}\n")
            f.write(f"  Runs: {sched_df['run_name'].nunique()}\n")

            # Best epoch
            if "val_loss" in sched_df.columns:
                best_epochs = []
                for run_name in sched_df["run_name"].unique():
                    run_df = sched_df[sched_df["run_name"] == run_name]
                    best_epoch = run_df.loc[run_df["val_loss"].idxmin(), "epoch"]
                    best_epochs.append(best_epoch)
                f.write(f"  Best Epoch (mean ± std):            {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 90 + "\n")
        f.write("""
GLOBAL vs PER-HEAD LEARNING RATE SCHEDULES:

Global Schedule (Single LR for all parameters):
  - Simpler, fewer hyperparameters
  - Forces rotation and translation to learn at same pace
  - May under-optimize one task if they converge at different rates
  - Baseline approach

Per-Head Schedule (Task-specific LR for rotation/translation heads):
  - More complex, additional hyperparameters
  - Allows each task to find optimal learning pace independently
  - May achieve better balance between rotation and translation
  - Research hypothesis: per-head better captures task-specific dynamics

KEY METRICS TO COMPARE:
1. Final accuracy (loss_q + loss_trans) → Does per-head win on both?
2. Convergence speed (best epoch) → Does per-head converge faster?
3. Task balance (loss_q vs loss_trans) → Does per-head balance tasks better?
4. Robustness (error bars) → Is per-head stable across seeds?
5. Signed errors → Does per-head reduce systematic biases?

EXPECTED RESULTS:
✓ Per-head better: Clear wins on rotation or translation; better overall balance
⚠️ Trade-off: Per-head better on one task, worse on another
❌ Similar performance: LR schedule not the bottleneck; focus on architecture/weighting
""")

    print(f"✓ Summary report saved to {output_dir / 'lr_schedule_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Q3: Learning Rate Schedule Analysis (Global vs Per-Head)"
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
        default="analysis_q3_lr_schedule",
        help="Output directory for plots and reports"
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs='+',
        choices=["png", "svg", "plotly_html"],
        default=["png"],
        help="Output format(s) for plots (can specify multiple: --format png svg plotly_html)"
    )
    parser.add_argument(
        "--plots",
        type=str,
        nargs='+',
        choices=["scatter", "loss_q", "loss_trans", "signed_errors", "absolute_errors",
                "lr_schedules", "task_decoupling", "best_epoch"],
        default=["scatter", "loss_q", "loss_trans", "signed_errors", "absolute_errors",
                "lr_schedules", "task_decoupling", "best_epoch"],
        help="Which plots to generate"
    )

    args = parser.parse_args()

    runs_dir = Path(args.path)
    output_dir = Path(args.output_dir)

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    print(f"Loading data from {runs_dir}...")
    master_df, run_configs = load_and_prepare_data(runs_dir)

    print(f"Loaded {len(master_df)} total metric rows from {len(run_configs)} runs")
    print(f"LR schedules found: {sorted(master_df['schedule_type'].unique())}")

    print("\nGenerating analysis plots...")

    # Generate requested plots
    if "scatter" in args.plots:
        plot_final_accuracy_scatter(master_df, output_dir / "performance", fmt=args.format)
    if "loss_q" in args.plots:
        plot_convergence_loss_q(master_df, output_dir / "convergence", fmt=args.format)
    if "loss_trans" in args.plots:
        plot_convergence_loss_trans(master_df, output_dir / "convergence", fmt=args.format)
    if "lr_schedules" in args.plots:
        plot_learning_rate_schedules(master_df, output_dir / "convergence", fmt=args.format)
    if "signed_errors" in args.plots:
        plot_signed_errors(master_df, output_dir / "errors", fmt=args.format)
    if "absolute_errors" in args.plots:
        plot_absolute_errors(master_df, output_dir / "errors", fmt=args.format)
    if "task_decoupling" in args.plots:
        plot_task_decoupling(master_df, output_dir / "dynamics", fmt=args.format)
    if "best_epoch" in args.plots:
        plot_best_epoch_tracking(master_df, output_dir / "convergence", fmt=args.format)

    # Summary report
    generate_summary_report(master_df, output_dir)

    print(f"\n✓ Analysis complete. Output saved to {output_dir}")
    print("\nNext steps:")
    print("  1. Check scatter plot: Do per-head schedules dominate?")
    print("  2. Check convergence plots: Do tasks learn at different rates?")
    print("  3. Check signed errors: Does per-head reduce bias?")
    print("  4. Read summary report: Quantitative comparison")


if __name__ == "__main__":
    main()
