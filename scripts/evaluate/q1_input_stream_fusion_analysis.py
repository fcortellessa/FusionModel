#!/usr/bin/env python3
"""
Input Stream Fusion Analysis Pipeline
Compares fusion architectures: concatenation | gating | gated_concatenation
Analyzes: compression vs raw data preservation trade-offs
Visualizes: task performance, convergence rates, error distributions
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


def save_figure(fig, output_dir: Path, filename_base: str, fmt: str = "png"):
    """
    Save matplotlib figure in both PNG and SVG formats for publication.

    Args:
        fig: matplotlib figure object
        output_dir: directory to save to
        filename_base: filename without extension (e.g., 'rotation_vs_translation_scatter')
        fmt: primary format (png or svg)
    """
    plt.tight_layout()

    # Always save both PNG and SVG for maximum compatibility
    png_path = output_dir / f"{filename_base}.png"
    svg_path = output_dir / f"{filename_base}.svg"

    fig.savefig(str(png_path), dpi=150, bbox_inches='tight', format='png')
    fig.savefig(str(svg_path), dpi=150, bbox_inches='tight', format='svg')

    return png_path, svg_path


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


def extract_fusion_type(run_name: str, cfg: dict) -> Tuple[str, int]:
    """
    Extract fusion type and seed from config.

    Returns:
        (fusion_type, seed)
        e.g., ("gating", 40)
    """
    fusion_type = cfg.get("model.fusion", "unknown")
    seed = cfg.get("training.seed", "?")
    return (fusion_type, seed)


def load_and_prepare_data(runs_dir: Path) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[str, pd.DataFrame]]:
    """
    Load all metrics.csv and test results files.

    Returns:
        (master_df, run_configs, test_results)
    """
    metrics_files = glob.glob(os.path.join(str(runs_dir), "*/metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.csv files found in {runs_dir}")

    all_dfs = []
    run_configs = {}
    test_results = {}

    for path in metrics_files:
        run_name = os.path.basename(os.path.dirname(path))
        config_path = Path(os.path.dirname(path)) / "config.yml"
        run_dir = Path(os.path.dirname(path))

        cfg = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg = flatten_dict(yaml.safe_load(f) or {})
            except Exception as e:
                print(f"Warning: Failed to load config for {run_name}: {e}")

        run_configs[run_name] = cfg

        # Load metrics.csv
        try:
            df = pd.read_csv(path)
            if not df.empty:
                df["run_name"] = run_name
                all_dfs.append(df)
        except Exception as e:
            print(f"Error reading metrics for {run_name}: {e}")

        # Load test results if available
        test_files = glob.glob(os.path.join(str(run_dir), "results/results_*.csv")) + \
                     glob.glob(os.path.join(str(run_dir), "results_*.csv"))
        if test_files:
            try:
                latest_test_file = max(test_files, key=os.path.getctime)
                test_df = pd.read_csv(latest_test_file)
                if not test_df.empty:
                    test_df["run_name"] = run_name
                    test_results[run_name] = test_df
            except Exception as e:
                print(f"Warning: Failed to load test results for {run_name}: {e}")

    if not all_dfs:
        raise ValueError("No valid metrics files loaded")

    master_df = pd.concat(all_dfs, ignore_index=True)

    # Add fusion type classification
    master_df["fusion_type"] = master_df["run_name"].apply(
        lambda rn: extract_fusion_type(rn, run_configs.get(rn, {}))[0]
    )
    master_df["seed"] = master_df["run_name"].apply(
        lambda rn: extract_fusion_type(rn, run_configs.get(rn, {}))[1]
    )

    return master_df, run_configs, test_results


def plot_rotation_vs_translation_scatter(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Scatter plot: val_abs_rot_x vs val_abs_trans by fusion type.
    Shows if compression (gating) hurts one task more than the other.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())
    colors = {'concat': '#1f77b4', 'gating': '#ff7f0e', 'gated_concat': '#2ca02c',
              'gated_concatenation': '#2ca02c', 'simple_gated': '#d62728', 'ARGate_Fusion': '#9467bd'}

    # ===== MATPLOTLIB SCATTER PLOT =====
    fig, ax = plt.subplots(figsize=(10, 8))

    for fusion_type in fusion_types:
        fuse_df = master_df[master_df["fusion_type"] == fusion_type]

        # Take last 5 epochs for final performance estimate, then average
        # per run_name so each seed contributes exactly one point instead
        # of one point per epoch (5x overplotting per seed otherwise).
        final_epochs = fuse_df[fuse_df["epoch"] >= fuse_df.groupby("run_name")["epoch"].transform("max") - 4]
        per_run = final_epochs.groupby("run_name").agg(
            val_abs_rot_x=("val_abs_rot_x", "mean"),
            val_abs_trans=("val_abs_trans", "mean"),
        ) if "val_abs_rot_x" in final_epochs.columns and "val_abs_trans" in final_epochs.columns else None

        if per_run is not None:
            rot_vals = per_run["val_abs_rot_x"].values
            trans_vals = per_run["val_abs_trans"].values

            color = colors.get(fusion_type, '#000000')
            ax.scatter(trans_vals, rot_vals, label=fusion_type.upper(), s=100, alpha=0.6, color=color)

    ax.set_xlabel("Validation Translation Error [m]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Validation Rotation Error X [deg]", fontsize=12, fontweight='bold')
    ax.set_title("Rotation vs Translation Task Performance by Fusion Architecture\n(Final 5 Epochs Average)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, "rotation_vs_translation_scatter", fmt)
    plt.close()

    # ===== PLOTLY INTERACTIVE SCATTER =====
    if PLOTLY_AVAILABLE:
        fig = go.Figure()

        for fusion_type in fusion_types:
            fuse_df = master_df[master_df["fusion_type"] == fusion_type]
            final_epochs = fuse_df[fuse_df["epoch"] >= fuse_df.groupby("run_name")["epoch"].transform("max") - 4]
            per_run = final_epochs.groupby("run_name").agg(
                val_abs_rot_x=("val_abs_rot_x", "mean"),
                val_abs_trans=("val_abs_trans", "mean"),
                seed=("seed", "first"),
            ) if "val_abs_rot_x" in final_epochs.columns and "val_abs_trans" in final_epochs.columns else None

            if per_run is not None:
                rot_vals = per_run["val_abs_rot_x"].values
                trans_vals = per_run["val_abs_trans"].values
                seeds = per_run["seed"].values

                color = colors.get(fusion_type, '#000000')
                fig.add_trace(go.Scatter(
                    x=trans_vals,
                    y=rot_vals,
                    mode='markers',
                    name=fusion_type.upper(),
                    marker=dict(size=10, opacity=0.6, color=color),
                    text=[f"Seed: {s}" for s in seeds],
                    hovertemplate='<b>%{fullData.name}</b><br>Trans: %{x:.5f} m<br>Rot: %{y:.4f} deg<br>%{text}<extra></extra>'
                ))

        fig.update_layout(
            title="Rotation vs Translation Task Performance by Fusion Architecture<br><sub>Final 5 Epochs Average</sub>",
            xaxis_title="Validation Translation Error [m]",
            yaxis_title="Validation Rotation Error X [deg]",
            template="plotly_white",
            height=700,
            width=1000,
            hovermode='closest'
        )

        fig.write_html(str(output_dir / "rotation_vs_translation_scatter_interactive.html"), include_plotlyjs='cdn')
        print(f"✓ Interactive Plotly scatter plot saved: {output_dir / 'rotation_vs_translation_scatter_interactive.html'}")

    print(f"✓ Rotation vs Translation scatter plots saved to {output_dir}")


def plot_rotation_vs_translation_scatter_comparison(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Rotation vs translation scatter, plus the same scatter re-run against the
    Y-only and Z-only translation error separately (in addition to the
    combined val_abs_trans = sqrt(y^2+z^2)-style metric), since y/z can have
    different units/scales/behavior and a combined metric can mask a
    per-axis effect.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())
    colors = {'concat': '#1f77b4', 'gating': '#ff7f0e', 'gated_concat': '#2ca02c',
              'gated_concatenation': '#2ca02c', 'simple_gated': '#d62728', 'ARGate_Fusion': '#9467bd'}
    markers = {'concat': 'o', 'gating': 's', 'gated_concat': '^',
               'gated_concatenation': '^', 'simple_gated': 'D', 'ARGate_Fusion': 'P'}

    # (translation column, axis label, filename suffix)
    trans_metrics = [
        ("val_abs_trans", "Translation", "trans"),
        ("val_abs_y", "Y Translation", "y"),
        ("val_abs_z", "Z Translation", "z"),
    ]

    for trans_col, trans_label, suffix in trans_metrics:
        # Take final epochs for evaluation, then average per run_name so each
        # seed contributes exactly one point instead of one point per epoch
        # (5x overplotting per seed otherwise).
        per_fusion_points = {}
        for fusion_type in fusion_types:
            fusion_df = master_df[master_df["fusion_type"] == fusion_type]
            final_epochs = fusion_df[fusion_df["epoch"] >= fusion_df.groupby("run_name")["epoch"].transform("max") - 4]
            if "val_abs_rot_x" not in final_epochs.columns or trans_col not in final_epochs.columns:
                continue
            per_run = final_epochs.groupby("run_name").agg(
                val_abs_rot_x=("val_abs_rot_x", "mean"),
                trans_val=(trans_col, "mean"),
                seed=("seed", "first"),
            )
            if not per_run.empty:
                per_fusion_points[fusion_type] = per_run

        if not per_fusion_points:
            print(f"Warning: no data for {trans_col}, skipping task performance scatter")
            continue

        filename_suffix = "" if suffix == "trans" else f"_{suffix}"

        # ===== Plot 1: all fusion methods on one scatter =====
        fig, ax = plt.subplots(figsize=(10, 8))
        for fusion_type, per_run in per_fusion_points.items():
            color = colors.get(fusion_type, '#000000')
            marker = markers.get(fusion_type, 'o')
            ax.scatter(per_run["trans_val"], per_run["val_abs_rot_x"], label=fusion_type.upper(),
                       s=100, alpha=0.7, color=color, marker=marker)

        ax.set_xlabel(f"Validation {trans_label} Error [m]", fontsize=12, fontweight='bold')
        ax.set_ylabel("Validation Rotation Error X [deg]", fontsize=12, fontweight='bold')
        ax.set_title(f"Rotation vs {trans_label} Task Performance\n(Final 5 Epochs Average, All Fusion Methods)",
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, output_dir, f"task_performance_all_fusion_methods{filename_suffix}", fmt)
        plt.close()

        # ===== Plot 2: same data, with seed number labeled next to each point =====
        fig, ax = plt.subplots(figsize=(10, 8))
        for fusion_type, per_run in per_fusion_points.items():
            color = colors.get(fusion_type, '#000000')
            marker = markers.get(fusion_type, 'o')
            ax.scatter(per_run["trans_val"], per_run["val_abs_rot_x"], label=fusion_type.upper(),
                       s=100, alpha=0.7, color=color, marker=marker)
            for trans, rot, seed in zip(per_run["trans_val"], per_run["val_abs_rot_x"], per_run["seed"]):
                ax.annotate(str(seed), (trans, rot), textcoords="offset points", xytext=(6, 4), fontsize=8)

        ax.set_xlabel(f"Validation {trans_label} Error [m]", fontsize=12, fontweight='bold')
        ax.set_ylabel("Validation Rotation Error X [deg]", fontsize=12, fontweight='bold')
        ax.set_title(f"Rotation vs {trans_label} Task Performance\n(Final 5 Epochs Average, Seed Labeled)",
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, output_dir, f"task_performance_all_fusion_methods_seeds{filename_suffix}", fmt)
        plt.close()

    print(f"✓ Task performance scatter plots (trans/y/z) saved to {output_dir}")

def plot_convergence_comparison(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Line plots comparing convergence rates: high-dim (concat/gated_concat) vs low-dim (gating).
    Shows train_* and val_* over epochs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # loss_q / loss_y+loss_z are the raw, fixed-scale per-task losses computed
    # by rotation_handler.compute_loss() *before* any uncertainty/gradnorm/dwa
    # task weighting is applied to form train_loss/val_loss. Plotting these
    # instead of train_loss/val_loss avoids conflating architecture behavior
    # with the learned uncertainty weights inflating/deflating the composite
    # loss (see training.task_weighting / training.uncertainty_weighting).
    if "train_loss_trans" not in master_df.columns:
        master_df = master_df.copy()
        master_df["train_loss_trans"] = master_df["loss_y"] + master_df["loss_z"]
        master_df["val_loss_trans"] = master_df["val_loss_y"] + master_df["val_loss_z"]

    fusion_types = sorted(master_df["fusion_type"].unique())
    metrics = [
        ("loss_q", "val_loss_q", "Rotation Loss (fixed-scale, unweighted)", "Loss Value"),
        ("train_loss_trans", "val_loss_trans", "Translation Loss (unweighted)", "Loss Value"),
        ("loss_y", "val_loss_y", "Translation Y Loss (unweighted)", "Loss Value"),
        ("loss_z", "val_loss_z", "Translation Z Loss (unweighted)", "Loss Value"),
        ("train_abs_rot_x", "val_abs_rot_x", "Rotation Error", "Error [deg]"),
        ("train_abs_trans", "val_abs_trans", "Translation Error", "Error [m]"),
        ("train_abs_y", "val_abs_y", "Translation Y Error", "Error [m]"),
        ("train_abs_z", "val_abs_z", "Translation Z Error", "Error [m]"),
    ]

    colors = {'concat': '#1f77b4', 'gating': '#ff7f0e', 'gated_concat': '#2ca02c',
              'gated_concatenation': '#2ca02c', 'simple_gated': '#d62728', 'ARGate_Fusion': '#9467bd'}

    def _draw_convergence_ax(ax, col, split_label, log_scale):
        # On the log-scale variant, skip the std shading: mean-std can dip
        # to <=0 for near-zero error metrics, which is undefined on a log
        # axis and would silently clip/warn. Mean lines alone are what
        # actually shows where each curve flattens out.
        for fusion_type in fusion_types:
            fuse_df = master_df[master_df["fusion_type"] == fusion_type].sort_values("epoch")
            by_epoch = fuse_df.groupby("epoch")[col].agg(['mean', 'std']).reset_index()

            color = colors.get(fusion_type, '#000000')
            ax.plot(by_epoch["epoch"], by_epoch["mean"], label=fusion_type.upper(),
                    linewidth=2, color=color)
            if not log_scale and by_epoch['std'].notna().any():
                ax.fill_between(by_epoch["epoch"],
                                 by_epoch["mean"] - by_epoch["std"],
                                 by_epoch["mean"] + by_epoch["std"],
                                 alpha=0.15, color=color)

        ax.set_title(f"{split_label}{' (log scale)' if log_scale else ''}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel + (" [log]" if log_scale else ""))
        if log_scale:
            ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both' if log_scale else 'major')

    # ===== MATPLOTLIB PLOTS =====
    for train_col, val_col, title, ylabel in metrics:
        if train_col not in master_df.columns or val_col not in master_df.columns:
            print(f"Warning: Columns {train_col}/{val_col} not found, skipping")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        _draw_convergence_ax(axes[0], train_col, f"Training {title}", log_scale=False)
        _draw_convergence_ax(axes[1], val_col, f"Validation {title}", log_scale=False)
        save_figure(fig, output_dir, f"convergence_{train_col.replace('train_', '')}", fmt)
        plt.close()

        # Log-y variant: early-training loss/error values are often
        # orders of magnitude above the plateau, which compresses the
        # late-epoch differences on a linear axis. Log scale makes the
        # actual convergence point (where curves flatten) legible.
        fig_log, axes_log = plt.subplots(1, 2, figsize=(14, 5))
        _draw_convergence_ax(axes_log[0], train_col, f"Training {title}", log_scale=True)
        _draw_convergence_ax(axes_log[1], val_col, f"Validation {title}", log_scale=True)
        save_figure(fig_log, output_dir, f"convergence_{train_col.replace('train_', '')}_log", fmt)
        plt.close()

    print(f"✓ Convergence comparison plots (linear + log-scale) saved to {output_dir}")


def plot_signed_errors_over_epochs(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Line plots of signed errors (not absolute) over training.
    Shows bias direction and magnitude changes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Look for signed error columns
    signed_cols = [c for c in master_df.columns if 'signed' in c.lower()]

    if not signed_cols:
        print("Warning: No signed error columns found in metrics (check train.py logging)")
        return

    fusion_types = sorted(master_df["fusion_type"].unique())
    colors = {'concat': '#1f77b4', 'gating': '#ff7f0e', 'gated_concat': '#2ca02c',
              'gated_concatenation': '#2ca02c', 'simple_gated': '#d62728', 'ARGate_Fusion': '#9467bd'}

    for signed_col in signed_cols:
        fig, ax = plt.subplots(figsize=(12, 6))

        for fusion_type in fusion_types:
            fuse_df = master_df[master_df["fusion_type"] == fusion_type].sort_values("epoch")
            by_epoch = fuse_df.groupby("epoch")[signed_col].agg(['mean', 'std']).reset_index()

            color = colors.get(fusion_type, '#000000')
            ax.plot(by_epoch["epoch"], by_epoch["mean"], label=fusion_type.upper(),
                   linewidth=2, marker='o', markersize=3, color=color)

            if 'std' in by_epoch.columns and by_epoch['std'].notna().any():
                ax.fill_between(by_epoch["epoch"],
                               by_epoch["mean"] - by_epoch["std"],
                               by_epoch["mean"] + by_epoch["std"],
                               alpha=0.15, color=color)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f"Signed Error Over Training: {signed_col}\n(Positive → Overprediction | Negative → Underprediction)",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Signed Error")
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        save_figure(fig, output_dir, f"signed_errors_{signed_col}", fmt)
        plt.close()

    print(f"✓ Signed error plots saved to {output_dir}")


def plot_signed_test_error_distribution(test_results: Dict[str, pd.DataFrame], output_dir: Path, fmt: str = "png"):
    """
    Violin plots of signed test errors by fusion type.
    Determines if errors follow uniform distribution or have directional bias.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_results:
        print("Warning: No test results found (run test.py to generate test results)")
        return

    # Collect signed error columns
    signed_error_cols = {}
    for run_name, test_df in test_results.items():
        signed_cols = [c for c in test_df.columns if 'signed' in c.lower() or
                       (('error' in c.lower() or 'diff' in c.lower()) and 'abs' not in c.lower())]
        for col in signed_cols:
            if col not in signed_error_cols:
                signed_error_cols[col] = []

    if not signed_error_cols:
        print("Warning: No signed error columns found in test results")
        return

    # Extract fusion types from run names
    fusion_map = {}
    for run_name in test_results.keys():
        # Parse fusion type from run name or config
        if 'concat' in run_name.lower():
            fusion_map[run_name] = 'concat'
        elif 'gated_concat' in run_name.lower():
            fusion_map[run_name] = 'gated_concat'
        elif 'gating' in run_name.lower():
            fusion_map[run_name] = 'gating'
        else:
            fusion_map[run_name] = 'unknown'

    colors = {'concat': '#1f77b4', 'gating': '#ff7f0e', 'gated_concat': '#2ca02c',
              'gated_concatenation': '#2ca02c', 'simple_gated': '#d62728', 'ARGate_Fusion': '#9467bd',
              'unknown': '#888888'}

    # Create violin plots for each error type
    for error_col in signed_error_cols.keys():
        plot_data = []
        for run_name, test_df in test_results.items():
            if error_col in test_df.columns:
                fusion_type = fusion_map.get(run_name, 'unknown')
                errors = test_df[error_col].dropna().values

                for err in errors:
                    plot_data.append({
                        'Fusion Type': fusion_type.upper(),
                        'Signed Error': err,
                        'Run': run_name
                    })

        if not plot_data:
            continue

        plot_df = pd.DataFrame(plot_data)

        # ===== MATPLOTLIB VIOLIN PLOT =====
        fig, ax = plt.subplots(figsize=(12, 6))

        fusion_types = sorted(plot_df["Fusion Type"].unique())
        positions = np.arange(len(fusion_types))
        violin_data = [plot_df[plot_df["Fusion Type"] == ft]["Signed Error"].values for ft in fusion_types]

        parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)

        # Customize colors
        violin_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(violin_colors[i % len(violin_colors)])
            pc.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(fusion_types)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_title(f"Signed Test Error Distribution: {error_col}\n(Skewed left → Underprediction | Right → Overprediction)",
                    fontsize=12, fontweight='bold')
        ax.set_ylabel("Signed Error")
        ax.set_xlabel("Fusion Architecture")
        ax.grid(True, alpha=0.3, axis='y')

        save_figure(fig, output_dir, f"signed_test_error_distribution_{error_col}", fmt)
        plt.close()

        # ===== PLOTLY INTERACTIVE VIOLIN =====
        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            for fusion_type in sorted(plot_df["Fusion Type"].unique()):
                errors = plot_df[plot_df["Fusion Type"] == fusion_type]["Signed Error"].values
                color = colors.get(fusion_type.lower(), '#888888')
                fig.add_trace(go.Violin(
                    x=[fusion_type] * len(errors),
                    y=errors,
                    name=fusion_type,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    marker_color=color
                ))

            fig.update_layout(
                title=f"Signed Test Error Distribution: {error_col}<br><sub>Skewed left → Underprediction | Right → Overprediction</sub>",
                xaxis_title="Fusion Architecture",
                yaxis_title="Signed Error",
                template="plotly_white",
                height=600,
                width=1000,
                showlegend=False,
                violinmode='group'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

            fig.write_html(str(output_dir / f"signed_test_error_distribution_{error_col}_interactive.html"),
                          include_plotlyjs='cdn')

    print(f"✓ Signed test error distribution plots saved to {output_dir}")


def generate_summary_report(master_df: pd.DataFrame, test_results: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Generate a text summary report comparing fusion architectures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())

    with open(output_dir / "fusion_architecture_summary.txt", "w") as f:
        f.write("=" * 90 + "\n")
        f.write("INPUT STREAM FUSION ARCHITECTURE COMPARISON REPORT\n")
        f.write("=" * 90 + "\n\n")

        for fusion_type in fusion_types:
            fuse_df = master_df[master_df["fusion_type"] == fusion_type]
            final_epochs = fuse_df[fuse_df["epoch"] >= fuse_df.groupby("run_name")["epoch"].transform("max") - 4]

            f.write(f"\n{'='*80}\n")
            f.write(f"FUSION ARCHITECTURE: {fusion_type.upper()}\n")
            f.write(f"{'='*80}\n")

            # Validation metrics
            if "val_loss" in final_epochs.columns:
                val_loss_mean = final_epochs["val_loss"].mean()
                val_loss_std = final_epochs["val_loss"].std()
                f.write(f"  Val Loss (Last 5 Epochs):        {val_loss_mean:.5f} ± {val_loss_std:.5f}\n")

            if "val_abs_rot_x" in final_epochs.columns:
                rot_mean = final_epochs["val_abs_rot_x"].mean()
                rot_std = final_epochs["val_abs_rot_x"].std()
                f.write(f"  Val Abs Rotation X (deg):       {rot_mean:.4f} ± {rot_std:.4f}\n")

            if "val_abs_trans" in final_epochs.columns:
                trans_mean = final_epochs["val_abs_trans"].mean()
                trans_std = final_epochs["val_abs_trans"].std()
                f.write(f"  Val Abs Translation (m):        {trans_mean:.5f} ± {trans_std:.5f}\n")

            if "val_abs_y" in final_epochs.columns and "val_abs_z" in final_epochs.columns:
                y_mean = final_epochs["val_abs_y"].mean()
                z_mean = final_epochs["val_abs_z"].mean()
                f.write(f"  Val Abs Y (m):                  {y_mean:.5f}\n")
                f.write(f"  Val Abs Z (m):                  {z_mean:.5f}\n")

            # Convergence rate (early vs late epochs)
            if len(fuse_df) > 0:
                early_epochs = fuse_df[fuse_df["epoch"] <= 5]
                late_epochs = fuse_df[fuse_df["epoch"] >= fuse_df.groupby("run_name")["epoch"].transform("max") - 4]

                if "val_loss" in early_epochs.columns and "val_loss" in late_epochs.columns:
                    early_loss = early_epochs["val_loss"].mean()
                    late_loss = late_epochs["val_loss"].mean()
                    improvement = (early_loss - late_loss) / early_loss * 100 if early_loss > 0 else 0
                    f.write(f"  Loss Improvement (Early→Late):  {improvement:.1f}%\n")

            f.write(f"\n  Seeds Used: {sorted(fuse_df['seed'].unique())}\n")
            f.write(f"  Total Runs: {fuse_df['run_name'].nunique()}\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("FUSION ARCHITECTURE CHARACTERISTICS\n")
        f.write("=" * 90 + "\n")
        f.write("""
CONCATENATION (High-Dimensional):
  - Preserves all raw features from both streams
  - High-dimensional input to fusion module
  - More parameters in fusion layers
  - May preserve redundant information
  → Expected: Slower convergence but potential for better final performance

GATING (Low-Dimensional):
  - Learns per-stream weight (scalar)
  - Collapses features to single vector per stream
  - Compression through gating mechanism
  - Forces feature selection before fusion
  → Expected: Faster convergence but may lose task-specific details

GATED CONCATENATION (Hybrid):
  - Concatenates features but with learned per-stream gates
  - Balances information preservation with selective emphasis
  - Medium-dimensional fusion input
  → Expected: Compromise between convergence and performance

KEY OBSERVATIONS:
1. Rotation vs Translation Trade-off:
   - Does compression hurt rotation or translation more?
   - High-dim preserves modality-specific information better

2. Convergence Rates:
   - High-dim may learn slower but reach better optima
   - Low-dim may converge faster but plateau earlier

3. Error Distributions:
   - Systematic biases (skewed errors) indicate architecture mismatch
   - Symmetric distributions suggest balanced task learning
""")

    print(f"✓ Summary report saved to {output_dir / 'fusion_architecture_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Input Stream Fusion Architecture Analysis Pipeline"
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
        default="analysis_fusion",
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

    args = parser.parse_args()

    runs_dir = Path(args.path)
    output_dir = Path(args.output_dir)

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    print(f"Loading data from {runs_dir}...")
    master_df, run_configs, test_results = load_and_prepare_data(runs_dir)

    print(f"Loaded {len(master_df)} total metric rows from {len(run_configs)} runs")
    print(f"Fusion architectures found: {sorted(master_df['fusion_type'].unique())}")
    print(f"Test results found for {len(test_results)} runs")

    print("\nGenerating analysis plots...")

    # All plots
    plot_rotation_vs_translation_scatter_comparison(master_df, output_dir / "performance", fmt=args.format)
    plot_convergence_comparison(master_df, output_dir / "convergence", fmt=args.format)
    plot_signed_errors_over_epochs(master_df, output_dir / "signed_errors", fmt=args.format)
    plot_signed_test_error_distribution(test_results, output_dir / "test_errors", fmt=args.format)

    # Report
    generate_summary_report(master_df, test_results, output_dir)

    print(f"\n✓ All analysis complete. Output saved to {output_dir}")
    print("\nNote: If plots are missing, check if corresponding metrics are logged in train.py:")
    print("  - Signed errors (e.g., signed_rot_x, signed_y, signed_z)")
    print("  - Test results with per-sample errors")


if __name__ == "__main__":
    main()
