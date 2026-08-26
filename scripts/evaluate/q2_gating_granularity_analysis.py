#!/usr/bin/env python3
"""
Q2: Gating Granularity Analysis Pipeline
Compares scalar vs channel-wise gating for gated_concat and residual_gated_concat
Analyzes: gate specialization, weight distributions, task performance trade-offs
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

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def save_figure(fig, output_dir: Path, filename_base: str, fmt=None):
    """
    Save matplotlib figure in specified formats for publication.

    Args:
        fig: matplotlib figure object
        output_dir: directory to save to
        filename_base: filename without extension
        fmt: format(s) to save in - string, list of strings, or None for all
             Options: "png", "svg", "plotly_html"
    """
    plt.tight_layout()

    # Normalize fmt to list
    if fmt is None:
        formats = ["png", "svg"]  # Default to both
    elif isinstance(fmt, str):
        formats = [fmt]
    elif isinstance(fmt, list):
        formats = fmt
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
        elif file_format == "plotly_html":
            # Plotly HTML is handled separately in plot functions
            saved_paths['plotly_html'] = output_dir / f"{filename_base}_interactive.html"

    return saved_paths


def save_plotly_figure(fig, output_dir: Path, filename_base: str):
    """
    Save Plotly figure as interactive HTML.

    Args:
        fig: Plotly figure object
        output_dir: directory to save to
        filename_base: filename without extension
    """
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


def parse_fusion_weights_string(s: str) -> List[float]:
    """Parse fusion weights from CSV format: '[0.4, 0.6]' → [0.4, 0.6]"""
    if pd.isna(s) or not isinstance(s, str):
        return []
    try:
        s = s.strip('[]')
        return [float(x.strip()) for x in s.split(',') if x.strip()]
    except (ValueError, AttributeError):
        return []


def extract_gating_config(run_name: str, cfg: dict) -> Tuple[str, str, int]:
    """
    Extract: (fusion_type, gating_type_or_approx, seed)
    gating_type is either 'scalar' or 'channelwise' (heuristic-based)
    """
    fusion_type = cfg.get("model.fusion", "unknown")
    seed = cfg.get("training.seed", "?")

    # Try to extract gating type from config or infer from fusion name
    gating_type = cfg.get("model.gating_type", "unknown")
    if gating_type == "unknown":
        # Heuristic: if fusion contains "channel" or "per_channel", it's channel-wise
        if "channel" in str(fusion_type).lower():
            gating_type = "channelwise"
        else:
            gating_type = "scalar"

    return (fusion_type, gating_type, seed)


def load_and_prepare_data(runs_dir: Path) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Load all metrics.csv files and extract gating configuration.

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

    # Add fusion and gating type classification
    master_df["fusion_type"] = master_df["run_name"].apply(
        lambda rn: extract_gating_config(rn, run_configs.get(rn, {}))[0]
    )
    master_df["gating_type"] = master_df["run_name"].apply(
        lambda rn: extract_gating_config(rn, run_configs.get(rn, {}))[1]
    )
    master_df["seed"] = master_df["run_name"].apply(
        lambda rn: extract_gating_config(rn, run_configs.get(rn, {}))[2]
    )

    return master_df, run_configs


def compute_gate_specialization(row: pd.Series, rgb_key: str, tactile_key: str) -> Tuple[float, float]:
    """
    Compute gate specialization = fraction_lt_0_1 + fraction_gt_0_9
    (bimodality indicator: how extreme are the gates?)

    Returns:
        (rgb_specialization, tactile_specialization)
    """
    rgb_vals = parse_fusion_weights_string(row.get(rgb_key, ""))
    tactile_vals = parse_fusion_weights_string(row.get(tactile_key, ""))

    rgb_spec = np.nan
    tactile_spec = np.nan

    if rgb_vals:
        rgb_spec = float(np.mean(rgb_vals))
    if tactile_vals:
        tactile_spec = float(np.mean(tactile_vals))

    return rgb_spec, tactile_spec


def plot_gate_specialization_over_training(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Q2.2: Gate Specialization Over Training
    Plot: fraction_lt_0_1 + fraction_gt_0_9 over epochs for scalar vs channel-wise
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by fusion and gating type
    fusion_types = sorted(master_df["fusion_type"].unique())
    gating_types = sorted(master_df["gating_type"].unique())

    colors = {
        'scalar': '#1f77b4',
        'channelwise': '#ff7f0e',
        'unknown': '#888888'
    }
    stream_colors = {
        'rgb': '#2ca02c',
        'tactile': '#d62728'
    }

    # ===== MATPLOTLIB PLOTS =====
    for fusion_type in fusion_types:
        fig, axes = plt.subplots(1, len(gating_types), figsize=(6 * len(gating_types), 5))
        if len(gating_types) == 1:
            axes = [axes]

        for idx, gating_type in enumerate(gating_types):
            ax = axes[idx]
            subset_df = master_df[
                (master_df["fusion_type"] == fusion_type) &
                (master_df["gating_type"] == gating_type)
            ]

            if subset_df.empty:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(f"{fusion_type} - {gating_type}")
                continue

            # Compute specialization for each stream
            # Specialization = fraction_lt_0_1 + fraction_gt_0_9 (measure of bimodality)
            rgb_spec_col = f"train_fusion_gate_fraction_lt_0_1"  # Will parse as list
            tactile_spec_col = f"train_fusion_gate_fraction_gt_0_9"

            # For now, use std as proxy for specialization (higher std = more spread = more selective)
            rgb_std_col = "train_fusion_gate_std"

            # Check if columns exist
            if rgb_std_col not in subset_df.columns:
                ax.text(0.5, 0.5, f"Missing {rgb_std_col}", ha='center', va='center')
                ax.set_title(f"{fusion_type} - {gating_type}")
                continue

            # Parse the string values to extract RGB and Tactile
            # Format: "[rgb_val, tactile_val]"
            def extract_stream_vals(row_vals):
                """Extract RGB and Tactile from string representation."""
                rgb_list = []
                tactile_list = []

                for val in row_vals:
                    parsed = parse_fusion_weights_string(val)
                    if len(parsed) >= 1:
                        rgb_list.append(parsed[0])
                    if len(parsed) >= 2:
                        tactile_list.append(parsed[1])

                rgb_mean = np.mean(rgb_list) if rgb_list else np.nan
                tactile_mean = np.mean(tactile_list) if tactile_list else np.nan

                return rgb_mean, tactile_mean

            # Plot specialization over epochs
            rgb_specs = []
            tactile_specs = []
            epochs = []

            for _, row in subset_df.sort_values("epoch").iterrows():
                rgb_spec, tactile_spec = extract_stream_vals([row.get(rgb_std_col, "")])
                epochs.append(row["epoch"])
                rgb_specs.append(rgb_spec)
                tactile_specs.append(tactile_spec)

            # Aggregate by epoch
            epoch_df = pd.DataFrame({
                'epoch': epochs,
                'rgb_spec': rgb_specs,
                'tactile_spec': tactile_specs
            })
            epoch_agg = epoch_df.groupby('epoch').agg({
                'rgb_spec': ['mean', 'std'],
                'tactile_spec': ['mean', 'std']
            }).reset_index()

            if not epoch_agg.empty:
                # Plot RGB
                rgb_mean = epoch_agg[('rgb_spec', 'mean')].values
                rgb_std = epoch_agg[('rgb_spec', 'std')].values
                epochs_plot = epoch_agg['epoch'].values

                ax.plot(epochs_plot, rgb_mean, label='RGB', linewidth=2.5, marker='o',
                       markersize=3, color=stream_colors['rgb'])
                ax.fill_between(epochs_plot, rgb_mean - rgb_std, rgb_mean + rgb_std,
                               alpha=0.2, color=stream_colors['rgb'])

                # Plot Tactile
                tactile_mean = epoch_agg[('tactile_spec', 'mean')].values
                tactile_std = epoch_agg[('tactile_spec', 'std')].values

                ax.plot(epochs_plot, tactile_mean, label='Tactile', linewidth=2.5, marker='s',
                       markersize=3, color=stream_colors['tactile'])
                ax.fill_between(epochs_plot, tactile_mean - tactile_std, tactile_mean + tactile_std,
                               alpha=0.2, color=stream_colors['tactile'])

            ax.set_title(f"{fusion_type.upper()} - {gating_type.upper()}",
                        fontsize=11, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Gate Specialization (Std Dev)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        save_figure(fig, output_dir, f"gate_specialization_{fusion_type}", fmt)
        plt.close()

        # ===== PLOTLY INTERACTIVE VERSION =====
        if PLOTLY_AVAILABLE and fmt and 'plotly_html' in fmt:
            fig_plotly = make_subplots(
                rows=1, cols=len(gating_types),
                subplot_titles=[f"{gt.upper()}" for gt in gating_types],
                specs=[[{"secondary_y": False} for _ in gating_types]]
            )

            for idx, gating_type in enumerate(gating_types):
                subset_df = master_df[
                    (master_df["fusion_type"] == fusion_type) &
                    (master_df["gating_type"] == gating_type)
                ]

                if not subset_df.empty:
                    rgb_std_col = "train_fusion_gate_std"
                    if rgb_std_col in subset_df.columns:
                        epoch_data = subset_df.groupby("epoch")[rgb_std_col].mean().reset_index()

                        fig_plotly.add_trace(
                            go.Scatter(x=epoch_data["epoch"], y=epoch_data[rgb_std_col],
                                      mode='lines+markers', name=f"{fusion_type} - {gating_type}"),
                            row=1, col=idx+1
                        )

            fig_plotly.update_layout(
                title=f"Gate Specialization Over Training: {fusion_type.upper()}",
                height=500,
                template="plotly_white",
                hovermode='x unified'
            )
            fig_plotly.update_yaxes(title_text="Specialization Score", row=1, col=1)

            save_plotly_figure(fig_plotly, output_dir, f"gate_specialization_{fusion_type}")

    print(f"✓ Gate specialization plots saved to {output_dir}")


def plot_gating_weights_distribution(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Q2.1: Gating Weights Distribution
    Boxplot for scalar gating, Violin plot for channel-wise gating
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())
    gating_types = sorted(master_df["gating_type"].unique())

    for fusion_type in fusion_types:
        # Separate by gating type
        fig, axes = plt.subplots(1, len(gating_types), figsize=(6 * len(gating_types), 6))
        if len(gating_types) == 1:
            axes = [axes]

        for idx, gating_type in enumerate(gating_types):
            ax = axes[idx]
            subset_df = master_df[
                (master_df["fusion_type"] == fusion_type) &
                (master_df["gating_type"] == gating_type)
            ]

            if subset_df.empty:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(f"{fusion_type} - {gating_type}")
                continue

            # Take final epoch only for distribution snapshot
            final_epochs = subset_df[subset_df["epoch"] >= subset_df["epoch"].max() - 2]

            # Extract RGB and Tactile gate means
            gate_means = final_epochs["train_fusion_gate_mean"].values
            plot_data = []

            for val in gate_means:
                parsed = parse_fusion_weights_string(val)
                if len(parsed) >= 1:
                    plot_data.append({'Stream': 'RGB', 'Gate Weight': parsed[0]})
                if len(parsed) >= 2:
                    plot_data.append({'Stream': 'Tactile', 'Gate Weight': parsed[1]})

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Boxplot
                streams = sorted(plot_df['Stream'].unique())
                positions = range(len(streams))
                data_by_stream = [plot_df[plot_df['Stream'] == s]['Gate Weight'].values for s in streams]

                bp = ax.boxplot(data_by_stream, positions=positions, labels=streams,
                               patch_artist=True, widths=0.6)

                # Color boxes
                colors_list = ['#2ca02c', '#d62728']
                for patch, color in zip(bp['boxes'], colors_list[:len(streams)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_title(f"{fusion_type.upper()} - {gating_type.upper()}",
                            fontsize=11, fontweight='bold')
                ax.set_ylabel("Gate Weight Value [0-1]")
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_figure(fig, output_dir, f"gating_weights_distribution_{fusion_type}", fmt)
        plt.close()

    print(f"✓ Gating weights distribution plots saved to {output_dir}")


def plot_rotation_vs_translation_comparison(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Q2.3: Rotation vs Translation Task Performance
    Scatter plot comparing scalar vs channel-wise gating
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())

    for fusion_type in fusion_types:
        fusion_df = master_df[master_df["fusion_type"] == fusion_type]

        if fusion_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        gating_types = sorted(fusion_df["gating_type"].unique())
        colors = {'scalar': '#1f77b4', 'channelwise': '#ff7f0e', 'unknown': '#888888'}
        markers = {'scalar': 'o', 'channelwise': 's'}

        # Take final epochs for evaluation
        for gating_type in gating_types:
            subset_df = fusion_df[fusion_df["gating_type"] == gating_type]
            final_epochs = subset_df[subset_df["epoch"] >= subset_df.groupby("run_name")["epoch"].transform("max") - 4]

            if "val_abs_rot_x" in final_epochs.columns and "val_abs_trans" in final_epochs.columns:
                rot_vals = final_epochs["val_abs_rot_x"].values
                trans_vals = final_epochs["val_abs_trans"].values

                color = colors.get(gating_type, '#000000')
                marker = markers.get(gating_type, 'o')
                ax.scatter(trans_vals, rot_vals, label=gating_type.upper(), s=100, alpha=0.6,
                          color=color, marker=marker)

        ax.set_xlabel("Validation Translation Error [m]", fontsize=12, fontweight='bold')
        ax.set_ylabel("Validation Rotation Error X [deg]", fontsize=12, fontweight='bold')
        ax.set_title(f"Rotation vs Translation Task Performance\n{fusion_type.upper()}",
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, output_dir, f"task_performance_{fusion_type}", fmt)
        plt.close()

    print(f"✓ Task performance scatter plots saved to {output_dir}")


def generate_summary_report(master_df: pd.DataFrame, output_dir: Path):
    """Generate summary report comparing scalar vs channel-wise gating."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())

    with open(output_dir / "gating_granularity_summary.txt", "w") as f:
        f.write("=" * 90 + "\n")
        f.write("Q2: GATING GRANULARITY ANALYSIS - SCALAR VS CHANNEL-WISE\n")
        f.write("=" * 90 + "\n\n")

        for fusion_type in fusion_types:
            f.write(f"\n{'='*80}\n")
            f.write(f"FUSION ARCHITECTURE: {fusion_type.upper()}\n")
            f.write(f"{'='*80}\n")

            fusion_df = master_df[master_df["fusion_type"] == fusion_type]
            gating_types = sorted(fusion_df["gating_type"].unique())

            for gating_type in gating_types:
                subset_df = fusion_df[fusion_df["gating_type"] == gating_type]
                final_epochs = subset_df[subset_df["epoch"] >= subset_df.groupby("run_name")["epoch"].transform("max") - 4]

                f.write(f"\n  GATING TYPE: {gating_type.upper()}\n")
                f.write(f"  {'-'*60}\n")

                if "val_loss" in final_epochs.columns:
                    f.write(f"    Val Loss (Last 5 Epochs): {final_epochs['val_loss'].mean():.5f} ± {final_epochs['val_loss'].std():.5f}\n")

                if "val_abs_rot_x" in final_epochs.columns:
                    f.write(f"    Val Abs Rotation X (deg): {final_epochs['val_abs_rot_x'].mean():.4f} ± {final_epochs['val_abs_rot_x'].std():.4f}\n")

                if "val_abs_trans" in final_epochs.columns:
                    f.write(f"    Val Abs Translation (m):  {final_epochs['val_abs_trans'].mean():.5f} ± {final_epochs['val_abs_trans'].std():.5f}\n")

                f.write(f"\n    Seeds Used: {sorted(subset_df['seed'].unique())}\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 90 + "\n")
        f.write("""
GATE SPECIALIZATION (Q2.2):
  - Higher std/specialization = gates becoming more extreme (bimodal)
  - Lower std/specialization = gates remain uncertain (near 0.5)
  - Increasing trend = learning to specialize during training
  - Flat trend = gates remain uncertain throughout

GATING WEIGHTS DISTRIBUTION (Q2.1):
  Scalar Gating:
    - Single value per stream (RGB or Tactile)
    - Boxplot shows spread across seeds/epochs
    - Higher value = more trust in that stream

  Channel-wise Gating:
    - Multiple values per stream (one per channel)
    - Histogram/violin shows per-channel trust distribution
    - Spread indicates selective channel emphasis

TASK PERFORMANCE (Q2.3):
  - Scalar gating typically: faster convergence, potentially lower final performance
  - Channel-wise gating typically: slower convergence, better selectivity
  - Does specialization = higher performance?
  - Which gating type balances rotation vs translation better?
""")

    print(f"✓ Summary report saved to {output_dir / 'gating_granularity_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Q2: Gating Granularity Analysis (Scalar vs Channel-wise)"
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
        default="analysis_q2_gating_granularity",
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
    master_df, run_configs = load_and_prepare_data(runs_dir)

    print(f"Loaded {len(master_df)} total metric rows from {len(run_configs)} runs")
    print(f"Fusion architectures found: {sorted(master_df['fusion_type'].unique())}")
    print(f"Gating types found: {sorted(master_df['gating_type'].unique())}")

    print("\nGenerating analysis plots...")

    # All plots
    plot_gate_specialization_over_training(master_df, output_dir / "specialization", fmt=args.format)
    plot_gating_weights_distribution(master_df, output_dir / "weights_distribution", fmt=args.format)
    plot_rotation_vs_translation_comparison(master_df, output_dir / "task_performance", fmt=args.format)

    # Report
    generate_summary_report(master_df, output_dir)

    print(f"\n✓ All analysis complete. Output saved to {output_dir}")
    print("\nNote: Ensure your config.yml files have:")
    print("  - model.fusion: 'gated_concat' or 'residual_gated_concat'")
    print("  - model.gating_type: 'scalar' or 'channelwise' (optional, will infer if missing)")
    print("  - training.seed: unique per run")


if __name__ == "__main__":
    main()
