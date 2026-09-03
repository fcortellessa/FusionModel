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
from matplotlib.patches import Patch
import torch
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


def _extract_stream_series(subset_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Parse a "[rgb_val, tactile_val]" string column into per-row RGB/Tactile
    floats, keeping epoch and run_name alongside for later grouping.
    """
    records = []
    for _, row in subset_df.iterrows():
        parsed = parse_fusion_weights_string(row.get(col, ""))
        rgb_val = parsed[0] if len(parsed) >= 1 else np.nan
        tactile_val = parsed[1] if len(parsed) >= 2 else np.nan
        records.append({"epoch": row["epoch"], "rgb": rgb_val, "tactile": tactile_val})
    return pd.DataFrame(records)


def plot_gate_specialization_over_training(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Q2.2: Per-stream (RGB vs Tactile) gate weight over training, one plot per
    fusion type. gating_type is a deterministic function of fusion_type (each
    fusion type is either scalar or channel-wise, never both), so there is no
    scalar/channel-wise split to show per fusion type -- doing so previously
    left one of every pair of subplots structurally empty ("No data").
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())
    stream_colors = {'rgb': '#2ca02c', 'tactile': '#d62728'}
    weights_col = "train_fusion_gate_mean"  # same as "fusion_weights"; "[rgb,tactile]" per epoch

    for fusion_type in fusion_types:
        fusion_df = master_df[master_df["fusion_type"] == fusion_type]

        if fusion_df.empty or weights_col not in fusion_df.columns:
            print(f"Warning: no {weights_col} data for {fusion_type}, skipping")
            continue

        stream_df = _extract_stream_series(fusion_df, weights_col)
        epoch_agg = stream_df.groupby("epoch").agg(
            rgb_mean=("rgb", "mean"), rgb_std=("rgb", "std"),
            tactile_mean=("tactile", "mean"), tactile_std=("tactile", "std"),
        ).reset_index()

        # ===== MATPLOTLIB =====
        fig, ax = plt.subplots(figsize=(8, 5.5))

        epochs_plot = epoch_agg["epoch"].values
        ax.plot(epochs_plot, epoch_agg["rgb_mean"], label='RGB', linewidth=2.5,
                marker='o', markersize=3, color=stream_colors['rgb'])
        ax.fill_between(epochs_plot, epoch_agg["rgb_mean"] - epoch_agg["rgb_std"].fillna(0),
                         epoch_agg["rgb_mean"] + epoch_agg["rgb_std"].fillna(0),
                         alpha=0.2, color=stream_colors['rgb'])

        ax.plot(epochs_plot, epoch_agg["tactile_mean"], label='Tactile', linewidth=2.5,
                marker='s', markersize=3, color=stream_colors['tactile'])
        ax.fill_between(epochs_plot, epoch_agg["tactile_mean"] - epoch_agg["tactile_std"].fillna(0),
                         epoch_agg["tactile_mean"] + epoch_agg["tactile_std"].fillna(0),
                         alpha=0.2, color=stream_colors['tactile'])

        ax.set_title(f"Gate Weight Over Training: {fusion_type.upper()}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gate Weight (mean ± std across seeds)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        save_figure(fig, output_dir, f"gate_specialization_{fusion_type}", fmt)
        plt.close()

        # ===== PLOTLY INTERACTIVE =====
        if PLOTLY_AVAILABLE and fmt and 'plotly_html' in fmt:
            fig_plotly = go.Figure()
            fig_plotly.add_trace(go.Scatter(x=epochs_plot, y=epoch_agg["rgb_mean"],
                                             mode='lines+markers', name='RGB',
                                             line=dict(color=stream_colors['rgb'])))
            fig_plotly.add_trace(go.Scatter(x=epochs_plot, y=epoch_agg["tactile_mean"],
                                             mode='lines+markers', name='Tactile',
                                             line=dict(color=stream_colors['tactile'])))
            fig_plotly.update_layout(
                title=f"Gate Weight Over Training: {fusion_type.upper()}",
                xaxis_title="Epoch", yaxis_title="Gate Weight",
                height=500, width=800, template="plotly_white", hovermode='x unified'
            )
            save_plotly_figure(fig_plotly, output_dir, f"gate_specialization_{fusion_type}")

    print(f"✓ Gate weight-over-training plots saved to {output_dir}")


def plot_gate_specialization_score_over_training(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Q2.2b: Gate specialization score over training, one plot per fusion type.
    specialization = fraction_lt_0_1 + fraction_gt_0_9 (per stream, per epoch),
    i.e. the fraction of gate weights sitting near the extremes 0 or 1 --
    a bimodality/decisiveness indicator distinct from the raw mean gate
    weight plotted by plot_gate_specialization_over_training().
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())
    stream_colors = {'rgb': '#2ca02c', 'tactile': '#d62728'}
    lt_col = "train_fusion_gate_fraction_lt_0_1"
    gt_col = "train_fusion_gate_fraction_gt_0_9"

    for fusion_type in fusion_types:
        fusion_df = master_df[master_df["fusion_type"] == fusion_type]

        if fusion_df.empty or lt_col not in fusion_df.columns or gt_col not in fusion_df.columns:
            print(f"Warning: no {lt_col}/{gt_col} data for {fusion_type}, skipping")
            continue

        lt_df = _extract_stream_series(fusion_df, lt_col)
        gt_df = _extract_stream_series(fusion_df, gt_col)
        spec_df = lt_df.copy()
        spec_df["rgb"] = lt_df["rgb"] + gt_df["rgb"]
        spec_df["tactile"] = lt_df["tactile"] + gt_df["tactile"]

        epoch_agg = spec_df.groupby("epoch").agg(
            rgb_mean=("rgb", "mean"), rgb_std=("rgb", "std"),
            tactile_mean=("tactile", "mean"), tactile_std=("tactile", "std"),
        ).reset_index()

        # ===== MATPLOTLIB =====
        fig, ax = plt.subplots(figsize=(8, 5.5))

        epochs_plot = epoch_agg["epoch"].values
        ax.plot(epochs_plot, epoch_agg["rgb_mean"], label='RGB', linewidth=2.5,
                marker='o', markersize=3, color=stream_colors['rgb'])
        ax.fill_between(epochs_plot, epoch_agg["rgb_mean"] - epoch_agg["rgb_std"].fillna(0),
                         epoch_agg["rgb_mean"] + epoch_agg["rgb_std"].fillna(0),
                         alpha=0.2, color=stream_colors['rgb'])

        ax.plot(epochs_plot, epoch_agg["tactile_mean"], label='Tactile', linewidth=2.5,
                marker='s', markersize=3, color=stream_colors['tactile'])
        ax.fill_between(epochs_plot, epoch_agg["tactile_mean"] - epoch_agg["tactile_std"].fillna(0),
                         epoch_agg["tactile_mean"] + epoch_agg["tactile_std"].fillna(0),
                         alpha=0.2, color=stream_colors['tactile'])

        ax.set_title(f"Gate Specialization Score Over Training: {fusion_type.upper()}",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Specialization Score (frac. < 0.1 + frac. > 0.9, mean ± std across seeds)")
        ax.set_ylim([0, 1])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        save_figure(fig, output_dir, f"gate_specialization_score_{fusion_type}", fmt)
        plt.close()

        # ===== PLOTLY INTERACTIVE =====
        if PLOTLY_AVAILABLE and fmt and 'plotly_html' in fmt:
            fig_plotly = go.Figure()
            fig_plotly.add_trace(go.Scatter(x=epochs_plot, y=epoch_agg["rgb_mean"],
                                             mode='lines+markers', name='RGB',
                                             line=dict(color=stream_colors['rgb'])))
            fig_plotly.add_trace(go.Scatter(x=epochs_plot, y=epoch_agg["tactile_mean"],
                                             mode='lines+markers', name='Tactile',
                                             line=dict(color=stream_colors['tactile'])))
            fig_plotly.update_layout(
                title=f"Gate Specialization Score Over Training: {fusion_type.upper()}",
                xaxis_title="Epoch", yaxis_title="Specialization Score",
                height=500, width=800, template="plotly_white", hovermode='x unified'
            )
            save_plotly_figure(fig_plotly, output_dir, f"gate_specialization_score_{fusion_type}")

    print(f"✓ Gate specialization-score-over-training plots saved to {output_dir}")


def plot_gate_sparsity_over_training_by_seed(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Q2.2c: Gate specialization ("sparsity") score -- fraction_lt_0_1 +
    fraction_gt_0_9, the fraction of gate weights sitting near the decisive
    extremes 0/1 -- plotted one line per seed instead of averaged across
    seeds. plot_gate_specialization_score_over_training's mean +/- std band
    has the same pooling problem as plot_gating_weights_distribution's merged
    violin: a seed that specializes hard (score -> 1) and a seed that never
    moves off ~0.5 (score -> 0) average into a misleading "moderate,
    high-variance" band that describes neither run. Keeping seeds as
    separate lines shows which/how many seeds actually specialize.
    One figure per fusion type, RGB and Tactile side by side.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())
    lt_col = "train_fusion_gate_fraction_lt_0_1"
    gt_col = "train_fusion_gate_fraction_gt_0_9"

    for fusion_type in fusion_types:
        fusion_df = master_df[master_df["fusion_type"] == fusion_type]

        if fusion_df.empty or lt_col not in fusion_df.columns or gt_col not in fusion_df.columns:
            print(f"Warning: no {lt_col}/{gt_col} data for {fusion_type}, skipping")
            continue

        run_names = sorted(fusion_df["run_name"].unique())
        cmap = plt.get_cmap("tab10")

        # ===== MATPLOTLIB =====
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

        per_run_series = []
        for idx, run_name in enumerate(run_names):
            run_df = fusion_df[fusion_df["run_name"] == run_name].sort_values("epoch")
            seed = run_df["seed"].iloc[0]
            lt_df = _extract_stream_series(run_df, lt_col)
            gt_df = _extract_stream_series(run_df, gt_col)
            spec_rgb = lt_df["rgb"] + gt_df["rgb"]
            spec_tactile = lt_df["tactile"] + gt_df["tactile"]
            color = cmap(idx % 10)
            per_run_series.append((seed, lt_df["epoch"], spec_rgb, spec_tactile, color))

            axes[0].plot(lt_df["epoch"], spec_rgb, label=f"seed {seed}", linewidth=2, color=color)
            axes[1].plot(lt_df["epoch"], spec_tactile, label=f"seed {seed}", linewidth=2, color=color)

        for ax, stream_name in zip(axes, ["RGB", "Tactile"]):
            ax.set_title(stream_name, fontsize=11, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')

        axes[0].set_ylabel("Specialization Score (frac. < 0.1 + frac. > 0.9)")
        fig.suptitle(f"Gate Specialization Over Training by Seed: {fusion_type.upper()}",
                     fontsize=13, fontweight='bold')

        save_figure(fig, output_dir, f"gate_sparsity_over_training_by_seed_{fusion_type}", fmt)
        plt.close()

        # ===== PLOTLY INTERACTIVE =====
        if PLOTLY_AVAILABLE and fmt and 'plotly_html' in fmt:
            fig_plotly = make_subplots(rows=1, cols=2, subplot_titles=("RGB", "Tactile"), shared_yaxes=True)
            for seed, epochs, spec_rgb, spec_tactile, color_rgba in per_run_series:
                r, g, b, _ = color_rgba
                color = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                fig_plotly.add_trace(go.Scatter(x=epochs, y=spec_rgb, mode='lines',
                                                 name=f"seed {seed}", line=dict(color=color),
                                                 legendgroup=str(seed)), row=1, col=1)
                fig_plotly.add_trace(go.Scatter(x=epochs, y=spec_tactile, mode='lines',
                                                 name=f"seed {seed}", line=dict(color=color),
                                                 legendgroup=str(seed), showlegend=False), row=1, col=2)
            fig_plotly.update_yaxes(range=[0, 1])
            fig_plotly.update_layout(
                title=f"Gate Specialization Over Training by Seed: {fusion_type.upper()}",
                height=500, width=1100, template="plotly_white", hovermode='x unified'
            )
            save_plotly_figure(fig_plotly, output_dir, f"gate_sparsity_over_training_by_seed_{fusion_type}")

    print(f"✓ Per-seed gate sparsity-over-training plots saved to {output_dir}")


def load_best_epoch_gate_weights(run_dir: Path) -> Optional[torch.Tensor]:
    """
    Load the raw (unaggregated) validation gate-weight tensor saved next to
    checkpoint_best.pth by train.py's _save_raw_gate_weights(). Shape is
    [n_val_samples, 2] for scalar gating or [n_val_samples, 2, n_channels]
    for channel-wise gating; dim 1 is stream order [rgb, tactile] (matches
    model.streams: [rgb_image, digit_image] in config.yml).
    Returns None if the file or key is missing.
    """
    path = run_dir / "gate_weights_best_epoch.pt"
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None
    return payload.get("val_gate_weights")


def plot_gating_weights_distribution(master_df: pd.DataFrame, runs_dir: Path, output_dir: Path, fmt: str = "png"):
    """
    Q2.1: Gating Weights Distribution, from the raw per-sample / per-channel
    best-epoch tensors in runs/<run>/gate_weights_best_epoch.pt (not the
    per-epoch reduced metrics.csv columns, which only ever store an
    already-averaged mean/std and can't show the true within-forward-pass
    spread).

    - Scalar gating: histogram of per-sample gate weight (one value per
      validation sample per stream), pooled across all seeds.
    - Channel-wise gating: violin plot of per-channel mean gate weight
      (mean over validation samples for each of the n_channels, per stream),
      pooled across all seeds -- this is the distribution the "granularity"
      question actually asks about: do channels diverge in which stream
      they trust, or does channel-wise gating collapse to near-uniform?
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())
    stream_colors = {'RGB': '#2ca02c', 'Tactile': '#d62728'}

    for fusion_type in fusion_types:
        fusion_df = master_df[master_df["fusion_type"] == fusion_type]
        if fusion_df.empty:
            continue

        gating_type = fusion_df["gating_type"].iloc[0]
        run_names = sorted(fusion_df["run_name"].unique())

        rgb_chunks, tactile_chunks = [], []
        for run_name in run_names:
            tensor = load_best_epoch_gate_weights(runs_dir / run_name)
            if tensor is None:
                print(f"Warning: no gate_weights_best_epoch.pt for {run_name}, skipping that seed")
                continue
            tensor = tensor.float()

            if tensor.dim() == 2:  # scalar gating: [n_samples, 2]
                rgb_chunks.append(tensor[:, 0])
                tactile_chunks.append(tensor[:, 1])
            elif tensor.dim() == 3:  # channel-wise gating: [n_samples, 2, n_channels]
                per_channel_mean = tensor.mean(dim=0)  # [2, n_channels]
                rgb_chunks.append(per_channel_mean[0])
                tactile_chunks.append(per_channel_mean[1])
            else:
                print(f"Warning: unexpected val_gate_weights shape {tuple(tensor.shape)} for {run_name}, skipping")

        if not rgb_chunks:
            print(f"Warning: no gate weight tensors loaded for {fusion_type}, skipping")
            continue

        rgb_vals = torch.cat(rgb_chunks).numpy()
        tactile_vals = torch.cat(tactile_chunks).numpy()

        fig, ax = plt.subplots(figsize=(7, 6))

        if gating_type == "channelwise":
            parts = ax.violinplot([rgb_vals, tactile_vals], positions=[0, 1], showmeans=True, showextrema=True)
            for body, stream in zip(parts['bodies'], ['RGB', 'Tactile']):
                body.set_facecolor(stream_colors[stream])
                body.set_alpha(0.7)
            for key in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                parts[key].set_color('black')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['RGB', 'Tactile'])
            ax.set_title(f"Gate Weight Distribution: {fusion_type.upper()}\n"
                         f"(Per-Channel Mean, Best Epoch, {len(rgb_chunks)} Seeds, n={len(rgb_vals):,} channels)",
                        fontsize=11, fontweight='bold')
        else:
            bins = np.linspace(0, 1, 41)
            ax.hist(rgb_vals, bins=bins, alpha=0.6, color=stream_colors['RGB'], label='RGB')
            ax.hist(tactile_vals, bins=bins, alpha=0.6, color=stream_colors['Tactile'], label='Tactile')
            ax.set_xlabel("Gate Weight Value [0-1]")
            ax.legend(fontsize=10)
            ax.set_title(f"Gate Weight Distribution: {fusion_type.upper()}\n"
                         f"(Per-Sample, Best Epoch, {len(rgb_chunks)} Seeds, n={len(rgb_vals):,} samples)",
                        fontsize=11, fontweight='bold')

        ax.set_ylabel("Gate Weight Value [0-1]" if gating_type == "channelwise" else "Count")
        if gating_type == "channelwise":
            ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        save_figure(fig, output_dir, f"gating_weights_distribution_{fusion_type}", fmt)
        plt.close()

    print(f"✓ Gating weights distribution plots saved to {output_dir}")


def plot_gate_weight_violin_by_seed(master_df: pd.DataFrame, runs_dir: Path, output_dir: Path, fmt: str = "png"):
    """
    Q2.1b: Same raw per-sample / per-channel best-epoch gate weights as
    plot_gating_weights_distribution, but faceted one violin pair (RGB,
    Tactile) per seed instead of concatenated across all seeds into one
    violin. Pooling seeds there can misrepresent heterogeneous behavior --
    e.g. a seed whose gate collapsed to two clusters near 0/1 concatenated
    with a seed that never moved off ~0.5 produces a blended density that
    describes neither actual run. This shows each seed's own distribution
    side by side so that heterogeneity is visible instead of averaged away.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())
    stream_colors = {'RGB': '#2ca02c', 'Tactile': '#d62728'}

    for fusion_type in fusion_types:
        fusion_df = master_df[master_df["fusion_type"] == fusion_type]
        if fusion_df.empty:
            continue

        gating_type = fusion_df["gating_type"].iloc[0]
        seed_runs = fusion_df[["seed", "run_name"]].drop_duplicates().sort_values("seed")

        seeds, rgb_by_seed, tactile_by_seed = [], [], []
        for _, row in seed_runs.iterrows():
            tensor = load_best_epoch_gate_weights(runs_dir / row["run_name"])
            if tensor is None:
                print(f"Warning: no gate_weights_best_epoch.pt for {row['run_name']}, skipping that seed")
                continue
            tensor = tensor.float()

            if tensor.dim() == 2:  # scalar gating: [n_samples, 2]
                rgb_vals, tactile_vals = tensor[:, 0], tensor[:, 1]
            elif tensor.dim() == 3:  # channel-wise gating: [n_samples, 2, n_channels]
                per_channel_mean = tensor.mean(dim=0)  # [2, n_channels]
                rgb_vals, tactile_vals = per_channel_mean[0], per_channel_mean[1]
            else:
                print(f"Warning: unexpected val_gate_weights shape {tuple(tensor.shape)} for {row['run_name']}, skipping")
                continue

            seeds.append(row["seed"])
            rgb_by_seed.append(rgb_vals.numpy())
            tactile_by_seed.append(tactile_vals.numpy())

        if not seeds:
            print(f"Warning: no gate weight tensors loaded for {fusion_type}, skipping seed-faceted violin")
            continue

        n_seeds = len(seeds)
        fig, ax = plt.subplots(figsize=(max(7, 1.8 * n_seeds), 6))

        # Two violins per seed slot: RGB at 2*i, Tactile at 2*i+1.
        positions_rgb = [2 * i for i in range(n_seeds)]
        positions_tactile = [2 * i + 1 for i in range(n_seeds)]

        parts_rgb = ax.violinplot(rgb_by_seed, positions=positions_rgb, widths=0.8,
                                   showmeans=True, showextrema=True)
        parts_tactile = ax.violinplot(tactile_by_seed, positions=positions_tactile, widths=0.8,
                                       showmeans=True, showextrema=True)
        for body in parts_rgb['bodies']:
            body.set_facecolor(stream_colors['RGB'])
            body.set_alpha(0.7)
        for body in parts_tactile['bodies']:
            body.set_facecolor(stream_colors['Tactile'])
            body.set_alpha(0.7)
        for parts in (parts_rgb, parts_tactile):
            for key in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                parts[key].set_color('black')

        ax.set_xticks([2 * i + 0.5 for i in range(n_seeds)])
        ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Gate Weight Value [0-1]")
        unit = "Per-Channel Mean" if gating_type == "channelwise" else "Per-Sample"
        ax.set_title(f"Gate Weight Distribution by Seed: {fusion_type.upper()}\n"
                     f"({unit}, Best Epoch)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        legend_elems = [
            Patch(facecolor=stream_colors['RGB'], alpha=0.7, label='RGB'),
            Patch(facecolor=stream_colors['Tactile'], alpha=0.7, label='Tactile'),
        ]
        ax.legend(handles=legend_elems, fontsize=9, loc='upper right')

        save_figure(fig, output_dir, f"gating_weights_distribution_by_seed_{fusion_type}", fmt)
        plt.close()

    print(f"✓ Seed-faceted gating weights distribution plots saved to {output_dir}")


_SCALE_STREAM_COL_LABELS = ["P3-RGB", "P3-Tac", "P4-RGB", "P4-Tac", "P5-RGB", "P5-Tac"]


def _has_scale_columns(master_df: pd.DataFrame, prefix: str, value_kind: str) -> bool:
    cols = [f"{prefix}_fusion_gate_{s}_{value_kind}" for s in ("p3", "p4", "p5")]
    return all(c in master_df.columns for c in cols)


def _heatmap_text_color(val: float) -> str:
    return "white" if (val < 0.25 or val > 0.75) else "black"


def plot_scale_stream_summary_heatmap(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Q2.4a: Per-scale (P3/P4/P5) x per-stream (RGB/Tactile) mean gate weight,
    one row per fusion type -- final-5-epochs average per run, then averaged
    across seeds. Requires the per-scale logging columns added to
    scripts/train.py (train_fusion_gate_p3_mean etc, channel-wise 1D-pipeline
    gates only via PoseEstimationNet.stream_scale_dims); runs predating that
    change won't have these columns and are skipped with a warning rather
    than silently plotting blanks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "val"

    if not _has_scale_columns(master_df, prefix, "mean"):
        print(f"Warning: no {prefix}_fusion_gate_p{{3,4,5}}_mean columns found -- rerun with "
              f"the updated per-scale logging (scripts/train.py) to populate them. Skipping "
              f"scale/stream summary heatmap.")
        return

    fusion_types = sorted(master_df["fusion_type"].unique())
    rows, row_labels = [], []

    for fusion_type in fusion_types:
        fusion_df = master_df[master_df["fusion_type"] == fusion_type]
        final_epochs = fusion_df[fusion_df["epoch"] >= fusion_df.groupby("run_name")["epoch"].transform("max") - 4]

        per_seed_vals = []
        for run_name, run_df in final_epochs.groupby("run_name"):
            vals, ok = [], True
            for scale in ("p3", "p4", "p5"):
                col = f"{prefix}_fusion_gate_{scale}_mean"
                parsed = [parse_fusion_weights_string(s) for s in run_df.get(col, [])]
                parsed = [p for p in parsed if len(p) == 2]
                if not parsed:
                    ok = False
                    break
                vals.extend(np.mean(np.array(parsed), axis=0).tolist())
            if ok:
                per_seed_vals.append(vals)

        if not per_seed_vals:
            continue
        row_labels.append(fusion_type)
        rows.append(np.mean(np.array(per_seed_vals), axis=0))

    if not rows:
        print("Warning: no fusion type had usable per-scale gate data, skipping summary heatmap")
        return

    matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(_SCALE_STREAM_COL_LABELS)), max(3, 1.0 * len(row_labels) + 1)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(_SCALE_STREAM_COL_LABELS)))
    ax.set_xticklabels(_SCALE_STREAM_COL_LABELS)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                     color=_heatmap_text_color(matrix[i, j]), fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Gate Weight [0=Tactile-trust, 1=RGB-trust]")
    ax.set_title("Per-Scale / Per-Stream Gate Trust by Fusion Type\n(Final 5 Epochs, Averaged Across Seeds)",
                 fontsize=12, fontweight='bold')

    save_figure(fig, output_dir, "scale_stream_summary_heatmap", fmt)
    plt.close()

    print(f"✓ Scale/stream summary heatmap saved to {output_dir}")


def plot_scale_stream_heatmap_by_seed(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Q2.4b: Same per-scale x per-stream mean gate weight as
    plot_scale_stream_summary_heatmap, but one small 3x2 tile per (fusion_type,
    seed) instead of averaged across seeds -- companion view showing whether
    all seeds agree on which scale/stream a fusion type leans on, or whether
    that varies by seed the way the raw weight distribution did (see
    plot_gate_weight_violin_by_seed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "val"

    if not _has_scale_columns(master_df, prefix, "mean"):
        print(f"Warning: no {prefix}_fusion_gate_p{{3,4,5}}_mean columns found -- rerun with "
              f"the updated per-scale logging (scripts/train.py) to populate them. Skipping "
              f"per-seed scale/stream heatmaps.")
        return

    fusion_types = sorted(master_df["fusion_type"].unique())
    row_labels, col_labels = ["P3", "P4", "P5"], ["RGB", "Tactile"]

    for fusion_type in fusion_types:
        fusion_df = master_df[master_df["fusion_type"] == fusion_type]
        final_epochs = fusion_df[fusion_df["epoch"] >= fusion_df.groupby("run_name")["epoch"].transform("max") - 4]
        run_names = sorted(final_epochs["run_name"].unique())

        tiles, seeds = [], []
        for run_name in run_names:
            run_df = final_epochs[final_epochs["run_name"] == run_name]
            matrix_rows, ok = [], True
            for scale in ("p3", "p4", "p5"):
                col = f"{prefix}_fusion_gate_{scale}_mean"
                parsed = [parse_fusion_weights_string(s) for s in run_df.get(col, [])]
                parsed = [p for p in parsed if len(p) == 2]
                if not parsed:
                    ok = False
                    break
                matrix_rows.append(np.mean(np.array(parsed), axis=0))
            if not ok:
                continue
            tiles.append(np.array(matrix_rows))
            seeds.append(run_df["seed"].iloc[0])

        if not tiles:
            print(f"Warning: no per-scale gate data for {fusion_type}, skipping")
            continue

        n = len(tiles)
        # save_figure() unconditionally calls plt.tight_layout() -- a colorbar spanning
        # a *list* of axes (fig.colorbar(im, ax=axes)) isn't compatible with that (either
        # overlaps a subplot's ticks, or raises with constrained_layout engine mismatch).
        # Attaching the colorbar to a single Axes instead is a layout tight_layout
        # understands natively, so give the last tile its own colorbar rather than
        # sharing one across all tiles.
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 0.6, 3.6))
        axes = [axes] if n == 1 else list(axes)
        for idx, (ax, tile, seed) in enumerate(zip(axes, tiles, seeds)):
            im = ax.imshow(tile, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(2))
            ax.set_xticklabels(col_labels)
            ax.set_yticks(range(3))
            ax.set_yticklabels(row_labels)
            ax.set_title(f"seed {seed}", fontsize=10)
            for i in range(3):
                for j in range(2):
                    ax.text(j, i, f"{tile[i, j]:.2f}", ha="center", va="center",
                             color=_heatmap_text_color(tile[i, j]), fontsize=8)
            if idx == n - 1:
                fig.colorbar(im, ax=ax, shrink=0.85, label="Mean Gate Weight")
        fig.suptitle(f"Per-Scale / Per-Stream Gate Trust by Seed: {fusion_type.upper()}\n(Final 5 Epochs)",
                     fontsize=12, fontweight='bold')

        save_figure(fig, output_dir, f"scale_stream_heatmap_by_seed_{fusion_type}", fmt)
        plt.close()

    print(f"✓ Per-seed scale/stream heatmaps saved to {output_dir}")


def plot_scale_stream_specialization_over_training(master_df: pd.DataFrame, output_dir: Path,
                                                     fmt: str = "png", n_epoch_bins: int = 20):
    """
    Q2.4c: Per-scale (P3/P4/P5) x per-stream (RGB/Tactile) specialization score
    (fraction_lt_0_1 + fraction_gt_0_9, logged directly as
    train_fusion_gate_{scale}_specialization -- see scripts/train.py) over
    training, one heatmap per fusion type: rows = epoch (binned into
    n_epoch_bins if there are more epochs than that), columns = scale x
    stream. Averaged across seeds -- see plot_gate_sparsity_over_training_by_seed
    for the per-seed line-plot version of the (non-scale-broken-down) score;
    a per-seed x per-scale x per-epoch view would be 4D and doesn't fit one
    legible heatmap, so this trades seed detail for scale detail.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "train"

    if not _has_scale_columns(master_df, prefix, "specialization"):
        print(f"Warning: no {prefix}_fusion_gate_p{{3,4,5}}_specialization columns found -- "
              f"rerun with the updated per-scale logging (scripts/train.py) to populate them. "
              f"Skipping per-scale specialization-over-training heatmaps.")
        return

    fusion_types = sorted(master_df["fusion_type"].unique())

    for fusion_type in fusion_types:
        fusion_df = master_df[master_df["fusion_type"] == fusion_type]

        per_scale_epoch_means = {}
        for scale in ("p3", "p4", "p5"):
            col = f"{prefix}_fusion_gate_{scale}_specialization"
            stream_df = _extract_stream_series(fusion_df, col)
            per_scale_epoch_means[scale] = stream_df.groupby("epoch").agg(
                rgb=("rgb", "mean"), tactile=("tactile", "mean")
            )

        combined = pd.concat(per_scale_epoch_means, axis=1)
        combined.columns = [f"{scale}_{stream}" for scale, stream in combined.columns]
        combined = combined[["p3_rgb", "p3_tactile", "p4_rgb", "p4_tactile", "p5_rgb", "p5_tactile"]]
        combined = combined.sort_index().dropna(how="all")

        if combined.empty:
            print(f"Warning: no per-scale specialization data for {fusion_type}, skipping")
            continue

        n_epochs = len(combined)
        if n_epochs > n_epoch_bins:
            bin_edges = np.linspace(0, n_epochs, n_epoch_bins + 1).astype(int)
            binned_rows, bin_labels = [], []
            for i in range(len(bin_edges) - 1):
                chunk = combined.iloc[bin_edges[i]:bin_edges[i + 1]]
                if chunk.empty:
                    continue
                binned_rows.append(chunk.mean(axis=0))
                bin_labels.append(f"{int(chunk.index.min())}-{int(chunk.index.max())}")
            plot_df = pd.DataFrame(binned_rows, index=bin_labels)
            y_label = "Epoch (binned)"
        else:
            plot_df = combined.copy()
            plot_df.index = [str(int(e)) for e in plot_df.index]
            y_label = "Epoch"

        matrix_vals = plot_df.values.astype(float)

        fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(plot_df) + 1)))
        im = ax.imshow(matrix_vals, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(_SCALE_STREAM_COL_LABELS)))
        ax.set_xticklabels(_SCALE_STREAM_COL_LABELS)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df.index, fontsize=7)
        ax.set_ylabel(y_label)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Specialization Score (frac. < 0.1 + frac. > 0.9)")
        ax.set_title(f"Per-Scale Gate Specialization Over Training: {fusion_type.upper()}\n"
                     f"(Averaged Across Seeds)", fontsize=12, fontweight='bold')

        save_figure(fig, output_dir, f"scale_stream_specialization_over_training_{fusion_type}", fmt)
        plt.close()

    print(f"✓ Per-scale specialization-over-training heatmaps saved to {output_dir}")


def plot_rotation_vs_translation_comparison(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Q2.3: Rotation vs Translation Task Performance.
    Single scatter plot with all 4 fusion methods as legend entries (one
    point per run/seed, final-5-epochs average), plus a second version with
    the seed number annotated next to each point. Repeated for the combined
    translation error and for the Y-only / Z-only translation error
    separately, since y/z can have different units/scales/behavior and a
    combined metric can mask a per-axis effect.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_types = sorted(master_df["fusion_type"].unique())
    colors = {'gated_concat': '#1f77b4', 'residual_gated_concat': '#ff7f0e',
              'channelwise_gated_concat': '#2ca02c', 'channelwise_residual_gated_concat': '#d62728'}
    markers = {'gated_concat': 'o', 'residual_gated_concat': 's',
               'channelwise_gated_concat': '^', 'channelwise_residual_gated_concat': 'D'}

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

        # ===== Plot 1: all 4 fusion methods on one scatter =====
        fig, ax = plt.subplots(figsize=(10, 8))
        for fusion_type, per_run in per_fusion_points.items():
            color = colors.get(fusion_type, '#000000')
            marker = markers.get(fusion_type, 'o')
            ax.scatter(per_run["trans_val"], per_run["val_abs_rot_x"], label=fusion_type.upper(),
                       s=100, alpha=0.7, color=color, marker=marker)

        ax.set_xlabel(f"Validation {trans_label} Error [m]", fontsize=12, fontweight='bold')
        ax.set_ylabel("Validation Rotation Error X [deg]", fontsize=12, fontweight='bold')
        # ax.set_title(f"Rotation vs {trans_label} Task Performance\n(Final 5 Epochs Average, All Fusion Methods)",
        #            fontsize=13, fontweight='bold')
        ax.set_title(f"Rotation vs {trans_label} Task Performance",
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
        # ax.set_title(f"Rotation vs {trans_label} Task Performance\n(Final 5 Epochs Average, Seed Labeled)",
        #            fontsize=13, fontweight='bold')
        ax.set_title(f"Rotation vs {trans_label} Task Performance\n",
                   fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, output_dir, f"task_performance_all_fusion_methods_seeds{filename_suffix}", fmt)
        plt.close()

    print(f"✓ Task performance scatter plots (trans/y/z) saved to {output_dir}")


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
    plot_gate_specialization_score_over_training(master_df, output_dir / "specialization", fmt=args.format)
    plot_gate_sparsity_over_training_by_seed(master_df, output_dir / "specialization", fmt=args.format)
    plot_gating_weights_distribution(master_df, runs_dir, output_dir / "weights_distribution", fmt=args.format)
    plot_gate_weight_violin_by_seed(master_df, runs_dir, output_dir / "weights_distribution", fmt=args.format)
    plot_scale_stream_summary_heatmap(master_df, output_dir / "scale_breakdown", fmt=args.format)
    plot_scale_stream_heatmap_by_seed(master_df, output_dir / "scale_breakdown", fmt=args.format)
    plot_scale_stream_specialization_over_training(master_df, output_dir / "scale_breakdown", fmt=args.format)
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
