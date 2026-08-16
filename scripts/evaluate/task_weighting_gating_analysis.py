#!/usr/bin/env python3
"""
Task Weighting & Gating Analysis Pipeline
Compares task_weighting strategies (none, uncertainty, dwa) and visualizes:
1. Absolute vs Signed Errors (Train/Val)
2. Gating Weight Distribution (Violin plots)
3. Gating Sparsity Over Training (Line plots)
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


def extract_task_weighting_config(run_name: str, cfg: dict) -> Tuple[str, str, int]:
    """
    Extract: (task_weighting_type, task_weighting_enabled, seed)

    Returns:
        (task_weighting_type, task_weighting_enabled_str, seed)
        e.g., ("uncertainty", "enabled", 46)
    """
    seed = cfg.get("training.seed", "?")

    task_weighting_enabled = cfg.get("training.task_weighting.enabled", False)
    task_weighting_type = cfg.get("training.task_weighting.type", "none")

    if not task_weighting_enabled:
        return ("none", "disabled", seed)
    else:
        return (task_weighting_type, "enabled", seed)


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


def parse_gate_cell(val) -> List[float]:
    """
    Parse a single gate-column cell into a list of floats.
    Values are logged either as a plain float or as a stringified
    per-stream list, e.g. "[0.4720,0.4668]".
    """
    if pd.isna(val):
        return []
    if isinstance(val, str):
        out = []
        for piece in val.strip("[]").split(","):
            piece = piece.strip()
            if piece:
                try:
                    out.append(float(piece))
                except ValueError:
                    continue
        return out
    return [float(val)]


def compute_gating_sparsity(row: pd.Series) -> float:
    """
    Compute sparsity as the fraction of *individual per-channel* gate values
    near 0 or 1 (i.e., strongly suppressed or strongly trusted channels).

    train.py already computes this correctly, per stream, from the full
    per-channel gate distribution before any averaging (see
    _accumulate_fusion_gate_stats / _summarize_fusion_gate_stats), and logs it
    to *_fusion_gate_fraction_lt_0_1 / *_fusion_gate_fraction_gt_0_9. Those are
    the columns to use.

    NOTE: this must NOT be derived from *_fusion_gate_mean, which is already
    averaged across every channel/sample in the epoch (down to one scalar per
    stream) -- extremity of that aggregate mean is a different, much coarser
    quantity than per-channel sparsity, and will rarely if ever cross a 0.1/0.9
    threshold even when the underlying channels are highly selective.
    """
    frac_cols = [
        c for c in row.index
        if 'fusion_gate_fraction_lt_0_1' in c or 'fusion_gate_fraction_gt_0_9' in c
    ]
    if not frac_cols:
        return np.nan

    frac_values = []
    for col in frac_cols:
        frac_values.extend(parse_gate_cell(row[col]))

    if not frac_values:
        return np.nan

    # Each value is already a per-stream fraction of extreme (<0.1 or >0.9)
    # channels; average across streams/columns for a single sparsity score.
    return float(np.mean(frac_values))


def find_tensorboard_dir(run_dir: Path) -> Optional[Path]:
    """
    Locate the actual TensorBoard log directory for a run.

    train.py's own default (`runs/<run>/tensorboard/<run_name>`) is NOT
    necessarily where events end up: when launched via
    train_multiple_runs.py, --tensorboard_root is overridden to
    logs/tensorboard/<session_tag>/<run_name> instead, and that's where the
    writer actually lands. Rather than guess, parse the ground truth out of
    the run's own train_*.log ("TensorBoard log dir: ..." line), which
    train.py logs unconditionally at startup.
    """
    log_files = sorted(run_dir.glob("train_*.log"))
    if not log_files:
        return None

    pattern = re.compile(r"TensorBoard log dir:\s*(\S+)")
    with open(log_files[0], "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                candidate = Path(m.group(1))
                return candidate if candidate.is_dir() else None
    return None


def extract_raw_gate_samples_from_tb(
    tb_dir: Path,
    tag: str = "fusion/val_weights_distribution",
    max_samples: int = 5000,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """
    Reconstruct an approximate sample of raw per-channel gate weights from a
    logged TensorBoard histogram (the true, unaggregated distribution -- see
    utils/logging_utils.py's `tb_writer.add_histogram("fusion/*_weights_distribution", ...)`).

    TensorBoard histograms store variable-width bucket counts, not the
    original values, so this draws `max_samples` points from the bucket
    midpoints weighted by bucket count -- a faithful reconstruction of the
    empirical distribution for plotting purposes (exact values aren't
    recoverable, but the shape/spread is).

    Uses the LAST logged histogram for `tag`, i.e. the final epoch's
    distribution (most representative of the converged model).
    """
    if not TENSORBOARD_AVAILABLE:
        return None

    ea = EventAccumulator(str(tb_dir), size_guidance={"histograms": 0})
    ea.Reload()
    if tag not in ea.Tags().get("histograms", []):
        return None

    events = ea.Histograms(tag)
    if not events:
        return None

    hist = events[-1].histogram_value
    bucket_limit = np.array(hist.bucket_limit, dtype=float)
    bucket = np.array(hist.bucket, dtype=float)
    if bucket.sum() <= 0:
        return None

    lower_edges = np.concatenate([[bucket_limit[0]], bucket_limit[:-1]])
    midpoints = (lower_edges + bucket_limit) / 2.0

    weights = bucket / bucket.sum()
    rng = np.random.default_rng(seed)
    n = min(max_samples, int(bucket.sum()))
    return rng.choice(midpoints, size=n, p=weights)


def extract_true_gating_weights(
    master_df: pd.DataFrame, runs_dir: Path, max_samples_per_run: int = 5000
) -> Dict[str, np.ndarray]:
    """
    Build per-strategy arrays of reconstructed raw per-channel gate values,
    pulled from each run's TensorBoard histogram rather than the coarse
    per-stream means in metrics.csv. Runs without a discoverable TB
    histogram are skipped (with a warning) rather than silently dropped.
    """
    run_to_strategy = (
        master_df[["run_name", "task_weighting_type"]]
        .drop_duplicates()
        .set_index("run_name")["task_weighting_type"]
        .to_dict()
    )

    by_strategy: Dict[str, List[np.ndarray]] = {}
    n_found, n_missing = 0, 0

    for run_name, strategy in run_to_strategy.items():
        run_dir = runs_dir / run_name
        tb_dir = find_tensorboard_dir(run_dir)
        if tb_dir is None:
            n_missing += 1
            continue

        samples = extract_raw_gate_samples_from_tb(
            tb_dir, tag="fusion/val_weights_distribution", max_samples=max_samples_per_run
        )
        if samples is None:
            n_missing += 1
            continue

        by_strategy.setdefault(strategy, []).append(samples)
        n_found += 1

    print(f"  TensorBoard gate histograms: found for {n_found}/{n_found + n_missing} runs")

    return {strategy: np.concatenate(arrs) for strategy, arrs in by_strategy.items()}


def plot_true_gating_weight_distribution(
    master_df: pd.DataFrame, runs_dir: Path, output_dir: Path, fmt: str = "png"
):
    """
    Violin plots of the TRUE per-channel gating weight distribution
    (reconstructed from each run's TensorBoard histogram), as opposed to
    plot_gating_weight_distribution()'s coarser per-epoch stream-mean view.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not TENSORBOARD_AVAILABLE:
        print("Warning: `tensorboard` package not available; skipping true per-channel violin plot.")
        return

    gating_by_strategy = extract_true_gating_weights(master_df, runs_dir)
    if not gating_by_strategy:
        print("Warning: No TensorBoard gate histograms found; skipping true per-channel violin plot.")
        return

    plot_data = [
        {"Strategy": strategy.upper(), "Gate Weight": val}
        for strategy, values in gating_by_strategy.items()
        for val in values
    ]
    plot_df = pd.DataFrame(plot_data)

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for strategy in sorted(plot_df["Strategy"].unique()):
            strategy_data = plot_df[plot_df["Strategy"] == strategy]["Gate Weight"].values
            fig.add_trace(go.Violin(
                x=[strategy] * len(strategy_data),
                y=strategy_data,
                name=strategy,
                box_visible=True,
                meanline_visible=True,
                points=False,
            ))
        fig.update_layout(
            title="TRUE Per-Channel Gating Weight Distribution by Task Weighting Strategy"
                  "<br><sub>Reconstructed from TensorBoard fusion/val_weights_distribution histograms "
                  "(one point per channel, not per-epoch mean)</sub>",
            xaxis_title="Task Weighting Strategy",
            yaxis_title="Gate Weight Value [0-1]",
            template="plotly_white",
            height=600,
            width=1000,
            showlegend=False,
            violinmode="group",
        )
        fig.write_html(str(output_dir / "true_gating_weight_distribution_violin.html"), include_plotlyjs="cdn")
        print(f"✓ Interactive TRUE per-channel violin plot saved: {output_dir / 'true_gating_weight_distribution_violin.html'}")

    fig, ax = plt.subplots(figsize=(12, 6))
    strategies = sorted(plot_df["Strategy"].unique())
    positions = np.arange(len(strategies))
    violin_data = [plot_df[plot_df["Strategy"] == s]["Gate Weight"].values for s in strategies]
    parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(strategies)
    ax.set_title(
        "TRUE Per-Channel Gating Weight Distribution by Task Weighting Strategy\n"
        "(reconstructed from TensorBoard histograms, one point per channel)",
        fontsize=12, fontweight='bold'
    )
    ax.set_ylabel("Gate Weight Value [0-1]")
    ax.set_xlabel("Task Weighting Strategy")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"true_gating_weight_distribution_violin.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Static TRUE per-channel violin plot saved: {output_dir / f'true_gating_weight_distribution_violin.{fmt}'}")

    print("\n=== TRUE Per-Channel Gating Weight Statistics ===")
    for strategy in sorted(gating_by_strategy.keys()):
        values = gating_by_strategy[strategy]
        print(f"\n{strategy.upper()}:")
        print(f"  Reconstructed sample count: {len(values)}")
        print(f"  Mean: {np.mean(values):.4f}  Std: {np.std(values):.4f}  Median: {np.median(values):.4f}")
        sparsity = np.mean((values < 0.1) | (values > 0.9))
        print(f"  Sparsity (< 0.1 or > 0.9): {sparsity:.4f}")


def load_and_prepare_data(runs_dir: Path) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Load all metrics.csv files and extract task_weighting configuration.

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

    # Add task_weighting classification
    master_df["task_weighting_type"] = master_df["run_name"].apply(
        lambda rn: extract_task_weighting_config(rn, run_configs.get(rn, {}))[0]
    )
    master_df["task_weighting_enabled"] = master_df["run_name"].apply(
        lambda rn: extract_task_weighting_config(rn, run_configs.get(rn, {}))[1]
    )
    master_df["seed"] = master_df["run_name"].apply(
        lambda rn: extract_task_weighting_config(rn, run_configs.get(rn, {}))[2]
    )

    # Compute sparsity for each epoch
    master_df["gating_sparsity"] = master_df.apply(compute_gating_sparsity, axis=1)

    return master_df, run_configs


def plot_absolute_vs_signed_errors(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Plot absolute errors vs signed errors for train/val splits.
    Separate subplots for different task_weighting strategies.
    Creates both Plotly (interactive) and matplotlib (static) versions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = sorted(master_df["task_weighting_type"].unique())
    n_strategies = len(strategies)

    def _plot_per_run(ax, strat_df, col, label, color):
        """Plot one line per run (seed) so runs aren't connected end-to-end."""
        if col not in strat_df.columns:
            return
        first = True
        for _, run_df in strat_df.groupby("run_name"):
            run_df = run_df.sort_values("epoch")
            ax.plot(
                run_df["epoch"], run_df[col],
                label=label if first else None,
                linewidth=2, alpha=0.7, color=color,
            )
            first = False

    # ===== MATPLOTLIB STATIC PLOTS =====
    # 1. Absolute Errors - Rotation
    fig, axes = plt.subplots(1, n_strategies, figsize=(5 * n_strategies, 5))
    if n_strategies == 1:
        axes = [axes]

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strat_df = master_df[master_df["task_weighting_type"] == strategy]

        _plot_per_run(ax, strat_df, "train_abs_rot_x", "Train Abs Rot X", '#1f77b4')
        _plot_per_run(ax, strat_df, "val_abs_rot_x", "Val Abs Rot X", '#ff7f0e')

        ax.set_title(f"Rotation Error: {strategy.upper()}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Absolute Error [deg]")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"absolute_errors_rotation.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Absolute Errors - Translation (Y + Z)
    fig, axes = plt.subplots(2, n_strategies, figsize=(5 * n_strategies, 8))
    if n_strategies == 1:
        axes = axes.reshape(-1, 1)

    for idx, strategy in enumerate(strategies):
        strat_df = master_df[master_df["task_weighting_type"] == strategy]

        # Y axis
        ax_y = axes[0, idx]
        _plot_per_run(ax_y, strat_df, "train_abs_y", "Train Abs Y", '#1f77b4')
        _plot_per_run(ax_y, strat_df, "val_abs_y", "Val Abs Y", '#ff7f0e')
        ax_y.set_title(f"Y Translation Error: {strategy.upper()}", fontsize=11, fontweight='bold')
        ax_y.set_ylabel("Absolute Error [m]")
        ax_y.legend(fontsize=9)
        ax_y.grid(True, alpha=0.3)

        # Z axis
        ax_z = axes[1, idx]
        _plot_per_run(ax_z, strat_df, "train_abs_z", "Train Abs Z", '#2ca02c')
        _plot_per_run(ax_z, strat_df, "val_abs_z", "Val Abs Z", '#9467bd')
        ax_z.set_title(f"Z Translation Error: {strategy.upper()}", fontsize=11, fontweight='bold')
        ax_z.set_xlabel("Epoch")
        ax_z.set_ylabel("Absolute Error [m]")
        ax_z.legend(fontsize=9)
        ax_z.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"absolute_errors_translation.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Signed Errors (if available) - Rotation
    signed_rot_cols = [c for c in master_df.columns if 'signed' in c.lower() and 'rot' in c.lower()]
    if signed_rot_cols:
        fig, axes = plt.subplots(1, n_strategies, figsize=(5 * n_strategies, 5))
        if n_strategies == 1:
            axes = [axes]

        for idx, strategy in enumerate(strategies):
            ax = axes[idx]
            strat_df = master_df[master_df["task_weighting_type"] == strategy]

            for i, col in enumerate(signed_rot_cols):
                _plot_per_run(ax, strat_df, col, col, f'C{i}')

            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.set_title(f"Signed Rotation Error: {strategy.upper()}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Signed Error [deg]")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"signed_errors_rotation.{fmt}", dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Signed Errors - Translation
    signed_trans_cols = [c for c in master_df.columns if 'signed' in c.lower() and ('_y' in c.lower() or '_z' in c.lower())]
    if signed_trans_cols:
        fig, axes = plt.subplots(1, n_strategies, figsize=(5 * n_strategies, 5))
        if n_strategies == 1:
            axes = [axes]

        for idx, strategy in enumerate(strategies):
            ax = axes[idx]
            strat_df = master_df[master_df["task_weighting_type"] == strategy]

            for i, col in enumerate(signed_trans_cols):
                _plot_per_run(ax, strat_df, col, col, f'C{i}')

            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.set_title(f"Signed Translation Error: {strategy.upper()}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Signed Error [m]")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"signed_errors_translation.{fmt}", dpi=150, bbox_inches='tight')
        plt.close()

    # ===== PLOTLY INTERACTIVE VERSION =====
    if PLOTLY_AVAILABLE:
        # Rotation errors
        fig = make_subplots(
            rows=1, cols=n_strategies,
            subplot_titles=[f"{s.upper()}" for s in strategies],
            specs=[[{"secondary_y": False}] * n_strategies]
        )

        for idx, strategy in enumerate(strategies):
            strat_df = master_df[master_df["task_weighting_type"] == strategy]

            for run_i, (_, run_df) in enumerate(strat_df.groupby("run_name")):
                run_df = run_df.sort_values("epoch")
                if "train_abs_rot_x" in run_df.columns:
                    fig.add_trace(
                        go.Scatter(x=run_df["epoch"], y=run_df["train_abs_rot_x"],
                                  mode='lines', name=f"{strategy} Train",
                                  line=dict(color='#1f77b4'), showlegend=(idx==0 and run_i==0)),
                        row=1, col=idx+1
                    )
                if "val_abs_rot_x" in run_df.columns:
                    fig.add_trace(
                        go.Scatter(x=run_df["epoch"], y=run_df["val_abs_rot_x"],
                                  mode='lines', name=f"{strategy} Val",
                                  line=dict(color='#ff7f0e'), showlegend=(idx==0 and run_i==0)),
                        row=1, col=idx+1
                    )

        fig.update_layout(
            title="Absolute Rotation Errors by Task Weighting Strategy",
            height=500,
            template="plotly_white",
            hovermode='x unified'
        )
        fig.update_yaxes(title_text="Error [deg]", row=1, col=1)
        fig.update_xaxes(title_text="Epoch")

        fig.write_html(str(output_dir / "absolute_errors_rotation_interactive.html"), include_plotlyjs='cdn')

    print(f"✓ Absolute/Signed error plots saved to {output_dir}")


def extract_gating_weights(master_df: pd.DataFrame) -> Dict[str, List[np.ndarray]]:
    """
    Extract gating weight distributions per strategy.
    Looks for columns matching: *_fusion_gate_mean (train/val, plus
    rotation/translation branches when dual_fusion is enabled).

    NOTE: these are per-stream means already averaged over every channel and
    sample in the epoch (2 values per row: RGB, tactile) -- this shows how
    the *per-epoch stream-level average* gate value is distributed across
    epochs/seeds, not the true per-channel gate distribution. metrics.csv
    doesn't retain individual per-channel values (only mean/std/fraction/
    percentile summaries), so an exact per-channel violin isn't possible from
    this data -- see compute_gating_sparsity's docstring / the surrounding
    discussion for what would be needed to get one.
    """
    strategies = sorted(master_df["task_weighting_type"].unique())
    gate_cols = [c for c in master_df.columns if re.search(r'fusion_gate_mean$', c)]

    if not gate_cols:
        print("Warning: No gating weight columns found in metrics")
        return {}

    gating_by_strategy = {}
    for strategy in strategies:
        strat_df = master_df[master_df["task_weighting_type"] == strategy]
        gate_values = []

        for col in gate_cols:
            if col in strat_df.columns:
                for val in strat_df[col].dropna().values:
                    gate_values.extend(parse_gate_cell(val))

        gating_by_strategy[strategy] = np.array(gate_values) if gate_values else np.array([])

    return gating_by_strategy


def plot_gating_weight_distribution(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Violin plots of gating weight distributions per strategy.
    Wide distributions → feature-level selectivity matters
    Narrow distributions → scalar gating suffices

    Creates both Plotly (interactive HTML) and matplotlib (static PNG/SVG) versions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gating_by_strategy = extract_gating_weights(master_df)
    if not gating_by_strategy:
        print("Warning: No gating weight data to plot")
        return

    # Prepare data for violin plot
    plot_data = []
    for strategy, values in gating_by_strategy.items():
        if len(values) > 0:
            for val in values:
                plot_data.append({"Strategy": strategy.upper(), "Gate Weight": val})

    if not plot_data:
        print("Warning: No gating weight values found")
        return

    plot_df = pd.DataFrame(plot_data)

    # ===== Plotly Interactive Violin Plot =====
    if PLOTLY_AVAILABLE:
        fig = go.Figure()

        for strategy in sorted(plot_df["Strategy"].unique()):
            strategy_data = plot_df[plot_df["Strategy"] == strategy]["Gate Weight"].values
            fig.add_trace(go.Violin(
                x=[strategy] * len(strategy_data),
                y=strategy_data,
                name=strategy,
                box_visible=True,
                meanline_visible=True,
                points=False
            ))

        fig.update_layout(
            title="Gating Weight Distribution by Task Weighting Strategy<br><sub>Wide → Feature-level Selectivity | Narrow → Scalar Gating Sufficient</sub>",
            xaxis_title="Task Weighting Strategy",
            yaxis_title="Gate Weight Value [0-1]",
            template="plotly_white",
            height=600,
            width=1000,
            showlegend=False,
            violinmode='group'
        )

        fig.write_html(str(output_dir / "gating_weight_distribution_violin.html"), include_plotlyjs='cdn')
        print(f"✓ Interactive Plotly violin plot saved: {output_dir / 'gating_weight_distribution_violin.html'}")

    # ===== Matplotlib Static Violin Plot =====
    fig, ax = plt.subplots(figsize=(12, 6))

    strategies = sorted(plot_df["Strategy"].unique())
    positions = np.arange(len(strategies))

    # Prepare data for manual violin plot
    violin_data = []
    for strategy in strategies:
        values = plot_df[plot_df["Strategy"] == strategy]["Gate Weight"].values
        violin_data.append(values)

    # Create violin plot
    parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)

    # Customize colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(strategies)
    ax.set_title("Gating Weight Distribution by Task Weighting Strategy\n(Wide → Feature-level Selectivity | Narrow → Scalar Gating Sufficient)",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("Gate Weight Value [0-1]")
    ax.set_xlabel("Task Weighting Strategy")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"gating_weight_distribution_violin.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Static matplotlib violin plot saved: {output_dir / f'gating_weight_distribution_violin.{fmt}'}")

    # Statistics summary
    print("\n=== Gating Weight Distribution Statistics ===")
    for strategy in sorted(gating_by_strategy.keys()):
        values = gating_by_strategy[strategy]
        if len(values) > 0:
            print(f"\n{strategy.upper()}:")
            print(f"  Count: {len(values)}")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std: {np.std(values):.4f}")
            print(f"  Min: {np.min(values):.4f}")
            print(f"  Max: {np.max(values):.4f}")
            print(f"  Median: {np.median(values):.4f}")
            # Sparsity: fraction near 0 or 1
            sparsity = np.mean((values < 0.2) | (values > 0.8))
            print(f"  Sparsity (gates < 0.2 or > 0.8): {sparsity:.4f}")


def plot_gating_sparsity_over_training(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Line plot of gating sparsity over training epochs.
    Increasing sparsity → model learns to select channels
    Flat distribution → gates remain uncertain

    Creates both Plotly (interactive) and matplotlib (static) versions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = sorted(master_df["task_weighting_type"].unique())

    # ===== MATPLOTLIB STATIC PLOT =====
    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 5))
    if len(strategies) == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strat_df = master_df[master_df["task_weighting_type"] == strategy]

        # Group by epoch and compute mean sparsity
        sparsity_by_epoch = strat_df.groupby("epoch")["gating_sparsity"].agg(['mean', 'std']).reset_index()

        ax.plot(sparsity_by_epoch["epoch"], sparsity_by_epoch["mean"],
                label="Mean Sparsity", linewidth=2.5, marker='o', markersize=4, color=colors[idx % len(colors)])

        if 'std' in sparsity_by_epoch.columns and sparsity_by_epoch['std'].notna().any():
            ax.fill_between(
                sparsity_by_epoch["epoch"],
                sparsity_by_epoch["mean"] - sparsity_by_epoch["std"],
                sparsity_by_epoch["mean"] + sparsity_by_epoch["std"],
                alpha=0.2, color=colors[idx % len(colors)]
            )

        ax.set_title(f"Gating Sparsity: {strategy.upper()}\n(Increasing → Channel Learning | Flat → Uncertainty)",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Sparsity (Fraction of Extreme Gates)")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f"gating_sparsity_over_training.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()

    # ===== PLOTLY INTERACTIVE VERSION =====
    if PLOTLY_AVAILABLE:
        fig = go.Figure()

        for strategy in strategies:
            strat_df = master_df[master_df["task_weighting_type"] == strategy]
            sparsity_by_epoch = strat_df.groupby("epoch")["gating_sparsity"].agg(['mean', 'std']).reset_index()

            fig.add_trace(go.Scatter(
                x=sparsity_by_epoch["epoch"],
                y=sparsity_by_epoch["mean"],
                mode='lines+markers',
                name=strategy.upper(),
                line=dict(width=2.5),
                error_y=dict(
                    type='data',
                    array=sparsity_by_epoch['std'],
                    visible=True
                )
            ))

        fig.update_layout(
            title="Gating Sparsity Over Training<br><sub>Increasing → Channel Learning | Flat → Uncertainty</sub>",
            xaxis_title="Epoch",
            yaxis_title="Sparsity (Fraction of Extreme Gates)",
            template="plotly_white",
            height=600,
            width=1000,
            hovermode='x unified'
        )
        fig.update_yaxes(range=[0, 1])

        fig.write_html(str(output_dir / "gating_sparsity_over_training_interactive.html"), include_plotlyjs='cdn')
        print(f"✓ Interactive Plotly sparsity plot saved: {output_dir / 'gating_sparsity_over_training_interactive.html'}")

    # Statistics
    print("\n=== Gating Sparsity Over Training ===")
    for strategy in strategies:
        strat_df = master_df[master_df["task_weighting_type"] == strategy]
        sparsity_by_epoch = strat_df.groupby("epoch")["gating_sparsity"].mean()

        if len(sparsity_by_epoch) > 0:
            print(f"\n{strategy.upper()}:")
            print(f"  Initial Sparsity (Epoch {sparsity_by_epoch.index[0]}): {sparsity_by_epoch.iloc[0]:.4f}")
            print(f"  Final Sparsity (Epoch {sparsity_by_epoch.index[-1]}): {sparsity_by_epoch.iloc[-1]:.4f}")
            print(f"  Mean Sparsity: {sparsity_by_epoch.mean():.4f}")
            print(f"  Sparsity Trend (Final - Initial): {sparsity_by_epoch.iloc[-1] - sparsity_by_epoch.iloc[0]:.4f}")

    print(f"\n✓ Gating sparsity plots saved to {output_dir}")


def plot_loss_convergence_comparison(master_df: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """
    Compare loss convergence across task_weighting strategies.
    Creates both Plotly (interactive) and matplotlib (static) versions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = sorted(master_df["task_weighting_type"].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # ===== MATPLOTLIB STATIC PLOT =====
    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 5))
    if len(strategies) == 1:
        axes = [axes]

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strat_df = master_df[master_df["task_weighting_type"] == strategy]

        if "val_loss" in strat_df.columns:
            val_loss_by_epoch = strat_df.groupby("epoch")["val_loss"].agg(['mean', 'std']).reset_index()
            ax.plot(val_loss_by_epoch["epoch"], val_loss_by_epoch["mean"],
                    label="Val Loss", linewidth=2.5, marker='o', markersize=4, color=colors[idx % len(colors)])

            if 'std' in val_loss_by_epoch.columns and val_loss_by_epoch['std'].notna().any():
                ax.fill_between(
                    val_loss_by_epoch["epoch"],
                    val_loss_by_epoch["mean"] - val_loss_by_epoch["std"],
                    val_loss_by_epoch["mean"] + val_loss_by_epoch["std"],
                    alpha=0.2, color=colors[idx % len(colors)]
                )

        ax.set_title(f"Validation Loss: {strategy.upper()}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f"loss_convergence_comparison.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()

    # ===== PLOTLY INTERACTIVE VERSION =====
    if PLOTLY_AVAILABLE:
        fig = go.Figure()

        for strategy in strategies:
            strat_df = master_df[master_df["task_weighting_type"] == strategy]

            if "val_loss" in strat_df.columns:
                val_loss_by_epoch = strat_df.groupby("epoch")["val_loss"].agg(['mean', 'std']).reset_index()

                fig.add_trace(go.Scatter(
                    x=val_loss_by_epoch["epoch"],
                    y=val_loss_by_epoch["mean"],
                    mode='lines+markers',
                    name=strategy.upper(),
                    line=dict(width=2.5),
                    error_y=dict(
                        type='data',
                        array=val_loss_by_epoch['std'],
                        visible=True
                    )
                ))

        fig.update_layout(
            title="Validation Loss Convergence by Task Weighting Strategy",
            xaxis_title="Epoch",
            yaxis_title="Loss Value",
            template="plotly_white",
            height=600,
            width=1000,
            hovermode='x unified'
        )

        fig.write_html(str(output_dir / "loss_convergence_comparison_interactive.html"), include_plotlyjs='cdn')
        print(f"✓ Interactive Plotly loss plot saved: {output_dir / 'loss_convergence_comparison_interactive.html'}")

    print(f"✓ Loss convergence comparison plots saved to {output_dir}")


def generate_summary_report(master_df: pd.DataFrame, output_dir: Path):
    """
    Generate a text summary report comparing task_weighting strategies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = sorted(master_df["task_weighting_type"].unique())

    with open(output_dir / "task_weighting_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TASK WEIGHTING STRATEGY COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")

        for strategy in strategies:
            strat_df = master_df[master_df["task_weighting_type"] == strategy]
            final_epochs = strat_df[strat_df["epoch"] >= strat_df["epoch"].max() - 4]

            f.write(f"\n{'='*60}\n")
            f.write(f"STRATEGY: {strategy.upper()}\n")
            f.write(f"{'='*60}\n")

            if "val_loss" in final_epochs.columns:
                val_loss_mean = final_epochs["val_loss"].mean()
                val_loss_std = final_epochs["val_loss"].std()
                f.write(f"  Val Loss (Last 5 Epochs):        {val_loss_mean:.5f} ± {val_loss_std:.5f}\n")

            if "val_abs_rot_x" in final_epochs.columns:
                rot_mean = final_epochs["val_abs_rot_x"].mean()
                rot_std = final_epochs["val_abs_rot_x"].std()
                f.write(f"  Val Abs Rotation X (deg):       {rot_mean:.4f} ± {rot_std:.4f}\n")

            if "val_abs_y" in final_epochs.columns:
                y_mean = final_epochs["val_abs_y"].mean()
                y_std = final_epochs["val_abs_y"].std()
                f.write(f"  Val Abs Y Translation (m):      {y_mean:.5f} ± {y_std:.5f}\n")

            if "val_abs_z" in final_epochs.columns:
                z_mean = final_epochs["val_abs_z"].mean()
                z_std = final_epochs["val_abs_z"].std()
                f.write(f"  Val Abs Z Translation (m):      {z_mean:.5f} ± {z_std:.5f}\n")

            if "gating_sparsity" in strat_df.columns:
                sparsity = strat_df["gating_sparsity"].mean()
                f.write(f"  Mean Gating Sparsity:            {sparsity:.4f}\n")

            f.write(f"\n  Seeds Used: {sorted(strat_df['seed'].unique())}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 80 + "\n")
        f.write("""
GATING WEIGHT DISTRIBUTION:
  - Wide distribution (high std, values near 0 & 1) → Feature-level selectivity matters
  - Narrow distribution (low std, values near 0.5) → Scalar gating suffices

GATING SPARSITY OVER TRAINING:
  - Increasing trend → Model learns to select important channels
  - Flat trend → Gates remain uncertain about feature importance
  - High values (>0.5) → Strong channel selectivity

TASK WEIGHTING COMPARISON:
  - "none" → No learned weighting between rotation & translation
  - "uncertainty" → Learns per-task uncertainty (homoscedastic)
  - "dwa" → Dynamic weighting adjustment based on gradient magnitudes
""")

    print(f"✓ Summary report saved to {output_dir / 'task_weighting_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Task Weighting & Gating Analysis Pipeline"
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
    print(f"Task weighting strategies found: {sorted(master_df['task_weighting_type'].unique())}")

    print("\nGenerating analysis plots...")

    # All plots
    plot_absolute_vs_signed_errors(master_df, output_dir / "errors", fmt=args.format)
    plot_gating_weight_distribution(master_df, output_dir / "gating", fmt=args.format)
    plot_true_gating_weight_distribution(master_df, runs_dir, output_dir / "gating", fmt=args.format)
    plot_gating_sparsity_over_training(master_df, output_dir / "gating", fmt=args.format)
    plot_loss_convergence_comparison(master_df, output_dir / "convergence", fmt=args.format)

    # Report
    generate_summary_report(master_df, output_dir)

    print(f"\n✓ All analysis complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
