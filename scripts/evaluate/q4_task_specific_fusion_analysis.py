#!/usr/bin/env python3
"""
Q4: Task-Specific Fusion Analysis Pipeline
Compares three ways of handling the rotation/translation branches on top of
a common gated_concat base fusion:
  - shared:               plain gated_concat, one fused vector feeds both heads
                           (no task-specific mechanism; the baseline)
  - task_specific_adapter: gated_concat + a post-fusion task_feature_adapter
                           (late specialization -- re-weights the one shared
                           fused vector per-channel, separately per branch)
  - dual_fusion:           gated_concat + model.dual_fusion (early
                           specialization -- independent fusion module
                           instances per branch, own parameters each)

Research question: is early specialization (dual fusion) better than late
specialization (a task-feature adapter on a shared fusion), and do either
beat the shared-fusion baseline? See configs/generated/Q4_dualFusion_spec.yml
for the full experimental rationale (param overhead, epoch counts, etc.).

Unlike Q1/Q2/Q3, the three variants are not all under one run directory --
the "shared" baseline is deliberately reused from the Q2 gating-granularity
sweep rather than re-run (see Q4_dualFusion_spec.yml), so --path accepts one
or more run directories to search.
"""

import os
import glob
import argparse
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


VARIANT_ORDER = ["shared", "task_specific_adapter", "dual_fusion"]
VARIANT_LABELS = {
    "shared": "Shared (baseline)",
    "task_specific_adapter": "Task-Specific Adapter (late)",
    "dual_fusion": "Dual Fusion (early)",
}
VARIANT_COLORS = {
    "shared": "#1f77b4",
    "task_specific_adapter": "#ff7f0e",
    "dual_fusion": "#2ca02c",
}
VARIANT_MARKERS = {
    "shared": "o",
    "task_specific_adapter": "s",
    "dual_fusion": "^",
}


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
        path = output_dir / f"{filename_base}.{file_format}"
        fig.savefig(str(path), dpi=150, bbox_inches='tight', format=file_format)
        saved_paths[file_format] = path

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


def extract_variant_config(cfg: dict) -> Optional[Tuple[str, int]]:
    """
    Classify a run's flattened config into one of VARIANT_ORDER, restricted
    to runs actually relevant to the Q4 comparison (base fusion ==
    gated_concat, since that's the common architecture the three variants
    are built on -- other fusion types belong to Q2's gating-granularity
    question, not this one).

    Returns:
        (variant, seed) or None if the run isn't part of the Q4 comparison.
    """
    fusion_type = cfg.get("model.fusion", "unknown")
    if fusion_type != "gated_concat":
        return None

    seed = cfg.get("training.seed", "?")
    dual_fusion = bool(cfg.get("model.dual_fusion", False))
    adapter_enabled = bool(cfg.get("model.task_feature_adapter.enabled", False))

    if dual_fusion and adapter_enabled:
        # Not part of the Q4 spec (which composes at most one mechanism at a
        # time) -- skip rather than silently mislabel.
        return None
    elif dual_fusion:
        variant = "dual_fusion"
    elif adapter_enabled:
        variant = "task_specific_adapter"
    else:
        variant = "shared"

    return (variant, seed)


def load_and_prepare_data(runs_dirs: List[Path]) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[str, pd.DataFrame]]:
    """
    Load all metrics.csv and test results files from one or more run
    directories, classify each run into a Q4 variant, and drop runs that
    aren't part of the Q4 comparison (wrong base fusion, or an
    adapter+dual_fusion combination outside the spec).

    Returns:
        (master_df, run_configs, test_results)
    """
    metrics_files = []
    for runs_dir in runs_dirs:
        metrics_files.extend(glob.glob(os.path.join(str(runs_dir), "*/metrics.csv")))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.csv files found in {runs_dirs}")

    all_dfs = []
    run_configs = {}
    test_results = {}
    skipped = []

    for path in metrics_files:
        run_dir = Path(os.path.dirname(path))
        run_name = run_dir.name
        config_path = run_dir / "config.yml"

        cfg = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg = flatten_dict(yaml.safe_load(f) or {})
            except Exception as e:
                print(f"Warning: Failed to load config for {run_name}: {e}")
                continue
        else:
            print(f"Warning: no config.yml for {run_name}, skipping")
            continue

        classification = extract_variant_config(cfg)
        if classification is None:
            skipped.append(run_name)
            continue
        variant, seed = classification

        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["run_name"] = run_name
            df["variant"] = variant
            df["seed"] = seed
            all_dfs.append(df)
            run_configs[run_name] = cfg
        except Exception as e:
            print(f"Error reading metrics for {run_name}: {e}")
            continue

        test_files = glob.glob(os.path.join(str(run_dir), "results/results_*.csv")) + \
                     glob.glob(os.path.join(str(run_dir), "results_*.csv"))
        if test_files:
            try:
                latest_test_file = max(test_files, key=os.path.getctime)
                test_df = pd.read_csv(latest_test_file)
                if not test_df.empty:
                    test_df["run_name"] = run_name
                    test_df["variant"] = variant
                    test_results[run_name] = test_df
            except Exception as e:
                print(f"Warning: Failed to load test results for {run_name}: {e}")

    if not all_dfs:
        raise ValueError("No valid metrics files loaded for the Q4 comparison "
                          "(base fusion must be gated_concat)")

    if skipped:
        print(f"Skipped {len(skipped)} run(s) not part of the Q4 comparison "
              f"(wrong base fusion, or adapter+dual_fusion together): {skipped}")

    master_df = pd.concat(all_dfs, ignore_index=True)
    return master_df, run_configs, test_results


def _final_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """Last 5 epochs per run, for a final-performance estimate."""
    return df[df["epoch"] >= df.groupby("run_name")["epoch"].transform("max") - 4]


def plot_rotation_vs_translation_scatter(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Scatter: val_abs_rot_x vs translation error, one point per run/seed
    (final-5-epochs average), colored/marked by variant. Repeated for the
    combined translation error and for Y-only/Z-only separately, since y/z
    can have different units/scales/behavior and a combined metric can mask
    a per-axis effect.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [v for v in VARIANT_ORDER if v in master_df["variant"].unique()]
    trans_metrics = [
        ("val_abs_trans", "Translation", "trans"),
        ("val_abs_y", "Y Translation", "y"),
        ("val_abs_z", "Z Translation", "z"),
    ]

    for trans_col, trans_label, suffix in trans_metrics:
        if "val_abs_rot_x" not in master_df.columns or trans_col not in master_df.columns:
            print(f"Warning: val_abs_rot_x or {trans_col} not found, skipping scatter plot")
            continue

        per_variant_points = {}
        for variant in variants:
            var_df = master_df[master_df["variant"] == variant]
            final_epochs = _final_epochs(var_df)
            per_run = final_epochs.groupby("run_name").agg(
                val_abs_rot_x=("val_abs_rot_x", "mean"),
                trans_val=(trans_col, "mean"),
                seed=("seed", "first"),
            )
            if not per_run.empty:
                per_variant_points[variant] = per_run

        if not per_variant_points:
            continue

        filename_suffix = "" if suffix == "trans" else f"_{suffix}"

        fig, ax = plt.subplots(figsize=(10, 8))
        for variant, per_run in per_variant_points.items():
            ax.scatter(per_run["trans_val"], per_run["val_abs_rot_x"],
                       label=VARIANT_LABELS.get(variant, variant), s=120, alpha=0.75,
                       color=VARIANT_COLORS.get(variant, '#000000'),
                       marker=VARIANT_MARKERS.get(variant, 'o'))
            for trans, rot, seed in zip(per_run["trans_val"], per_run["val_abs_rot_x"], per_run["seed"]):
                ax.annotate(str(seed), (trans, rot), textcoords="offset points", xytext=(6, 4), fontsize=8)

        ax.set_xlabel(f"Validation {trans_label} Error [m]", fontsize=12, fontweight='bold')
        ax.set_ylabel("Validation Rotation Error X [deg]", fontsize=12, fontweight='bold')
        ax.set_title(f"Rotation vs {trans_label} Task Performance\n(Final 5 Epochs Average, Seed Labeled)",
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        save_figure(fig, output_dir, f"task_performance_by_variant{filename_suffix}", fmt)
        plt.close()

        if PLOTLY_AVAILABLE and fmt and 'plotly_html' in fmt:
            fig_plotly = go.Figure()
            for variant, per_run in per_variant_points.items():
                fig_plotly.add_trace(go.Scatter(
                    x=per_run["trans_val"], y=per_run["val_abs_rot_x"], mode='markers',
                    name=VARIANT_LABELS.get(variant, variant),
                    marker=dict(size=12, opacity=0.75, color=VARIANT_COLORS.get(variant, '#000000')),
                    text=[f"Seed: {s}" for s in per_run["seed"]],
                    hovertemplate=f'<b>%{{fullData.name}}</b><br>{trans_label}: %{{x:.5f}} m<br>Rot: %{{y:.4f}} deg<br>%{{text}}<extra></extra>'
                ))
            fig_plotly.update_layout(
                title=f"Rotation vs {trans_label} Task Performance by Variant",
                xaxis_title=f"Validation {trans_label} Error [m]",
                yaxis_title="Validation Rotation Error X [deg]",
                template="plotly_white", height=700, width=1000, hovermode='closest'
            )
            save_plotly_figure(fig_plotly, output_dir, f"task_performance_by_variant{filename_suffix}")

    print(f"✓ Task performance scatter plots (trans/y/z) saved to {output_dir}")


def plot_convergence_comparison(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Line plots comparing convergence: shared vs task_specific_adapter vs
    dual_fusion. Shows train_*/val_* over epochs for rotation, combined
    translation, and Y/Z translation separately.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "train_loss_trans" not in master_df.columns:
        master_df = master_df.copy()
        master_df["train_loss_trans"] = master_df["loss_y"] + master_df["loss_z"]
        master_df["val_loss_trans"] = master_df["val_loss_y"] + master_df["val_loss_z"]

    variants = [v for v in VARIANT_ORDER if v in master_df["variant"].unique()]
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

    def _draw_ax(ax, col, split_label, log_scale):
        # Log-scale variant skips std shading: mean-std can dip to <=0 near
        # a plateau, undefined on a log axis.
        for variant in variants:
            var_df = master_df[master_df["variant"] == variant].sort_values("epoch")
            by_epoch = var_df.groupby("epoch")[col].agg(['mean', 'std']).reset_index()

            color = VARIANT_COLORS.get(variant, '#000000')
            ax.plot(by_epoch["epoch"], by_epoch["mean"], label=VARIANT_LABELS.get(variant, variant),
                    linewidth=2.5, color=color)
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

    for train_col, val_col, title, ylabel in metrics:
        if train_col not in master_df.columns or val_col not in master_df.columns:
            print(f"Warning: Columns {train_col}/{val_col} not found, skipping")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        _draw_ax(axes[0], train_col, f"Training {title}", log_scale=False)
        _draw_ax(axes[1], val_col, f"Validation {title}", log_scale=False)
        save_figure(fig, output_dir, f"convergence_{train_col.replace('train_', '')}", fmt)
        plt.close()

        # Log-y variant: early-training loss/error values are often orders
        # of magnitude above the plateau, compressing late-epoch differences
        # on a linear axis.
        fig_log, axes_log = plt.subplots(1, 2, figsize=(14, 5))
        _draw_ax(axes_log[0], train_col, f"Training {title}", log_scale=True)
        _draw_ax(axes_log[1], val_col, f"Validation {title}", log_scale=True)
        save_figure(fig_log, output_dir, f"convergence_{train_col.replace('train_', '')}_log", fmt)
        plt.close()

    print(f"✓ Convergence comparison plots (linear + log-scale) saved to {output_dir}")


def plot_signed_errors_over_epochs(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """Line plots of signed (not absolute) errors over training, by variant."""
    output_dir.mkdir(parents=True, exist_ok=True)

    signed_cols = [c for c in master_df.columns if 'signed' in c.lower()]
    if not signed_cols:
        print("Warning: No signed error columns found in metrics")
        return

    variants = [v for v in VARIANT_ORDER if v in master_df["variant"].unique()]

    for signed_col in signed_cols:
        fig, ax = plt.subplots(figsize=(12, 6))

        for variant in variants:
            var_df = master_df[master_df["variant"] == variant].sort_values("epoch")
            by_epoch = var_df.groupby("epoch")[signed_col].agg(['mean', 'std']).reset_index()

            color = VARIANT_COLORS.get(variant, '#000000')
            ax.plot(by_epoch["epoch"], by_epoch["mean"], label=VARIANT_LABELS.get(variant, variant),
                   linewidth=2, marker='o', markersize=3, color=color)
            if by_epoch['std'].notna().any():
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


def plot_parameter_overhead(master_df: pd.DataFrame, output_dir: Path, fmt=None):
    """
    Q4-specific: bar chart of parameter count (total and trainable) per
    variant, and % overhead relative to the shared baseline -- the spec's
    own rationale for choosing residual_gate over channel_attention and for
    giving the adapter variant more epochs hinges on this overhead being
    small and asymmetric between the two mechanisms.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "num_parameters" not in master_df.columns:
        print("Warning: num_parameters not found, skipping parameter overhead plot")
        return

    variants = [v for v in VARIANT_ORDER if v in master_df["variant"].unique()]
    per_variant = master_df.groupby("variant")[["num_parameters", "num_trainable_parameters"]].first()

    baseline_params = per_variant.loc["shared", "num_parameters"] if "shared" in per_variant.index else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    totals = [per_variant.loc[v, "num_parameters"] for v in variants]
    colors = [VARIANT_COLORS.get(v, '#000000') for v in variants]
    ax.bar(range(len(variants)), totals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], rotation=15, ha='right')
    ax.set_ylabel("Total Parameters")
    ax.set_title("Total Parameter Count by Variant", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    if baseline_params:
        pct_overhead = [100.0 * (per_variant.loc[v, "num_parameters"] - baseline_params) / baseline_params
                         for v in variants]
        ax.bar(range(len(variants)), pct_overhead, color=colors, alpha=0.8, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel("Parameter Overhead vs Shared Baseline [%]")
        ax.set_title("Parameter Overhead by Variant", fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, "No 'shared' baseline loaded\n(overhead % unavailable)",
                ha='center', va='center', fontsize=10)

    save_figure(fig, output_dir, "parameter_overhead_by_variant", fmt)
    plt.close()

    print(f"✓ Parameter overhead plot saved to {output_dir}")


def generate_summary_report(master_df: pd.DataFrame, output_dir: Path):
    """Generate a text summary report comparing the three variants."""
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [v for v in VARIANT_ORDER if v in master_df["variant"].unique()]

    with open(output_dir / "task_specific_fusion_summary.txt", "w") as f:
        f.write("=" * 90 + "\n")
        f.write("Q4: TASK-SPECIFIC FUSION COMPARISON REPORT (SHARED vs ADAPTER vs DUAL FUSION)\n")
        f.write("=" * 90 + "\n\n")

        for variant in variants:
            var_df = master_df[master_df["variant"] == variant]
            final_epochs = _final_epochs(var_df)

            f.write(f"\n{'='*80}\n")
            f.write(f"VARIANT: {VARIANT_LABELS.get(variant, variant).upper()}\n")
            f.write(f"{'='*80}\n")

            if "val_loss" in final_epochs.columns:
                f.write(f"  Val Loss (Last 5 Epochs):        {final_epochs['val_loss'].mean():.5f} ± {final_epochs['val_loss'].std():.5f}\n")
            if "val_abs_rot_x" in final_epochs.columns:
                f.write(f"  Val Abs Rotation X (deg):        {final_epochs['val_abs_rot_x'].mean():.4f} ± {final_epochs['val_abs_rot_x'].std():.4f}\n")
            if "val_abs_trans" in final_epochs.columns:
                f.write(f"  Val Abs Translation (m):         {final_epochs['val_abs_trans'].mean():.5f} ± {final_epochs['val_abs_trans'].std():.5f}\n")
            if "val_abs_y" in final_epochs.columns and "val_abs_z" in final_epochs.columns:
                f.write(f"  Val Abs Y (m):                   {final_epochs['val_abs_y'].mean():.5f}\n")
                f.write(f"  Val Abs Z (m):                   {final_epochs['val_abs_z'].mean():.5f}\n")
            if "num_parameters" in final_epochs.columns:
                f.write(f"  Total Parameters:                {int(final_epochs['num_parameters'].iloc[0]):,}\n")
            if "num_trainable_parameters" in final_epochs.columns:
                f.write(f"  Trainable Parameters:             {int(final_epochs['num_trainable_parameters'].iloc[0]):,}\n")
            if "epoch" in var_df.columns:
                f.write(f"  Epochs Trained:                  {var_df.groupby('run_name')['epoch'].max().mean():.0f}\n")

            f.write(f"\n  Seeds Used: {sorted(var_df['seed'].unique())}\n")
            f.write(f"  Total Runs: {var_df['run_name'].nunique()}\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 90 + "\n")
        f.write("""
SHARED (baseline):
  - One gated_concat fusion instance feeds both rotation and translation heads
  - No task-specific mechanism; simplest, fewest parameters

TASK-SPECIFIC ADAPTER (late specialization):
  - Same shared fused vector, then a post-fusion residual-gate adapter
    re-weights it per-channel, separately for rotation vs translation
  - ~5.42% new parameters (near-identity-initialized -- needs more epochs to
    learn a useful deviation from identity, hence the longer training run)

DUAL FUSION (early specialization):
  - Independent fusion module instances per branch (own parameters each),
    so RGB+tactile are combined separately for rotation vs translation
    from the start, rather than sharing one fused vector
  - Only ~1.33% new parameters, no cold-start identity behavior (the gate
    is copied from a trained init), so trained for the same epoch budget
    as the baseline

KEY QUESTIONS:
1. Does either task-specific mechanism beat the shared baseline, and on
   which task (rotation vs translation vs both)?
2. Early vs late specialization: does dual_fusion's from-scratch per-branch
   fusion outperform the adapter's re-weighting of an already-shared
   representation?
3. Is any win worth the extra parameters/training time (see
   parameter_overhead_by_variant plot)?
4. Are results stable across seeds, or is one variant more seed-sensitive?
""")

    print(f"✓ Summary report saved to {output_dir / 'task_specific_fusion_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Q4: Task-Specific Fusion Analysis (Shared vs Adapter vs Dual Fusion)"
    )
    parser.add_argument(
        "--path",
        type=str,
        nargs='+',
        required=True,
        help="One or more run directories to search for metrics.csv (the "
             "shared baseline is typically reused from a different sweep "
             "directory than the adapter/dual_fusion runs -- see "
             "configs/generated/Q4_dualFusion_spec.yml)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_q4_task_specific_fusion",
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

    runs_dirs = [Path(p) for p in args.path]
    for runs_dir in runs_dirs:
        if not runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    output_dir = Path(args.output_dir)

    print(f"Loading data from {runs_dirs}...")
    master_df, run_configs, test_results = load_and_prepare_data(runs_dirs)

    print(f"Loaded {len(master_df)} total metric rows from {len(run_configs)} runs")
    print(f"Variants found: {sorted(master_df['variant'].unique())}")
    for variant in VARIANT_ORDER:
        if variant in master_df["variant"].unique():
            seeds = sorted(master_df[master_df["variant"] == variant]["seed"].unique())
            print(f"  {variant}: seeds {seeds}")

    print("\nGenerating analysis plots...")

    plot_rotation_vs_translation_scatter(master_df, output_dir / "performance", fmt=args.format)
    plot_convergence_comparison(master_df, output_dir / "convergence", fmt=args.format)
    plot_signed_errors_over_epochs(master_df, output_dir / "signed_errors", fmt=args.format)
    plot_parameter_overhead(master_df, output_dir / "performance", fmt=args.format)

    generate_summary_report(master_df, output_dir)

    print(f"\n✓ All analysis complete. Output saved to {output_dir}")
    print("\nExample usage combining the dual_fusion/adapter sweep with the reused baseline:")
    print("  python scripts/evaluate/q4_task_specific_fusion_analysis.py \\")
    print("      --path runs/Q4_dualFusion runs/q2_ScalarVsChannelwise_gradclip \\")
    print("      --output-dir analysis_q4_task_specific_fusion")


if __name__ == "__main__":
    main()
