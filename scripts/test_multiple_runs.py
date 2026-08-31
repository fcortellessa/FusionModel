"""
This script runs test.py for multiple run folders, then builds an aggregate
overview (leaderboards + pooled distribution plots) across all runs it processed.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ==============================================================================
# CONFIGURATION: Paste your folders here
# ==============================================================================

# Option 1: Paste an explicit list of run folders
RUN_FOLDERS_TO_BE_INFERENCED = [
    # "runs/run1",
    # "runs/run2",
]

# Subdirectories within the ablation/run folder to completely skip
IGNORE_FOLDERS = ["results", "overview"]

# Test-result columns the overview leaderboards/plots are built from.
METRIC_COLS = ["val_abs_trans", "val_abs_y", "val_abs_z", "val_abs_rot_x", "val_abs_rot_z"]
SIGNED_METRIC_COLS = ["val_signed_y", "val_signed_z", "val_signed_rot_x"]

# Fixed colors for the fusion types this repo actually uses; anything else
# (new fusion strategies, or grouping by a non-fusion key) falls back to a
# deterministic colormap cycle so plots stay reproducible run to run.
_KNOWN_GROUP_COLORS = {
    'concat': '#1f77b4', 'gated': '#ff7f0e', 'gated_concat': '#2ca02c',
    'gated_concatenation': '#2ca02c', 'simple_gated': '#d62728', 'ARGate_Fusion': '#9467bd',
    'gated_Softmax': '#ff7f0e', 'gated_Sigmoid': '#e377c2', 'attention': '#8c564b',
    'spatial_gated_2d': '#17becf', 'residual_gated_concat': '#bcbd22',
    'channelwise_residual_gated_concat': '#7f7f7f',
}


def _group_colors(group_values):
    cmap = plt.get_cmap("tab10")
    colors = {}
    fallback_i = 0
    for g in group_values:
        if g in _KNOWN_GROUP_COLORS:
            colors[g] = _KNOWN_GROUP_COLORS[g]
        else:
            colors[g] = cmap(fallback_i % 10)
            fallback_i += 1
    return colors


def _get_dotted(cfg, dotted_key, default="unknown"):
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node if node is not None else default


def _resolve_run_config_path(folder_path):
    manifest_candidates = [
        os.path.join(folder_path, "manifest.json"),
        os.path.join(folder_path, "run_manifest.json"),
    ]

    for manifest_path in manifest_candidates:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            config_path = manifest.get("config_snapshot_path") or manifest.get("config_path")
            if config_path and os.path.exists(config_path):
                return config_path

    for fallback_name in ["config.yml", "config.resolved.yml"]:
        fallback_path = os.path.join(folder_path, fallback_name)
        if os.path.exists(fallback_path):
            return fallback_path

    raise FileNotFoundError(f"Could not resolve a config file for run folder: {folder_path}")


def _ensure_matching_split(cfg, folder_path):
    from utils.utils import resolve_data_root
    from data_processing.preprocess_data import ensure_test_split_current

    data_root = resolve_data_root(cfg, REPO_ROOT)
    split_seed = ensure_test_split_current(
        cfg, data_root, log=lambda msg: print(f"🔄 {msg} ({folder_path})")
    )
    print(f"✅ Split ready (seed={split_seed}) for {folder_path}")


# ==============================================================================
# EXECUTION LOGIC
# ==============================================================================

def run_inference_on_folder(folder_path, plot_formats):
    """Run test.py on one run folder. Returns a record dict on success, or None on failure."""
    print(f"\n" + "="*80)
    print(f"🚀 Processing: {folder_path}")
    print("="*80 + "\n")

    from utils.utils import load_config

    cfg_path = _resolve_run_config_path(folder_path)
    cfg = load_config(cfg_path)
    _ensure_matching_split(cfg, folder_path)

    # Explicitly specify results subfolder inside the run folder
    results_dir = os.path.join(folder_path, "results")

    test_script = REPO_ROOT / "scripts" / "test.py"
    cmd = [
        sys.executable, str(test_script),
        "--run_folder", folder_path,
        "--results_dir", results_dir,
        "--plot_format", *plot_formats
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while processing: {folder_path}")
        print(f"Details: {e}\n")
        return None

    return {"folder_path": folder_path, "cfg": cfg, "results_dir": results_dir}


# ==============================================================================
# OVERVIEW: leaderboards + pooled per-group distribution plots
# ==============================================================================

def _load_latest_test_csv(results_dir):
    candidates = glob.glob(os.path.join(results_dir, "results_*.csv"))
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    try:
        return pd.read_csv(latest)
    except Exception as e:
        print(f"⚠️ Failed to read test results at {latest}: {e}")
        return None


def _collect_run_records(processed, group_by):
    from utils.utils import resolve_split_seed

    records = []
    for entry in processed:
        df = _load_latest_test_csv(entry["results_dir"])
        if df is None or df.empty:
            print(f"⚠️ No test results found for {entry['folder_path']}, skipping in overview")
            continue

        cfg = entry["cfg"]
        run_name = os.path.basename(os.path.normpath(entry["folder_path"]))
        group_val = _get_dotted(cfg, group_by)
        training_seed = cfg.get("training", {}).get("seed", "?")
        split_seed = resolve_split_seed(cfg)

        records.append({
            "run_name": run_name,
            "group": group_val,
            "training_seed": training_seed,
            "split_seed": split_seed if split_seed is not None else "?",
            "df": df,
        })
    return records


def build_leaderboards(records, sort_metric="val_abs_rot_z"):
    per_run_rows = []
    for rec in records:
        df = rec["df"]
        row = {
            "run_name": rec["run_name"],
            "group": rec["group"],
            "training_seed": rec["training_seed"],
            "split_seed": rec["split_seed"],
            "n_samples": len(df),
        }
        for col in METRIC_COLS:
            if col in df.columns:
                row[f"{col}_mean"] = df[col].mean()
                row[f"{col}_median"] = df[col].median()
                row[f"{col}_std"] = df[col].std()
        per_run_rows.append(row)

    per_run_df = pd.DataFrame(per_run_rows)
    sort_col = f"{sort_metric}_mean"
    if sort_col in per_run_df.columns:
        per_run_df = per_run_df.sort_values(sort_col, na_position="last").reset_index(drop=True)

    per_group_rows = []
    if not per_run_df.empty:
        for group_val, grp in per_run_df.groupby("group"):
            row = {
                "group": group_val,
                "n_runs": len(grp),
                "training_seeds": sorted({s for s in grp["training_seed"] if s != "?"}),
            }
            for col in METRIC_COLS:
                mean_col = f"{col}_mean"
                if mean_col in grp.columns:
                    row[f"{col}_mean_of_runs"] = grp[mean_col].mean()
                    row[f"{col}_std_across_runs"] = grp[mean_col].std()
            per_group_rows.append(row)

    per_group_df = pd.DataFrame(per_group_rows)
    group_sort_col = f"{sort_metric}_mean_of_runs"
    if not per_group_df.empty and group_sort_col in per_group_df.columns:
        per_group_df = per_group_df.sort_values(group_sort_col, na_position="last").reset_index(drop=True)

    return per_run_df, per_group_df


def _write_leaderboard_txt(df, path, title, rank_cols):
    with open(path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 100 + "\n\n")
        for i, row in df.iterrows():
            f.write(f"#{i+1}\n")
            for col in rank_cols:
                if col in df.columns:
                    f.write(f"  {col}: {row[col]}\n")
            f.write("\n")


def write_leaderboards(per_run_df, per_group_df, output_dir, group_by):
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run_df.to_csv(output_dir / "leaderboard_per_run.csv", index=False)
    per_group_df.to_csv(output_dir / "leaderboard_per_group.csv", index=False)

    run_rank_cols = ["run_name", "group", "training_seed", "split_seed", "n_samples"] + \
        [f"{c}_mean" for c in METRIC_COLS] + [f"{c}_median" for c in METRIC_COLS] + [f"{c}_std" for c in METRIC_COLS]
    _write_leaderboard_txt(
        per_run_df, output_dir / "leaderboard_per_run.txt",
        f"PER-RUN LEADERBOARD (grouped by {group_by}, sorted by val_abs_rot_z_mean)",
        run_rank_cols,
    )

    group_rank_cols = ["group", "n_runs", "training_seeds"] + \
        [f"{c}_mean_of_runs" for c in METRIC_COLS] + [f"{c}_std_across_runs" for c in METRIC_COLS]
    _write_leaderboard_txt(
        per_group_df, output_dir / "leaderboard_per_group.txt",
        f"PER-GROUP LEADERBOARD (grouped by {group_by}, sorted by val_abs_rot_z_mean_of_runs)",
        group_rank_cols,
    )

    print(f"✓ Leaderboards saved to {output_dir}")


def _pooled_by_group(records, col):
    pooled = {}
    for rec in records:
        if col not in rec["df"].columns:
            continue
        pooled.setdefault(rec["group"], []).append(rec["df"][col].dropna().values)
    return {g: np.concatenate(v) for g, v in pooled.items() if v}


def plot_pooled_distributions(records, output_dir, formats, group_by):
    """Same shape as test.py's per-run summary plots, but pooling every
    run/seed sharing a group value into one figure per metric so groups are
    directly comparable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted({rec["group"] for rec in records}, key=str)
    colors = _group_colors(groups)

    abs_metrics = [
        ("val_abs_y", "Y-Translation Abs Error [m]"),
        ("val_abs_z", "Z-Translation Abs Error [m]"),
        ("val_abs_rot_x", "Rotation Abs Error (roll) [deg]"),
        ("val_abs_rot_z", "Rotation Abs Error (geodesic) [deg]"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Pooled Test Error Distributions by {group_by}", fontsize=16)
    for ax, (col, label) in zip(axs.flat, abs_metrics):
        pooled = _pooled_by_group(records, col)
        for g in groups:
            if g not in pooled:
                continue
            ax.hist(pooled[g], bins=30, alpha=0.5, label=str(g), color=colors[g], density=True)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    for fmt in [f for f in formats if f in ("png", "svg")]:
        fig.savefig(output_dir / f"overview_abs_error_distributions.{fmt}", dpi=150)
    plt.close(fig)

    # Grouped boxplot comparison, one subplot per metric
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Test Error Comparison by {group_by} (boxplots)", fontsize=16)
    for ax, (col, label) in zip(axs.flat, abs_metrics):
        pooled = _pooled_by_group(records, col)
        present_groups = [g for g in groups if g in pooled]
        data = [pooled[g] for g in present_groups]
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, notch=True, labels=[str(g) for g in present_groups])
        for patch, g in zip(bp['boxes'], present_groups):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.6)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    for fmt in [f for f in formats if f in ("png", "svg")]:
        fig.savefig(output_dir / f"overview_boxplot_by_group.{fmt}", dpi=150)
    plt.close(fig)

    # Signed error distributions (bias direction)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Pooled Signed Test Error Distributions by {group_by}", fontsize=16)
    for ax, col in zip(axs, SIGNED_METRIC_COLS):
        pooled = _pooled_by_group(records, col)
        for g in groups:
            if g not in pooled:
                continue
            ax.hist(pooled[g], bins=30, alpha=0.5, label=str(g), color=colors[g], density=True)
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    for fmt in [f for f in formats if f in ("png", "svg")]:
        fig.savefig(output_dir / f"overview_signed_error_distributions.{fmt}", dpi=150)
    plt.close(fig)

    if "html" in formats and PLOTLY_AVAILABLE:
        fig = go.Figure()
        for col, label in abs_metrics:
            pooled = _pooled_by_group(records, col)
            for g in groups:
                if g not in pooled:
                    continue
                fig.add_trace(go.Box(y=pooled[g], name=f"{label} | {g}", marker_color=colors[g],
                                      legendgroup=str(g), visible=(col == abs_metrics[0][0])))
        fig.update_layout(
            title=f"Test Error Comparison by {group_by} (interactive)",
            template="plotly_white", height=700, width=1100,
        )
        fig.write_html(str(output_dir / "overview_boxplot_by_group_interactive.html"), include_plotlyjs='cdn')

    print(f"✓ Pooled distribution plots saved to {output_dir}")


def generate_overview(processed, overview_dir, plot_formats, group_by):
    records = _collect_run_records(processed, group_by)
    if not records:
        print("⚠️ No test results available to build an overview from.")
        return

    per_run_df, per_group_df = build_leaderboards(records)
    write_leaderboards(per_run_df, per_group_df, overview_dir, group_by)
    plot_pooled_distributions(records, overview_dir, plot_formats, group_by)


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch inference runner")
    parser.add_argument(
        "--plot_format",
        nargs="+",
        choices=["png", "svg", "html", "all"],
        default=["all"],
        help="Plot formats to generate (default: all)"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory containing sub-run folders, or a single run directory."
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="model.fusion",
        help="Dotted config key to group the overview leaderboard/plots by (default: model.fusion)."
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Skip building the aggregate leaderboard/overview plots after processing runs."
    )
    args = parser.parse_args()

    plot_formats = args.plot_format
    if "all" in plot_formats:
        plot_formats = ["png", "svg", "html"]

    folders_to_process = []

    # 1. Add explicitly listed folders from top configuration
    if RUN_FOLDERS_TO_BE_INFERENCED:
        folders_to_process.extend(RUN_FOLDERS_TO_BE_INFERENCED)

    # 2. Add folders specified by --run_dir argument
    if args.run_dir:
        run_path = Path(args.run_dir)
        if run_path.is_dir():
            # Find subdirectories inside run_dir that aren't ignored
            subdirs = [
                str(d) for d in run_path.iterdir()
                if d.is_dir() and d.name not in IGNORE_FOLDERS
            ]

            if subdirs:
                subdirs.sort()
                folders_to_process.extend(subdirs)
            else:
                # If no child run folders exist, assume run_dir itself is the run directory
                folders_to_process.append(str(run_path))
        else:
            print(f"⚠️ Warning: Provided --run_dir does not exist: {args.run_dir}")

    # Remove duplicates while preserving order
    folders_to_process = list(dict.fromkeys(folders_to_process))

    if not folders_to_process:
        print("⚠️ No folders found to process. Please check your CLI arguments or script config.")
        return

    print(f"📋 Total runs found to process: {len(folders_to_process)}")

    # Loop through and run inference sequentially
    processed = []
    for folder in folders_to_process:
        absolute_folder = os.path.abspath(folder)
        if os.path.exists(absolute_folder):
            record = run_inference_on_folder(absolute_folder, plot_formats)
            if record is not None:
                processed.append(record)
        else:
            print(f"⚠️ Warning: Folder path does not exist, skipping: {folder}")

    print("\n✅ All runs processed successfully.")

    if not args.skip_overview and processed:
        if args.run_dir:
            overview_dir = Path(args.run_dir).resolve() / "overview"
        else:
            overview_dir = Path(os.path.commonpath([os.path.abspath(f) for f in folders_to_process])) / "overview"

        print(f"\n📊 Building overview (grouped by {args.group_by}) at {overview_dir} ...")
        generate_overview(processed, overview_dir, plot_formats, args.group_by)


if __name__ == "__main__":
    main()
