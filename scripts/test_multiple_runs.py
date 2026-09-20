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
import matplotlib.colors as mcolors

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

# Subdirectories within the ablation/run folder to completely skip -- these
# aren't run folders themselves (no manifest.json/config.yml inside), they're
# other output trees that live alongside the run folders.
IGNORE_FOLDERS = ["results", "overview", "tensorboard"]

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
    # Without fixed entries these fell back to tab10[0]/[1], which is concat's/gated's
    # own blue/orange. Same colors as scatter_test_results_by_fusion.py's candidates 5/6.
    'channelwise_gated_concat_local': '#8172B2',
    'channelwise_residual_gated_concat_local': '#937860',
    # dual_fusion builds two independent instances of the base fusion module
    # (own params per task branch, see pose_estimation_model.py) -- a
    # genuinely different architecture from the shared-fusion variant, so it
    # gets its own color rather than sharing the base fusion's slot.
    'gated_concat (dual_fusion)': '#393b79',
}


def _group_colors(group_values):
    cmap = plt.get_cmap("tab10")
    colors = {}
    fallback_i = 0
    for g in group_values:
        if g in _KNOWN_GROUP_COLORS:
            colors[g] = _KNOWN_GROUP_COLORS[g]
        else:
            # matplotlib.colors.to_hex, not the raw RGBA tuple cmap() returns --
            # plotly's marker_color rejects tuples, and hex works for both libs.
            colors[g] = mcolors.to_hex(cmap(fallback_i % 10))
            fallback_i += 1
    return colors


def _get_dotted(cfg, dotted_key, default="unknown"):
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node if node is not None else default


def _group_label(cfg, group_by):
    """Resolve the group value for group_by, but when grouping by fusion
    (model.fusion) fold in model.dual_fusion -- a dual_fusion run builds two
    independent instances of the fusion module (separate params per task
    branch) rather than sharing one, so it's a different architecture from
    the base fusion type and must not be pooled into the same bucket.
    """
    val = _get_dotted(cfg, group_by)
    if group_by == "model.fusion" and _get_dotted(cfg, "model.dual_fusion", False):
        val = f"{val} (dual_fusion)"
    return val


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


def _split_seed_sort_key(folder_path):
    """Sort key putting runs that share a split seed next to each other.
    Folders without a resolvable config/seed sort last (they get skipped later)."""
    from utils.utils import load_config, resolve_split_seed

    try:
        seed = resolve_split_seed(load_config(_resolve_run_config_path(folder_path)))
    except FileNotFoundError:
        seed = None
    return (seed is None, seed if seed is not None else 0)


def _existing_results_record(folder_path):
    """Same record run_inference_on_folder returns, but for a run whose test results
    already exist (used by --overview-only). None if the folder has no results."""
    from utils.utils import load_config

    results_dir = os.path.join(folder_path, "results")
    if not glob.glob(os.path.join(results_dir, "results_*.csv")):
        print(f"⚠️ Skipping {folder_path}: no results/results_*.csv")
        return None
    try:
        cfg = load_config(_resolve_run_config_path(folder_path))
    except FileNotFoundError as e:
        print(f"⚠️ Skipping {folder_path}: {e}")
        return None
    return {"folder_path": folder_path, "cfg": cfg, "results_dir": results_dir}


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

    try:
        cfg_path = _resolve_run_config_path(folder_path)
        cfg = load_config(cfg_path)
        _ensure_matching_split(cfg, folder_path)
    except FileNotFoundError as e:
        # Folder isn't actually a run folder (e.g. an unanticipated non-run
        # subdirectory under --run_dir) -- skip it instead of aborting the
        # whole batch so one bad folder can't swallow every run behind it.
        print(f"⚠️ Skipping {folder_path}: {e}\n")
        return None

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
        group_val = _group_label(cfg, group_by)
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


def _violin_panel(ax, pooled, groups, colors, label, log=False):
    """One violin per group, sharing a single axis so groups sit side by
    side instead of overlapping -- unlike alpha-blended histograms, colors
    never blend into mud and each group's shape stays fully legible.

    Test-error metrics are non-negative and span ~4-5 orders of magnitude
    (sub-micron translation errors up to multi-degree outliers), which
    collapses a linear-scale histogram/violin into one spike at zero with an
    invisible tail. When log=True the KDE is computed in log10 space instead
    (after flooring exact zeros to half the smallest positive value observed,
    since log10(0) is undefined) and the y-ticks are relabeled back to the
    metric's real units so the axis still reads in meters/degrees.
    """
    present = [g for g in groups if g in pooled and len(pooled[g])]
    if not present:
        return

    if log:
        all_vals = np.concatenate([pooled[g] for g in present])
        positive = all_vals[all_vals > 0]
        floor = positive.min() * 0.5 if positive.size else 1e-9
        plot_data = [np.log10(np.clip(pooled[g], floor, None)) for g in present]
    else:
        plot_data = [pooled[g] for g in present]

    parts = ax.violinplot(plot_data, showmedians=True, showextrema=True, widths=0.8)
    for body, g in zip(parts["bodies"], present):
        body.set_facecolor(colors[g])
        body.set_edgecolor(colors[g])
        body.set_alpha(0.65)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1)

    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([str(g) for g in present], rotation=20, ha="right", fontsize=8)

    if log:
        ymin, ymax = ax.get_ylim()
        tick_locs = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
        ax.set_yticks(tick_locs)
        ax.set_yticklabels([f"{10 ** t:.3g}" for t in tick_locs])

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylabel(label + (" (log scale)" if log else ""))
    ax.grid(True, alpha=0.3, axis="y")


def _boxplot_panel(ax, pooled, groups, colors, label, showfliers=False):
    """One boxplot per group on a shared axis -- factored out so the abs-error
    and signed-error boxplot figures don't duplicate this.

    Fliers are hidden by default: with thousands of per-sample errors per group the
    outlier dots swamp the (tiny) boxes; the tails stay visible in the violin plots."""
    present_groups = [g for g in groups if g in pooled]
    data = [pooled[g] for g in present_groups]
    if not data:
        return
    bp = ax.boxplot(data, patch_artist=True, notch=True, showfliers=showfliers,
                    labels=[str(g) for g in present_groups])
    for patch, g in zip(bp["boxes"], present_groups):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.6)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylabel(label)
    # rotation alone (tick_params) leaves labels center-anchored, which
    # collides adjacent long group names -- match _violin_panel's ha="right".
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def _pooled_by_group(records, col):
    pooled = {}
    for rec in records:
        if col not in rec["df"].columns:
            continue
        pooled.setdefault(rec["group"], []).append(rec["df"][col].dropna().values)
    return {g: np.concatenate(v) for g, v in pooled.items() if v}


def plot_pooled_distributions(records, output_dir, formats, group_by, title_suffix=""):
    """Same shape as test.py's per-run summary plots, but pooling every
    run/seed sharing a group value into one figure per metric so groups are
    directly comparable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted({rec["group"] for rec in records}, key=str)
    colors = _group_colors(groups)
    group_by = f"{group_by}{title_suffix}"

    # val_abs_rot_z (full quaternion geodesic distance) is intentionally
    # dropped from these overview panels: this is a planar in-hand pose task
    # where roll is essentially the only rotation axis in play, so geodesic
    # tracks roll almost exactly (e.g. 0.628 vs 0.626 deg mean on one run) and
    # just duplicates the roll panel. Still logged per-sample in the test CSV
    # (results_*.csv) and in the leaderboards if it's ever needed as a check.
    abs_metrics = [
        ("val_abs_y", "Y-Translation Abs Error [m]"),
        ("val_abs_z", "Z-Translation Abs Error [m]"),
        ("val_abs_rot_x", "Rotation Abs Error (roll) [deg]"),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Pooled Test Error Distributions by {group_by} (log scale)", fontsize=16)
    for ax, (col, label) in zip(axs, abs_metrics):
        pooled = _pooled_by_group(records, col)
        _violin_panel(ax, pooled, groups, colors, label, log=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    for fmt in [f for f in formats if f in ("png", "svg")]:
        fig.savefig(output_dir / f"overview_abs_error_distributions.{fmt}", dpi=150)
    plt.close(fig)

    # Grouped boxplot comparison, one subplot per metric
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Test Error Comparison by {group_by} (boxplots, outliers hidden)", fontsize=16)
    for ax, (col, label) in zip(axs, abs_metrics):
        pooled = _pooled_by_group(records, col)
        _boxplot_panel(ax, pooled, groups, colors, label)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    for fmt in [f for f in formats if f in ("png", "svg")]:
        fig.savefig(output_dir / f"overview_boxplot_by_group.{fmt}", dpi=150)
    plt.close(fig)

    # Signed error distributions (bias direction)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Pooled Signed Test Error Distributions by {group_by}", fontsize=16)
    for ax, col in zip(axs, SIGNED_METRIC_COLS):
        pooled = _pooled_by_group(records, col)
        _violin_panel(ax, pooled, groups, colors, col, log=False)
        ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    for fmt in [f for f in formats if f in ("png", "svg")]:
        fig.savefig(output_dir / f"overview_signed_error_distributions.{fmt}", dpi=150)
    plt.close(fig)

    # Signed error boxplots (same pairing as the abs-error violin/boxplot pair above)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Signed Test Error Comparison by {group_by} (boxplots, outliers hidden)", fontsize=16)
    for ax, col in zip(axs, SIGNED_METRIC_COLS):
        pooled = _pooled_by_group(records, col)
        _boxplot_panel(ax, pooled, groups, colors, col)
        ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    for fmt in [f for f in formats if f in ("png", "svg")]:
        fig.savefig(output_dir / f"overview_boxplot_signed_by_group.{fmt}", dpi=150)
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


def generate_overview(processed, overview_dir, plot_formats, group_by, split_overview=True):
    records = _collect_run_records(processed, group_by)
    if not records:
        print("⚠️ No test results available to build an overview from.")
        return

    per_run_df, per_group_df = build_leaderboards(records)
    write_leaderboards(per_run_df, per_group_df, overview_dir, group_by)
    plot_pooled_distributions(records, overview_dir, plot_formats, group_by)

    # Each data.split_seed has its own test set, so pooling splits mixes different test
    # samples into one box. Also write one overview per split seed (comparable groups only).
    if split_overview:
        for split_seed in sorted({r["split_seed"] for r in records if r["split_seed"] != "?"}, key=str):
            subset = [r for r in records if r["split_seed"] == split_seed]
            sub_dir = overview_dir / f"split_seed-{split_seed}"
            sub_run_df, sub_group_df = build_leaderboards(subset)
            write_leaderboards(sub_run_df, sub_group_df, sub_dir, group_by)
            plot_pooled_distributions(subset, sub_dir, plot_formats, group_by,
                                      title_suffix=f" | split_seed={split_seed}")


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
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help="Don't run inference: build the overview from the results/ already present in each run folder."
    )
    parser.add_argument(
        "--no-split-overview",
        action="store_true",
        help="Only build the pooled overview; skip the per-split-seed overviews (overview/split_seed-<N>/)."
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

    # Group runs by split seed: each seed needs its own Data_split/test rebuild
    # (ensure_test_split_current), so consecutive same-seed runs reuse it.
    # sorted() is stable, so timestamp order is kept within a seed.
    folders_to_process.sort(key=_split_seed_sort_key)

    if not folders_to_process:
        print("⚠️ No folders found to process. Please check your CLI arguments or script config.")
        return

    print(f"📋 Total runs found to process: {len(folders_to_process)}")

    # Loop through and run inference sequentially
    processed = []
    for folder in folders_to_process:
        absolute_folder = os.path.abspath(folder)
        if os.path.exists(absolute_folder):
            if args.overview_only:
                record = _existing_results_record(absolute_folder)
            else:
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
        generate_overview(processed, overview_dir, plot_formats, args.group_by,
                          split_overview=not args.no_split_overview)


if __name__ == "__main__":
    main()
