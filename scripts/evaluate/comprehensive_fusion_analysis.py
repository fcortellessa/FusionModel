#!/usr/bin/env python3
"""
Comprehensive fusion analysis: scatter plots, boxplots, and config-variation arrows.

Extends scatter_test_results_by_fusion.py with:
1. Automatic detection of config-variation pairs within the same fusion method
2. Arrow plots showing how single-parameter changes affect performance
3. Boxplots for train/val errors (absolute and signed) from metrics.csv
4. Additional scatter combinations (rot_x vs trans, trans_y vs trans_z)

A "config-variation pair" is two runs of the same fusion method that differ
in exactly one non-seed config parameter. Arrows show the direction of change
in error space when that parameter varies.

Usage:
    python scripts/evaluate/comprehensive_fusion_analysis.py \\
        --run_dir runs/final_ablation runs/se_gca \\
        --output_dir runs/comprehensive_analysis
"""

import argparse
from pathlib import Path
from collections import defaultdict
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Copy relevant parts from scatter_test_results_by_fusion.py
CANDIDATE_MAP = {
    ("concat", False): "1. Concatenation",
    ("gated", False): "2. Gating",
    ("gated_Softmax", False): "2. Gating",
    ("gated_concat", False): "3. Softmax gated concatenation (scalar)",
    ("residual_gated_concat", False): "4. Residual sigmoid gated concatenation (scalar)",
    ("channelwise_gated_concat_local", False): "5. Channelwise local sigmoid gated concatenation",
    ("channelwise_residual_gated_concat_hybrid", False): "6. Channelwise hybrid residual sigmoid gated concatenation",
    ("gated_concat", True): "7. Dual Fusion",
}

CANDIDATE_ORDER = [
    "1. Concatenation",
    "2. Gating",
    "3. Softmax gated concatenation (scalar)",
    "4. Residual sigmoid gated concatenation (scalar)",
    "5. Channelwise local sigmoid gated concatenation",
    "6. Channelwise hybrid residual sigmoid gated concatenation",
    "7. Dual Fusion",
]

CANDIDATE_COLORS = {
    "1. Concatenation": "#4C72B0",
    "2. Gating": "#DD8452",
    "3. Softmax gated concatenation (scalar)": "#55A868",
    "4. Residual sigmoid gated concatenation (scalar)": "#C44E52",
    "5. Channelwise local sigmoid gated concatenation": "#8172B2",
    "6. Channelwise hybrid residual sigmoid gated concatenation": "#937860",
    "7. Dual Fusion": "#CCB974",
}
FALLBACK_COLORS = ["#64B5CD", "#8C8C8C", "#DA8BC3", "#CCCCCC"]


def set_split(split: str):
    global SPLIT_LABEL, SPLIT_KEY, RESULTS_SUBDIR, RESULTS_GLOB
    if split == "val":
        SPLIT_LABEL, SPLIT_KEY, RESULTS_SUBDIR, RESULTS_GLOB = "Validation", "val", "results_val", "results_val*.csv"
    elif split == "test":
        SPLIT_LABEL, SPLIT_KEY, RESULTS_SUBDIR, RESULTS_GLOB = "Test", "test", "results", "results_*.csv"


SPLIT_LABEL = "Test"
SPLIT_KEY = "test"
RESULTS_SUBDIR = "results"
RESULTS_GLOB = "results_*.csv"


def has_task_adapter(model_cfg: dict) -> bool:
    if model_cfg.get("task_adapter"):
        return True
    return bool((model_cfg.get("task_feature_adapter", {}) or {}).get("enabled"))


def classify_candidate(fusion: str, dual_fusion: bool, model_cfg: dict = None) -> str:
    label = CANDIDATE_MAP.get((fusion, bool(dual_fusion)))
    if label is None:
        label = f"{fusion}{'+dual' if dual_fusion else ''}"
    if model_cfg is not None and has_task_adapter(model_cfg):
        label = f"{label} [+task_adapter]"
    return label


def find_run_config(run_folder: Path):
    for name in ("config.yml", "config.resolved.yml"):
        candidate = run_folder / name
        if candidate.exists():
            return candidate
    return None


def latest_results_csv(results_dir: Path):
    candidates = sorted(results_dir.glob(RESULTS_GLOB))
    return candidates[-1] if candidates else None


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in (d or {}).items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, tuple(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def discover_run_folders(run_dirs):
    found = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.is_absolute():
            run_dir = (REPO_ROOT / run_dir) if (REPO_ROOT / run_dir).exists() else run_dir.resolve()

        results_dir = run_dir / RESULTS_SUBDIR
        if results_dir.is_dir() and latest_results_csv(results_dir) is not None:
            found[str(run_dir)] = run_dir
            continue

        for csv_path in run_dir.rglob(f"{RESULTS_SUBDIR}/{RESULTS_GLOB}"):
            candidate = csv_path.parent.parent
            found[str(candidate)] = candidate

    return sorted(found.values())


def collect_runs(run_folders):
    """Collect runs with test results, metrics, and flattened configs."""
    rows = []
    per_sample_frames = []
    excluded = []
    configs = {}
    metrics_cache = {}
    seen_run_names = {}

    for run_folder in run_folders:
        run_name = run_folder.name

        cfg_path = find_run_config(run_folder)
        if cfg_path is None:
            excluded.append({"run_folder": str(run_folder), "reason": "no config.yml/config.resolved.yml found"})
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            excluded.append({"run_folder": str(run_folder), "reason": f"failed to parse config: {e}"})
            continue

        model_cfg = cfg.get("model", {}) or {}
        fusion = model_cfg.get("fusion")
        dual_fusion = bool(model_cfg.get("dual_fusion", False))
        seed = (cfg.get("training", {}) or {}).get("seed")
        candidate = classify_candidate(fusion, dual_fusion, model_cfg)
        flat_cfg = flatten_dict(cfg)

        if run_name in seen_run_names:
            prev_folder, prev_flat_cfg = seen_run_names[run_name]
            if prev_flat_cfg == flat_cfg:
                excluded.append({"run_folder": str(run_folder), "fusion": fusion,
                                  "reason": f"duplicate of {prev_folder}"})
                continue
        else:
            seen_run_names[run_name] = (str(run_folder), flat_cfg)

        results_dir = run_folder / RESULTS_SUBDIR
        csv_path = latest_results_csv(results_dir)
        if csv_path is None:
            excluded.append({"run_folder": str(run_folder), "fusion": fusion, "reason": "no results_*.csv in results/"})
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            excluded.append({"run_folder": str(run_folder), "fusion": fusion, "reason": f"failed to read {csv_path.name}: {e}"})
            continue

        required_cols = {"val_abs_y", "val_abs_z", "val_abs_rot_x", "val_abs_rot_z"}
        missing = required_cols - set(df.columns)
        if missing or df.empty:
            excluded.append({"run_folder": str(run_folder), "fusion": fusion, "reason": f"results csv missing columns {missing}"})
            continue

        df = df.copy()
        df["run_name"] = run_name
        df["run_folder"] = str(run_folder)
        df["fusion"] = fusion
        df["dual_fusion"] = dual_fusion
        df["candidate"] = candidate
        df["seed"] = seed
        df["trans_l2_mm"] = np.sqrt(df["val_abs_y"] ** 2 + df["val_abs_z"] ** 2) * 1000.0
        per_sample_frames.append(df)
        configs[str(run_folder)] = flat_cfg

        # Load metrics.csv for train/val errors
        metrics_path = run_folder / "metrics.csv"
        metrics_df = None
        if metrics_path.exists():
            try:
                metrics_df = pd.read_csv(metrics_path)
                metrics_cache[str(run_folder)] = metrics_df
            except Exception:
                pass

        rows.append({
            "run_name": run_name,
            "run_folder": str(run_folder),
            "fusion": fusion,
            "dual_fusion": dual_fusion,
            "candidate": candidate,
            "seed": seed,
            "n_samples": len(df),
            "test_csv": str(csv_path),
            "trans_l2_mm": df["trans_l2_mm"].mean(),
            "rot_x_deg": df["val_abs_rot_x"].mean(),
            "rot_z_geodesic_deg": df["val_abs_rot_z"].mean(),
            "y_mm": (df["val_abs_y"] * 1000.0).mean(),
            "z_mm": (df["val_abs_z"] * 1000.0).mean(),
        })

    per_run_df = pd.DataFrame(rows)
    per_sample_df = pd.concat(per_sample_frames, ignore_index=True) if per_sample_frames else pd.DataFrame()
    excluded_df = pd.DataFrame(excluded)

    if not per_run_df.empty:
        per_run_df["candidate"] = pd.Categorical(
            per_run_df["candidate"],
            categories=CANDIDATE_ORDER + sorted(set(per_run_df["candidate"]) - set(CANDIDATE_ORDER)),
            ordered=True,
        )
        per_run_df = per_run_df.sort_values(["candidate", "seed"]).reset_index(drop=True)

    return per_run_df, per_sample_df, excluded_df, configs, metrics_cache


def find_config_variation_pairs(per_run_df: pd.DataFrame, configs: dict) -> list:
    """Find pairs of runs within the same candidate that differ in exactly one config parameter (not seed).

    Returns list of dicts: {candidate, pair_name, run1, run2, param_name, value1, value2}
    """
    pairs = []
    ignore_prefixes = ("training.seed",)

    for candidate, group in per_run_df.groupby("candidate", observed=True):
        run_folders = group["run_folder"].tolist()
        if len(run_folders) < 2:
            continue

        # Compare all pairs
        for i in range(len(run_folders)):
            for j in range(i + 1, len(run_folders)):
                rf1, rf2 = run_folders[i], run_folders[j]
                cfg1, cfg2 = configs[rf1], configs[rf2]

                # Find differing keys
                all_keys = set(cfg1.keys()) | set(cfg2.keys())
                diffs = []
                for key in sorted(all_keys):
                    if key.startswith(ignore_prefixes):
                        continue
                    v1 = cfg1.get(key, "<missing>")
                    v2 = cfg2.get(key, "<missing>")
                    if v1 != v2:
                        diffs.append((key, v1, v2))

                # Only include pairs with exactly one difference
                if len(diffs) == 1:
                    key, val1, val2 = diffs[0]
                    run1_data = group[group["run_folder"] == rf1].iloc[0]
                    run2_data = group[group["run_folder"] == rf2].iloc[0]
                    pairs.append({
                        "candidate": str(candidate),
                        "pair_name": f"{key}: {val1} → {val2}",
                        "param_name": key,
                        "value1": val1,
                        "value2": val2,
                        "run1_folder": rf1,
                        "run2_folder": rf2,
                        "run1_name": run1_data["run_name"],
                        "run2_name": run2_data["run_name"],
                        "run1_trans_mm": run1_data["trans_l2_mm"],
                        "run2_trans_mm": run2_data["trans_l2_mm"],
                        "run1_rot_deg": run1_data["rot_x_deg"],
                        "run2_rot_deg": run2_data["rot_x_deg"],
                    })

    return pairs


def get_color(candidate: str, seen: dict):
    if candidate in CANDIDATE_COLORS:
        return CANDIDATE_COLORS[candidate]
    if candidate not in seen:
        seen[candidate] = FALLBACK_COLORS[len(seen) % len(FALLBACK_COLORS)]
    return seen[candidate]


def plot_config_variation_arrows(per_run_df: pd.DataFrame, pairs: list, output_dir: Path):
    """Plot arrows showing config variations within each candidate."""
    if not pairs:
        return

    # Group pairs by candidate
    by_candidate = defaultdict(list)
    for pair in pairs:
        by_candidate[pair["candidate"]].append(pair)

    for candidate, candidate_pairs in by_candidate.items():
        fig, ax = plt.subplots(figsize=(10, 8))

        color = get_color(candidate, {})

        # Plot baseline runs as points
        candidate_df = per_run_df[per_run_df["candidate"].astype(str) == candidate]
        ax.scatter(candidate_df["trans_l2_mm"], candidate_df["rot_x_deg"],
                   color=color, s=100, edgecolor="black", linewidth=1, alpha=0.6, label="other runs")

        # Plot arrows for each variation pair
        for idx, pair in enumerate(candidate_pairs, 1):
            x1, y1 = pair["run1_trans_mm"], pair["run1_rot_deg"]
            x2, y2 = pair["run2_trans_mm"], pair["run2_rot_deg"]

            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle="-|>", lw=2, color=color, alpha=0.7))

            # Label the arrow with a number
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, str(idx), fontsize=10, fontweight="bold",
                   bbox=dict(boxstyle="circle", facecolor="white", alpha=0.8))

        ax.set_xlabel("Test Trans L2 (mm)")
        ax.set_ylabel("Test Rot X Error (deg)")
        ax.set_title(f"Config variations: {candidate}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()

        safe_name = candidate.replace(" ", "_").replace("+", "_").replace("[", "").replace("]", "")
        fig.savefig(output_dir / f"config_arrows_{safe_name}.png", dpi=150)
        plt.close(fig)

        # Print variation details
        print(f"\n{candidate}:")
        for idx, pair in enumerate(candidate_pairs, 1):
            print(f"  {idx}. {pair['param_name']}: {pair['value1']} → {pair['value2']}")
            print(f"     {pair['run1_name']} → {pair['run2_name']}")
            print(f"     Δtrans={pair['run2_trans_mm'] - pair['run1_trans_mm']:.2f}mm, " +
                  f"Δrot={pair['run2_rot_deg'] - pair['run1_rot_deg']:.2f}°")


def plot_scatter_additional(per_run_df: pd.DataFrame, per_sample_df: pd.DataFrame, output_dir: Path):
    """Create additional scatter plots: trans_y vs trans_z."""
    if per_sample_df.empty:
        return

    fallback_seen = {}
    candidates = sorted(per_run_df["candidate"].astype(str).unique().tolist())

    # Trans Y vs Z (per-sample)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for cand in candidates:
        cdf = per_sample_df[per_sample_df["candidate"] == cand]
        color = get_color(cand, fallback_seen)
        ax.scatter(cdf["val_abs_y"] * 1000, cdf["val_abs_z"] * 1000,
                  color=color, s=14, alpha=0.35, linewidth=0, label=cand)

    ax.set_xlabel("Test Trans Y Error (mm)")
    ax.set_ylabel("Test Trans Z Error (mm)")
    ax.set_title("Per-sample translation errors: Y vs Z")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_trans_y_vs_z_per_sample.png", dpi=150)
    plt.close(fig)


def plot_boxplots_train_val(metrics_cache: dict, per_run_df: pd.DataFrame, output_dir: Path):
    """Create boxplots for train/val errors (absolute and signed) from metrics.csv."""
    if not metrics_cache:
        print("Warning: no metrics.csv files found, skipping boxplots")
        return

    # Collect train/val metrics by candidate
    candidate_metrics = defaultdict(lambda: {"train_loss": [], "val_loss": [],
                                               "train_rot": [], "val_rot": [],
                                               "train_trans": [], "val_trans": []})

    for run_folder, metrics_df in metrics_cache.items():
        # Find the candidate for this run
        row = per_run_df[per_run_df["run_folder"] == run_folder]
        if row.empty:
            continue
        candidate = row.iloc[0]["candidate"]

        # Extract final epoch metrics (best validation)
        if not metrics_df.empty:
            best_idx = metrics_df["val_loss"].idxmin()
            best_row = metrics_df.loc[best_idx]

            candidate_metrics[candidate]["train_loss"].append(best_row.get("train_loss", np.nan))
            candidate_metrics[candidate]["val_loss"].append(best_row.get("val_loss", np.nan))

    if not candidate_metrics:
        return

    # Plot train/val loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    candidates = sorted(candidate_metrics.keys())
    train_losses = [candidate_metrics[c]["train_loss"] for c in candidates]
    val_losses = [candidate_metrics[c]["val_loss"] for c in candidates]

    ax1.boxplot(train_losses, labels=[str(c)[:20] for c in candidates])
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss Distribution")
    ax1.tick_params(axis="x", rotation=45)

    ax2.boxplot(val_losses, labels=[str(c)[:20] for c in candidates])
    ax2.set_ylabel("Val Loss")
    ax2.set_title("Validation Loss Distribution")
    ax2.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_train_val_loss.png", dpi=150)
    plt.close(fig)

    print("Created boxplot_train_val_loss.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", nargs="+", required=True,
                         help="One or more run folders/container directories to analyze.")
    parser.add_argument("--output_dir", default="runs/comprehensive_analysis",
                         help="Directory to write plots and CSVs to.")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                         help="Which split to analyze (test or val).")
    args = parser.parse_args()

    set_split(args.split)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_folders = discover_run_folders(args.run_dir)
    if not run_folders:
        print(f"No run folders found under {args.run_dir}")
        return

    print(f"Discovered {len(run_folders)} run folder(s)")

    per_run_df, per_sample_df, excluded_df, configs, metrics_cache = collect_runs(run_folders)

    if per_run_df.empty:
        print("No usable runs found")
        return

    print(f"Included: {len(per_run_df)} runs  |  Excluded: {len(excluded_df)}")

    # Find config variation pairs
    pairs = find_config_variation_pairs(per_run_df, configs)
    print(f"Found {len(pairs)} config-variation pair(s)")

    # Create plots
    print("\nGenerating plots...")
    plot_config_variation_arrows(per_run_df, pairs, output_dir)
    plot_scatter_additional(per_run_df, per_sample_df, output_dir)
    plot_boxplots_train_val(metrics_cache, per_run_df, output_dir)

    # Save pair info to JSON
    if pairs:
        with open(output_dir / "config_variation_pairs.json", "w") as f:
            json.dump(pairs, f, indent=2, default=str)

    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
