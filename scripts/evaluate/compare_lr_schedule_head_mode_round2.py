#!/usr/bin/env python3
"""
Compares the lr_schedule_head_mode_round2 grid (configs/generated/lr_schedule_head_mode_round2/)
against the matching subset of round-1 runs (configs/generated/lr_schedule_ablation/), and reports
which runs are eligible for apples-to-apples comparison, then plots joint vs. separate
translation_head_mode across schedule arms and learning rates.

Matching: manifest.json's "config_path" field stores the absolute path to whatever config file
was passed to train.py --config (train_multiple_runs.py passes the generated variant file
directly), so runs are identified by the stem of that path against the known variant names below.

Eligibility rule: a run only enters a head_mode x schedule_arm comparison cell if it shares the
same fusion, dual_fusion, and lr as the other runs in that cell. Round 1's separate-mode
warmup_cosine/global_plateau runs used lr=0.0001 (round 2 uses lr=0.0005 throughout), so they are
reported in a separate LR-sensitivity section instead of being pooled into the head-mode comparison.

Usage:
    python scripts/evaluate/compare_lr_schedule_head_mode_round2.py --runs_dir runs
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

ROUND1_NAMES = {
    "warmup_cosine_seed43", "warmup_cosine_seed44", "warmup_cosine_seed45",
    "global_plateau_seed43", "global_plateau_seed44", "global_plateau_seed45",
    "per_head_plateau_seed43", "per_head_plateau_seed44", "per_head_plateau_seed45",
}
ROUND2_NAMES = {
    "pure_fixed_joint_seed43", "pure_fixed_separate_seed43",
    "pure_fixed_joint_seed44", "pure_fixed_separate_seed44",
    "per_head_plateau_joint_seed44", "per_head_plateau_joint_seed45",
    "warmup_cosine_joint_seed43", "warmup_cosine_joint_seed44",
    "global_plateau_joint_seed43", "global_plateau_joint_seed44",
    "warmup_cosine_separate_seed43", "global_plateau_separate_seed43",
}
ALL_NAMES = ROUND1_NAMES | ROUND2_NAMES

ARM_ORDER = ["pure_fixed", "warmup_cosine", "global_plateau", "per_head_plateau"]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_schedule_arm(cfg):
    training = cfg.get("training", {})
    sched = training.get("lr_schedule", {}) or {}
    head_decay = training.get("head_lr_decay", {}) or {}

    sched_on = bool(sched.get("enabled", False))
    decay_on = bool(head_decay.get("enabled", False))

    if sched_on and str(sched.get("type", "")).lower() == "warmup_cosine":
        return "warmup_cosine"
    if sched_on and str(sched.get("type", "")).lower() == "reduce_on_plateau":
        return "global_plateau"
    if not sched_on and decay_on:
        return "per_head_plateau"
    if not sched_on and not decay_on:
        return "pure_fixed"
    return "unknown"


def find_run_config(run_folder: Path):
    for name in ("config.yml", "config.resolved.yml"):
        candidate = run_folder / name
        if candidate.exists():
            return candidate
    return None


def collect_runs(runs_dir: Path):
    records = []
    for run_folder in sorted(runs_dir.rglob("metrics.csv")):
        run_folder = run_folder.parent
        manifest_path = run_folder / "manifest.json"
        cfg_path = find_run_config(run_folder)
        if not manifest_path.exists() or cfg_path is None:
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        source_stem = Path(str(manifest.get("config_path", ""))).stem

        if source_stem not in ALL_NAMES:
            continue

        cfg = load_config(cfg_path)
        model_cfg = cfg.get("model", {})
        training_cfg = cfg.get("training", {})

        records.append({
            "source_round": "round1" if source_stem in ROUND1_NAMES else "round2",
            "variant_name": source_stem,
            "run_folder": str(run_folder),
            "head_mode": model_cfg.get("translation_head_mode", "separate"),
            "fusion": model_cfg.get("fusion"),
            "dual_fusion": bool(model_cfg.get("dual_fusion", False)),
            "seed": training_cfg.get("seed"),
            "lr": training_cfg.get("lr"),
            "schedule_arm": classify_schedule_arm(cfg),
            "metrics_path": str(run_folder / "metrics.csv"),
        })
    return records


def score_run(metrics_path: str, tail: int = 5):
    df = pd.read_csv(metrics_path)
    final = df.sort_values("epoch").tail(tail)

    val_rot_x = final["val_abs_rot_x"].mean() if "val_abs_rot_x" in final.columns else np.nan
    val_y = final["val_abs_y"].mean() if "val_abs_y" in final.columns else np.nan
    val_z = final["val_abs_z"].mean() if "val_abs_z" in final.columns else np.nan
    val_loss = final["val_loss"].mean() if "val_loss" in final.columns else np.nan
    epoch_time = final["epoch_time_sec"].mean() if "epoch_time_sec" in final.columns else np.nan

    trans_l2_mm = np.sqrt((val_y * 1000) ** 2 + (val_z * 1000) ** 2) if pd.notna(val_y) and pd.notna(val_z) else np.nan
    combined = val_rot_x + trans_l2_mm if pd.notna(val_rot_x) and pd.notna(trans_l2_mm) else np.nan

    return {
        "val_loss": val_loss,
        "val_rot_x_deg": val_rot_x,
        "val_trans_l2_mm": trans_l2_mm,
        "combined_score": combined,
        "epoch_time_sec": epoch_time,
        "epochs_logged": int(df["epoch"].max()) if "epoch" in df.columns else np.nan,
    }, df


def build_dataframe(runs_dir: Path, tail_epochs: int):
    records = collect_runs(runs_dir)
    rows = []
    curves = {}
    for rec in records:
        try:
            scores, full_df = score_run(rec["metrics_path"], tail=tail_epochs)
        except Exception as e:
            print(f"Warning: failed to score {rec['run_folder']}: {e}")
            continue
        rows.append({**rec, **scores})
        curves[rec["run_folder"]] = full_df

    df = pd.DataFrame(rows)
    if df.empty:
        return df, curves

    # Eligibility: within each schedule_arm, only rows at the modal (most common) lr for that
    # arm are "eligible" for the head-mode comparison; everything else goes in the LR-sensitivity
    # section instead.
    df["eligible"] = False
    for arm, group in df.groupby("schedule_arm"):
        modal_lr = group["lr"].mode().iloc[0]
        df.loc[group[group["lr"] == modal_lr].index, "eligible"] = True

    return df.sort_values(["schedule_arm", "head_mode", "seed"]).reset_index(drop=True), curves


def print_report(df: pd.DataFrame):
    eligible = df[df["eligible"]].copy()
    excluded = df[~df["eligible"]].copy()

    print("=" * 88)
    print("JOINT vs SEPARATE, BY SCHEDULE ARM (eligible runs only, same lr per arm)")
    print("=" * 88)
    for arm in ARM_ORDER:
        arm_df = eligible[eligible["schedule_arm"] == arm]
        if arm_df.empty:
            print(f"\n[{arm}] -- no eligible runs found yet")
            continue
        print(f"\n[{arm}]  (lr={arm_df['lr'].mode().iloc[0]})")
        for head_mode, hgroup in arm_df.groupby("head_mode"):
            seeds = sorted(hgroup["seed"].dropna().astype(int).unique().tolist())
            print(
                f"  head_mode={head_mode:9s} n_seeds={len(seeds)} seeds={seeds} "
                f"mean_combined_score={hgroup['combined_score'].mean():.4f}  "
                f"(mean_val_rot_x_deg={hgroup['val_rot_x_deg'].mean():.4f}, "
                f"mean_val_trans_l2_mm={hgroup['val_trans_l2_mm'].mean():.4f})"
            )

    print("\n" + "=" * 88)
    print("FULL LEADERBOARD (eligible runs, ranked by combined score, ascending = better)")
    print("=" * 88)
    for _, row in eligible.sort_values("combined_score").iterrows():
        print(
            f"  {row['combined_score']:.4f}  |  {row['schedule_arm']:16s} "
            f"head_mode={row['head_mode']:9s} seed={row['seed']}  lr={row['lr']}  "
            f"[{row['variant_name']}]  -> {row['run_folder']}"
        )

    if not excluded.empty:
        print("\n" + "=" * 88)
        print("LR SENSITIVITY (excluded from head-mode comparison: different lr than the rest of their arm)")
        print("=" * 88)
        for _, row in excluded.sort_values(["schedule_arm", "lr"]).iterrows():
            print(
                f"  {row['schedule_arm']:16s} head_mode={row['head_mode']:9s} seed={row['seed']} "
                f"lr={row['lr']}  combined_score={row['combined_score']:.4f}  "
                f"[{row['variant_name']}]  -> {row['run_folder']}"
            )


def plot_bar_comparison(df: pd.DataFrame, output_dir: Path):
    eligible = df[df["eligible"]]
    arms = [a for a in ARM_ORDER if a in eligible["schedule_arm"].unique()]
    if not arms:
        return

    head_modes = ["separate", "joint"]
    colors = {"separate": "#4C72B0", "joint": "#DD8452"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(arms))
    width = 0.35

    for i, hm in enumerate(head_modes):
        means, stds = [], []
        for arm in arms:
            cell = eligible[(eligible["schedule_arm"] == arm) & (eligible["head_mode"] == hm)]
            means.append(cell["combined_score"].mean() if not cell.empty else np.nan)
            stds.append(cell["combined_score"].std() if len(cell) > 1 else 0.0)
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=hm, color=colors[hm])

    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=15)
    ax.set_ylabel("Combined score: Val Rot X (deg) + Val Trans L2 (mm)")
    ax.set_title("Joint vs Separate translation_head_mode, by LR schedule arm\n(lr=0.0005, error bars = std across seeds)")
    ax.legend(title="translation_head_mode")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "bar_joint_vs_separate_by_arm.png", dpi=150)
    plt.close(fig)


def plot_val_loss_curves(df: pd.DataFrame, curves: dict, output_dir: Path):
    eligible = df[df["eligible"]]
    arms = [a for a in ARM_ORDER if a in eligible["schedule_arm"].unique()]
    if not arms:
        return

    fig, axes = plt.subplots(1, len(arms), figsize=(5.5 * len(arms), 4.5), sharey=True)
    if len(arms) == 1:
        axes = [axes]
    colors = {"separate": "#4C72B0", "joint": "#DD8452"}

    for ax, arm in zip(axes, arms):
        arm_df = eligible[eligible["schedule_arm"] == arm]
        for hm in ["separate", "joint"]:
            cell = arm_df[arm_df["head_mode"] == hm]
            for _, row in cell.iterrows():
                run_df = curves[row["run_folder"]]
                ax.plot(run_df["epoch"], run_df["val_loss"], color=colors[hm], alpha=0.7,
                         label=hm if row.equals(cell.iloc[0]) else None)
        ax.set_title(arm)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Val loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="translation_head_mode", loc="upper right")
    fig.suptitle("Validation loss curves: joint vs separate, per schedule arm (lr=0.0005)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "val_loss_curves_by_arm.png", dpi=150)
    plt.close(fig)


def plot_error_scatter(df: pd.DataFrame, output_dir: Path):
    eligible = df[df["eligible"]]
    if eligible.empty:
        return

    markers = {"pure_fixed": "o", "warmup_cosine": "s", "global_plateau": "^", "per_head_plateau": "D"}
    colors = {"separate": "#4C72B0", "joint": "#DD8452"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in eligible.iterrows():
        ax.scatter(
            row["val_trans_l2_mm"], row["val_rot_x_deg"],
            color=colors.get(row["head_mode"], "gray"),
            marker=markers.get(row["schedule_arm"], "x"),
            s=90, edgecolor="black", linewidth=0.5, alpha=0.85,
        )

    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=hm)
                     for hm, c in colors.items()]
    marker_handles = [Line2D([0], [0], marker=m, color="black", linestyle="", markersize=9, label=arm)
                       for arm, m in markers.items() if arm in eligible["schedule_arm"].unique()]
    legend1 = ax.legend(handles=color_handles, title="head_mode", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, title="schedule_arm", loc="upper right")

    ax.set_xlabel("Val Trans L2 Norm (mm)")
    ax.set_ylabel("Val Rot X Error (deg)")
    ax.set_title("Error trade-off: joint vs separate translation_head_mode\n(last-5-epoch mean, lr=0.0005)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_rot_vs_trans_error.png", dpi=150)
    plt.close(fig)


def plot_lr_sensitivity(df: pd.DataFrame, output_dir: Path):
    excluded = df[~df["eligible"]]
    if excluded.empty:
        return

    arms = [a for a in ARM_ORDER if a in excluded["schedule_arm"].unique()]
    if not arms:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(arms))
    lrs_seen = sorted(df[df["schedule_arm"].isin(arms) & (df["head_mode"] == "separate")]["lr"].unique())
    width = 0.8 / max(len(lrs_seen), 1)
    palette = plt.cm.viridis(np.linspace(0.2, 0.8, len(lrs_seen)))

    for i, lr in enumerate(lrs_seen):
        means = []
        for arm in arms:
            cell = df[(df["schedule_arm"] == arm) & (df["head_mode"] == "separate") & (df["lr"] == lr)]
            means.append(cell["combined_score"].mean() if not cell.empty else np.nan)
        offset = (i - (len(lrs_seen) - 1) / 2) * width
        ax.bar(x + offset, means, width, label=f"lr={lr}", color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=15)
    ax.set_ylabel("Combined score: Val Rot X (deg) + Val Trans L2 (mm)")
    ax.set_title("LR sensitivity, separate-mode runs\n(seed 43 unless noted; lower is better)")
    ax.legend(title="learning rate")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "lr_sensitivity_separate_mode.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare lr_schedule_head_mode_round2 runs")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory containing run folders (searched recursively)")
    parser.add_argument("--tail_epochs", type=int, default=5, help="Number of final epochs to average for scoring")
    parser.add_argument("--output_dir", type=str, default="runs/lr_schedule_head_mode_round2_analysis")
    args = parser.parse_args()

    runs_dir = (REPO_ROOT / args.runs_dir) if not os.path.isabs(args.runs_dir) else Path(args.runs_dir)
    output_dir = (REPO_ROOT / args.output_dir) if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, curves = build_dataframe(runs_dir, args.tail_epochs)
    if df.empty:
        print(
            f"No matching runs found under {runs_dir}. This script identifies runs via "
            "manifest.json's 'config_path' field matching a known variant filename from "
            "configs/generated/lr_schedule_ablation/ or configs/generated/lr_schedule_head_mode_round2/."
        )
        return

    df.to_csv(output_dir / "comparison_table.csv", index=False)
    print(f"Wrote full comparison table to {output_dir / 'comparison_table.csv'}\n")

    print_report(df)

    plot_bar_comparison(df, output_dir)
    plot_val_loss_curves(df, curves, output_dir)
    plot_error_scatter(df, output_dir)
    plot_lr_sensitivity(df, output_dir)

    print(f"\nPlots written to {output_dir}:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
