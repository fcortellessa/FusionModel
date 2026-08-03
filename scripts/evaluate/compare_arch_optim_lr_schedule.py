#!/usr/bin/env python3
"""
Compares training runs along four independent axes:

  1. Architecture:        model.translation_head_mode        joint | separate
  2. Optimization:         training.head_lr_decay.enabled      on | off
  3. Global vs task-specific LR schedule (both ReduceLROnPlateau-style):
                            training.lr_schedule (global, applied to every
                            param group together) vs training.head_lr_decay
                            (task-specific, one ReduceLROnPlateau per head)
                                                                none | global | task_specific | both
  4. LR schedule type:     training.lr_schedule.{enabled,type} none | reduce_on_plateau | warmup_cosine

Runs are discovered by scanning `--runs_dir` recursively for `metrics.csv`, reading the
run's own `config.yml`/`config.resolved.yml` snapshot for classification (not the folder
name), and are scored using the best (lowest mean val_loss) rolling window of
`--score_window` consecutive epochs in `metrics.csv` -- not just the final epochs, so a
run isn't penalized for a noisy uptick right at the end of training.

Cleaning step: any run whose logged epochs don't reach `--required_epochs` (default 100)
is dropped from every comparison and listed separately in excluded_runs.csv, since a run
that didn't finish training isn't comparable to one that did.

This script is meant to be run against the run directory on the remote machine where
training actually happens (pass --runs_dir accordingly); it does not search for or assume
any local runs.

Outputs (under --output_dir):
    comparison_table.csv                    - one row per included run, with scores + classification
    excluded_runs.csv                       - incomplete/unparseable runs, with reason
    summary.md                              - leaderboard + per-axis breakdown tables
    bar_head_mode.png/.html                 - axis 1: joint vs separate (bars labeled with n per group)
    bar_head_lr_decay.png/.html             - axis 2: head_lr_decay on vs off (n per group)
    bar_schedule_scope.png/.html            - axis 3: none vs global vs task_specific vs both (n per group)
    bar_schedule_type.png/.html             - axis 4: none vs reduce_on_plateau vs warmup_cosine (n per group)
    scatter_head_mode.png/.html             - axis 1: val rot error (deg) vs val trans L2 (mm), per run
    scatter_head_lr_decay.png/.html         - axis 2: same scatter, colored by head_lr_decay
    scatter_schedule_scope.png/.html        - axis 3: same scatter, colored by schedule scope
    scatter_schedule_type.png/.html         - axis 4: same scatter, colored by schedule type
    val_loss_curves_by_schedule_type.png/.html  - val_loss vs epoch, colored by schedule type

Usage:
    python scripts/evaluate/compare_arch_optim_lr_schedule.py --runs_dir /path/to/remote/runs
    python scripts/evaluate/compare_arch_optim_lr_schedule.py --runs_dir /path/to/remote/runs --required_epochs 100
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

HEAD_MODE_ORDER = ["separate", "joint"]
HEAD_MODE_COLORS = {"separate": "#4C72B0", "joint": "#DD8452"}

HEAD_LR_DECAY_ORDER = ["off", "on"]
HEAD_LR_DECAY_COLORS = {"off": "#4C72B0", "on": "#DD8452"}

SCHEDULE_SCOPE_ORDER = ["none", "global", "task_specific", "both"]
SCHEDULE_SCOPE_COLORS = {
    "none": "#8C8C8C",
    "global": "#4C72B0",
    "task_specific": "#DD8452",
    "both": "#55A868",
}

SCHEDULE_TYPE_ORDER = ["none", "reduce_on_plateau", "warmup_cosine"]
SCHEDULE_TYPE_COLORS = {
    "none": "#8C8C8C",
    "reduce_on_plateau": "#4C72B0",
    "warmup_cosine": "#DD8452",
}


def find_run_config(run_folder: Path):
    for name in ("config.yml", "config.resolved.yml"):
        candidate = run_folder / name
        if candidate.exists():
            return candidate
    return None


def classify_run(cfg: dict) -> dict:
    model_cfg = cfg.get("model", {}) or {}
    training_cfg = cfg.get("training", {}) or {}

    sched = training_cfg.get("lr_schedule", {}) or {}
    head_decay = training_cfg.get("head_lr_decay", {}) or {}

    sched_on = bool(sched.get("enabled", False))
    sched_type = str(sched.get("type", "none")).lower() if sched_on else "none"
    if sched_type not in ("reduce_on_plateau", "warmup_cosine"):
        sched_type = "none"

    decay_on = bool(head_decay.get("enabled", False))

    if sched_on and decay_on:
        scope = "both"
    elif sched_on:
        scope = "global"
    elif decay_on:
        scope = "task_specific"
    else:
        scope = "none"

    return {
        "head_mode": model_cfg.get("translation_head_mode", "separate"),
        "head_lr_decay": "on" if decay_on else "off",
        "schedule_scope": scope,
        "schedule_type": sched_type,
        "fusion": model_cfg.get("fusion"),
        "dual_fusion": bool(model_cfg.get("dual_fusion", False)),
        "seed": training_cfg.get("seed"),
        "lr": training_cfg.get("lr"),
    }


def collect_runs(runs_dir: Path):
    rows = []
    excluded = []
    curves = {}

    for metrics_path in sorted(runs_dir.rglob("metrics.csv")):
        run_folder = metrics_path.parent
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

        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            excluded.append({"run_folder": str(run_folder), "reason": f"failed to read metrics.csv: {e}"})
            continue

        if "epoch" not in df.columns or df.empty:
            excluded.append({"run_folder": str(run_folder), "reason": "metrics.csv missing epoch column or empty"})
            continue

        max_epoch = int(df["epoch"].max())
        classification = classify_run(cfg)

        curves[str(run_folder)] = df
        rows.append({
            "run_name": run_name,
            "run_folder": str(run_folder),
            "max_epoch": max_epoch,
            **classification,
        })

    return rows, excluded, curves


def score_run(df: pd.DataFrame, window: int):
    """Scores a run using the best (lowest val_loss) rolling window of `window` consecutive
    epochs, rather than just the final `window` epochs -- avoids penalizing a run whose last
    few logged epochs happen to be a noisy local uptick."""
    ordered = df.sort_values("epoch").reset_index(drop=True)
    win = min(window, len(ordered))

    if "val_loss" in ordered.columns:
        rolling_val_loss = ordered["val_loss"].rolling(win, min_periods=win).mean()
        best_end = rolling_val_loss.idxmin()
    else:
        best_end = len(ordered) - 1

    best_window = ordered.iloc[max(0, best_end - win + 1): best_end + 1]

    val_rot = best_window["val_abs_rot_x"].mean() if "val_abs_rot_x" in best_window.columns else np.nan
    val_y = best_window["val_abs_y"].mean() if "val_abs_y" in best_window.columns else np.nan
    val_z = best_window["val_abs_z"].mean() if "val_abs_z" in best_window.columns else np.nan
    val_loss = best_window["val_loss"].mean() if "val_loss" in best_window.columns else np.nan

    trans_l2_mm = np.sqrt((val_y * 1000) ** 2 + (val_z * 1000) ** 2) if pd.notna(val_y) and pd.notna(val_z) else np.nan
    combined = val_rot + trans_l2_mm if pd.notna(val_rot) and pd.notna(trans_l2_mm) else np.nan

    return {
        "val_loss": val_loss,
        "val_rot_x_deg": val_rot,
        "val_trans_l2_mm": trans_l2_mm,
        "combined_score": combined,
        "best_window_start_epoch": int(best_window["epoch"].iloc[0]) if not best_window.empty else np.nan,
        "best_window_end_epoch": int(best_window["epoch"].iloc[-1]) if not best_window.empty else np.nan,
    }


def build_dataframe(runs_dir: Path, required_epochs: int, score_window: int):
    rows, excluded, curves = collect_runs(runs_dir)

    included_rows = []
    for rec in rows:
        max_epoch = rec["max_epoch"]
        if max_epoch < required_epochs:
            excluded.append({
                "run_folder": rec["run_folder"],
                "reason": f"only reached epoch {max_epoch} (< required {required_epochs})",
            })
            continue

        df = curves[rec["run_folder"]]
        scores = score_run(df, score_window)
        included_rows.append({**rec, **scores})

    included_df = pd.DataFrame(included_rows)
    excluded_df = pd.DataFrame(excluded)

    if not included_df.empty:
        included_df = included_df.sort_values(["schedule_type", "schedule_scope", "head_mode", "head_lr_decay", "seed"]).reset_index(drop=True)

    return included_df, excluded_df, curves


def _axis_summary_table(df: pd.DataFrame, axis_col: str, order: list) -> list:
    lines = [f"| {axis_col} | n_runs | n_seeds | mean_combined_score | std_combined_score | mean_val_rot_x_deg | mean_val_trans_l2_mm |",
             "|---|---|---|---|---|---|---|"]
    for val in order:
        sub = df[df[axis_col] == val]
        if sub.empty:
            lines.append(f"| {val} | 0 | - | - | - | - | - |")
            continue
        seeds = sorted(sub["seed"].dropna().astype(int).unique().tolist())
        lines.append(
            f"| {val} | {len(sub)} | {len(seeds)} | {sub['combined_score'].mean():.4f} | "
            f"{sub['combined_score'].std():.4f} | {sub['val_rot_x_deg'].mean():.4f} | "
            f"{sub['val_trans_l2_mm'].mean():.4f} |"
        )
    return lines


def write_summary(df: pd.DataFrame, excluded_df: pd.DataFrame, output_dir: Path, required_epochs: int):
    lines = []
    lines.append("# Architecture / optimization / LR-schedule comparison\n")

    lines.append("## Leaderboard (included runs, ranked by combined score, ascending = better)\n")
    if df.empty:
        lines.append("No eligible runs found.\n")
    else:
        lines.append("| combined_score | head_mode | head_lr_decay | schedule_scope | schedule_type | seed | best_window (epochs) | run_folder |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, row in df.sort_values("combined_score").iterrows():
            lines.append(
                f"| {row['combined_score']:.4f} | {row['head_mode']} | {row['head_lr_decay']} | "
                f"{row['schedule_scope']} | {row['schedule_type']} | {row['seed']} | "
                f"{row['best_window_start_epoch']}-{row['best_window_end_epoch']} | {row['run_folder']} |"
            )
        lines.append("")

        lines.append("## Axis 1: Architecture -- joint vs separate translation heads\n")
        lines.extend(_axis_summary_table(df, "head_mode", HEAD_MODE_ORDER))
        lines.append("")

        lines.append("## Axis 2: Optimization -- head LR decay on vs off\n")
        lines.extend(_axis_summary_table(df, "head_lr_decay", HEAD_LR_DECAY_ORDER))
        lines.append("")

        lines.append("## Axis 3: Global vs task-specific LR schedule\n")
        lines.append(
            "`none` = no scheduling at all; `global` = training.lr_schedule only "
            "(applied to all param groups together); `task_specific` = training.head_lr_decay "
            "only (one ReduceLROnPlateau per head); `both` = both enabled simultaneously.\n"
        )
        lines.extend(_axis_summary_table(df, "schedule_scope", SCHEDULE_SCOPE_ORDER))
        lines.append("")

        lines.append("## Axis 4: LR schedule type -- none vs ReduceLROnPlateau vs Warmup+Cosine\n")
        lines.append("(training.lr_schedule.type, only meaningful when training.lr_schedule.enabled)\n")
        lines.extend(_axis_summary_table(df, "schedule_type", SCHEDULE_TYPE_ORDER))
        lines.append("")

    lines.append("## Excluded runs (did not reach the required epoch count, or failed to parse)\n")
    if excluded_df.empty:
        lines.append("None.\n")
    else:
        lines.append(f"Required epochs: {required_epochs}\n")
        lines.append("| reason | run_folder |")
        lines.append("|---|---|")
        for _, row in excluded_df.iterrows():
            lines.append(f"| {row['reason']} | {row['run_folder']} |")
        lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _bar_axis(df: pd.DataFrame, axis_col: str, order: list, colors: dict, title: str, filename_stem: str, output_dir: Path):
    present = [v for v in order if v in df[axis_col].unique()]
    if not present:
        return

    counts = [(df[axis_col] == v).sum() for v in present]
    means = [df[df[axis_col] == v]["combined_score"].mean() for v in present]
    stds = [df[df[axis_col] == v]["combined_score"].std() if (df[axis_col] == v).sum() > 1 else 0.0 for v in present]
    bar_colors = [colors[v] for v in present]
    tick_labels = [f"{v}\n(n={n})" for v, n in zip(present, counts)]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(present))
    ax.bar(x, means, yerr=stds, capsize=4, color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.set_ylabel("Combined score: Val Rot X (deg) + Val Trans L2 (mm)")
    ax.set_title(f"{title}\n(lower is better; error bars = std across seeds/other-axis values)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_stem}.png", dpi=150)
    plt.close(fig)

    fig_plotly = go.Figure()
    fig_plotly.add_bar(
        x=tick_labels, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=bar_colors,
        text=[f"n={n}" for n in counts],
        textposition="outside",
    )
    fig_plotly.update_layout(
        title=title,
        xaxis_title=axis_col,
        yaxis_title="Combined score: Val Rot X (deg) + Val Trans L2 (mm)",
        template="plotly_white",
    )
    fig_plotly.write_html(output_dir / f"{filename_stem}.html")


def _scatter_axis(df: pd.DataFrame, axis_col: str, order: list, colors: dict, title: str, filename_stem: str, output_dir: Path):
    present = [v for v in order if v in df[axis_col].unique()]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for v in present:
        vdf = df[df[axis_col] == v]
        ax.scatter(
            vdf["val_trans_l2_mm"], vdf["val_rot_x_deg"],
            color=colors[v], s=90, edgecolor="black", linewidth=0.5,
            alpha=0.85, label=f"{v} (n={len(vdf)})",
        )
        for _, row in vdf.iterrows():
            ax.annotate(
                f"seed={row['seed']}\n{row['schedule_type']}",
                (row["val_trans_l2_mm"], row["val_rot_x_deg"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8, color=colors[v],
            )
    ax.set_xlabel("Val Trans L2 Norm (mm)")
    ax.set_ylabel("Val Rot X Error (deg)")
    ax.set_title(f"{title}\n(best-window mean per run; labels = seed, LR schedule type)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title=axis_col)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_stem}.png", dpi=150)
    plt.close(fig)

    fig_plotly = go.Figure()
    for v in present:
        vdf = df[df[axis_col] == v]
        fig_plotly.add_trace(go.Scatter(
            x=vdf["val_trans_l2_mm"], y=vdf["val_rot_x_deg"],
            mode="markers+text", name=f"{v} (n={len(vdf)})",
            marker=dict(color=colors[v], size=12, line=dict(width=1, color="black")),
            text=[f"seed={s}, {st}" for s, st in zip(vdf["seed"], vdf["schedule_type"])],
            textposition="top center",
            hovertext=[
                f"{rn} (seed={s}, schedule_type={st})"
                for rn, s, st in zip(vdf["run_name"], vdf["seed"], vdf["schedule_type"])
            ],
            hovertemplate="%{hovertext}<br>trans_l2=%{x:.4f} mm<br>rot_x=%{y:.4f} deg<extra></extra>",
        ))
    fig_plotly.update_layout(
        title=f"{title} (best-window mean per run)",
        xaxis_title="Val Trans L2 Norm (mm)", yaxis_title="Val Rot X Error (deg)",
        template="plotly_white",
    )
    fig_plotly.write_html(output_dir / f"{filename_stem}.html")


def plot_val_loss_curves_by_schedule_type(df: pd.DataFrame, curves: dict, output_dir: Path):
    present = [v for v in SCHEDULE_TYPE_ORDER if v in df["schedule_type"].unique()]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    fig_plotly = go.Figure()
    seen_labels = set()
    plotly_names_added = set()

    for _, row in df.iterrows():
        run_df = curves[row["run_folder"]]
        color = SCHEDULE_TYPE_COLORS[row["schedule_type"]]
        label = row["schedule_type"] if row["schedule_type"] not in seen_labels else None
        seen_labels.add(row["schedule_type"])

        ax.plot(run_df["epoch"], run_df["val_loss"], color=color, alpha=0.7, label=label)
        fig_plotly.add_trace(go.Scatter(
            x=run_df["epoch"], y=run_df["val_loss"],
            mode="lines", line=dict(color=color),
            name=row["schedule_type"],
            legendgroup=row["schedule_type"],
            showlegend=row["schedule_type"] not in plotly_names_added,
            hovertemplate=f"{row['run_name']} (seed={row['seed']})<br>epoch=%{{x}}<br>val_loss=%{{y:.5f}}<extra></extra>",
        ))
        plotly_names_added.add(row["schedule_type"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val loss")
    ax.set_title("Validation loss curves by LR schedule type")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="schedule_type")
    fig.tight_layout()
    fig.savefig(output_dir / "val_loss_curves_by_schedule_type.png", dpi=150)
    plt.close(fig)

    fig_plotly.update_layout(
        title="Validation loss curves by LR schedule type",
        xaxis_title="Epoch", yaxis_title="Val loss",
        template="plotly_white",
    )
    fig_plotly.write_html(output_dir / "val_loss_curves_by_schedule_type.html")


def main():
    parser = argparse.ArgumentParser(
        description="Compare runs across architecture (head mode), optimization (head LR decay), "
                     "and LR schedule scope/type"
    )
    parser.add_argument("--runs_dir", type=str, required=True, help="Directory containing run folders (searched recursively)")
    parser.add_argument("--required_epochs", type=int, default=100, help="Runs logging fewer than this many epochs are excluded")
    parser.add_argument("--score_window", type=int, default=5,
                         help="Window size (epochs) for the rolling-mean val_loss score; the run is "
                              "scored using its best (lowest mean val_loss) window of this many "
                              "consecutive epochs, not just the final ones")
    parser.add_argument("--output_dir", type=str, default="runs/arch_optim_lr_schedule_analysis")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = (REPO_ROOT / runs_dir) if (REPO_ROOT / runs_dir).exists() else runs_dir.resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df, excluded_df, curves = build_dataframe(runs_dir, args.required_epochs, args.score_window)

    if df.empty and excluded_df.empty:
        print(f"No runs with metrics.csv + a parseable config found under {runs_dir}.")
        return

    df.to_csv(output_dir / "comparison_table.csv", index=False)
    excluded_df.to_csv(output_dir / "excluded_runs.csv", index=False)
    print(f"Included runs: {len(df)}  |  Excluded runs (incomplete/unparseable): {len(excluded_df)}")
    print(f"Wrote comparison_table.csv and excluded_runs.csv to {output_dir}")

    write_summary(df, excluded_df, output_dir, args.required_epochs)
    print(f"Wrote summary.md to {output_dir}")

    if not df.empty:
        _bar_axis(df, "head_mode", HEAD_MODE_ORDER, HEAD_MODE_COLORS,
                  "Architecture: joint vs separate translation heads", "bar_head_mode", output_dir)
        _bar_axis(df, "head_lr_decay", HEAD_LR_DECAY_ORDER, HEAD_LR_DECAY_COLORS,
                  "Optimization: head LR decay on vs off", "bar_head_lr_decay", output_dir)
        _bar_axis(df, "schedule_scope", SCHEDULE_SCOPE_ORDER, SCHEDULE_SCOPE_COLORS,
                  "Global vs task-specific LR schedule", "bar_schedule_scope", output_dir)
        _bar_axis(df, "schedule_type", SCHEDULE_TYPE_ORDER, SCHEDULE_TYPE_COLORS,
                  "LR schedule type: none vs ReduceLROnPlateau vs Warmup+Cosine", "bar_schedule_type", output_dir)

        _scatter_axis(df, "head_mode", HEAD_MODE_ORDER, HEAD_MODE_COLORS,
                      "Error trade-off: joint vs separate translation heads", "scatter_head_mode", output_dir)
        _scatter_axis(df, "head_lr_decay", HEAD_LR_DECAY_ORDER, HEAD_LR_DECAY_COLORS,
                      "Error trade-off: head LR decay on vs off", "scatter_head_lr_decay", output_dir)
        _scatter_axis(df, "schedule_scope", SCHEDULE_SCOPE_ORDER, SCHEDULE_SCOPE_COLORS,
                      "Error trade-off: global vs task-specific LR schedule", "scatter_schedule_scope", output_dir)
        _scatter_axis(df, "schedule_type", SCHEDULE_TYPE_ORDER, SCHEDULE_TYPE_COLORS,
                      "Error trade-off: LR schedule type", "scatter_schedule_type", output_dir)

        plot_val_loss_curves_by_schedule_type(df, curves, output_dir)

        print(f"\nPlots (png + html) written to {output_dir}:")
        for f in sorted(output_dir.glob("*.png")):
            print(f"  - {f.name}")
    else:
        print("\nNo included runs after epoch-count filtering; no plots generated.")


if __name__ == "__main__":
    main()
