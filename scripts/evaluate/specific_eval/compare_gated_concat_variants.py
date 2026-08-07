import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

FUSION_VARIANTS = [
    "gated_concat",
    "channelwise_gated_concat",
    "residual_gated_concat",
    "channelwise_residual_gated_concat",
]

VARIANT_COLORS = {
    "gated_concat": "#4C72B0",
    "channelwise_gated_concat": "#DD8452",
    "residual_gated_concat": "#55A868",
    "channelwise_residual_gated_concat": "#C44E52",
}


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


def find_run_config(run_folder: Path):
    for name in ("config.yml", "config.resolved.yml"):
        candidate = run_folder / name
        if candidate.exists():
            return candidate
    return None



def collect_runs(runs_dir: Path):
    """Scans runs_dir for metrics.csv + config, keeping only the target fusion variants."""
    rows = []
    excluded = []
    curves = {}
    configs = {}

    for metrics_path in sorted(runs_dir.rglob("metrics.csv")):
        run_folder = metrics_path.parent
        run_name = run_folder.name
        cfg_path = find_run_config(run_folder)
        if cfg_path is None:
            continue

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            excluded.append({"run_folder": str(run_folder), "reason": f"failed to parse config: {e}"})
            continue

        fusion = (cfg.get("model", {}) or {}).get("fusion")
        if fusion not in FUSION_VARIANTS:
            continue

        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            excluded.append({"run_folder": str(run_folder), "fusion": fusion, "reason": f"failed to read metrics.csv: {e}"})
            continue

        if "epoch" not in df.columns or df.empty:
            excluded.append({"run_folder": str(run_folder), "fusion": fusion, "reason": "metrics.csv missing epoch column or empty"})
            continue

        max_epoch = int(df["epoch"].max())
        seed = (cfg.get("training", {}) or {}).get("seed")

        curves[str(run_folder)] = df
        configs[str(run_folder)] = flatten_dict(cfg)

        rows.append({
            "run_name": run_name,
            "run_folder": str(run_folder),
            "fusion": fusion,
            "seed": seed,
            "max_epoch": max_epoch,
        })

    return rows, excluded, curves, configs


def score_run(df: pd.DataFrame, tail: int):
    final = df.sort_values("epoch").tail(tail)

    val_rot = final["val_abs_rot_x"].mean() if "val_abs_rot_x" in final.columns else np.nan
    val_y = final["val_abs_y"].mean() if "val_abs_y" in final.columns else np.nan
    val_z = final["val_abs_z"].mean() if "val_abs_z" in final.columns else np.nan
    val_loss = final["val_loss"].mean() if "val_loss" in final.columns else np.nan

    trans_l2_mm = np.sqrt((val_y * 1000) ** 2 + (val_z * 1000) ** 2) if pd.notna(val_y) and pd.notna(val_z) else np.nan
    combined = val_rot + trans_l2_mm if pd.notna(val_rot) and pd.notna(trans_l2_mm) else np.nan

    return {
        "val_loss": val_loss,
        "val_rot_x_deg": val_rot,
        "val_trans_l2_mm": trans_l2_mm,
        "combined_score": combined,
    }


def build_dataframe(runs_dir: Path, required_epochs: int, tail_epochs: int):
    rows, excluded, curves, configs = collect_runs(runs_dir)

    included_rows = []
    for rec in rows:
        max_epoch = rec["max_epoch"]
        if max_epoch < required_epochs:
            excluded.append({
                "run_folder": rec["run_folder"],
                "fusion": rec["fusion"],
                "reason": f"only reached epoch {max_epoch} (< required {required_epochs})",
            })
            continue

        df = curves[rec["run_folder"]]
        scores = score_run(df, tail_epochs)
        included_rows.append({**rec, **scores})

    included_df = pd.DataFrame(included_rows)
    excluded_df = pd.DataFrame(excluded)

    if not included_df.empty:
        included_df = included_df.sort_values(["fusion", "seed"]).reset_index(drop=True)

    return included_df, excluded_df, curves, configs


def compute_config_drift_notes(df: pd.DataFrame, configs: dict) -> list:
    """For each fusion variant, flags config keys that differ across its included runs."""
    notes = []
    if df.empty:
        return notes

    # Keys whose drift across seeds of the same variant is expected/irrelevant.
    ignore_keys_prefixes = ("training.seed",)

    for fusion, group in df.groupby("fusion"):
        run_folders = group["run_folder"].tolist()
        if len(run_folders) < 2:
            continue

        flat_configs = [configs[rf] for rf in run_folders]
        all_keys = set()
        for fc in flat_configs:
            all_keys.update(fc.keys())

        diffs = []
        for key in sorted(all_keys):
            if key.startswith(ignore_keys_prefixes):
                continue
            values = [fc.get(key, "<missing>") for fc in flat_configs]
            if len(set(values)) > 1:
                diffs.append((key, values))

        if diffs:
            notes.append({"fusion": fusion, "run_folders": run_folders, "diffs": diffs})

    return notes


def write_summary(df: pd.DataFrame, excluded_df: pd.DataFrame, notes: list, output_dir: Path, required_epochs: int):
    lines = []
    lines.append("# Gated-concat fusion variant comparison\n")

    lines.append("## Leaderboard (included runs, ranked by combined score, ascending = better)\n")
    if df.empty:
        lines.append("No eligible runs found.\n")
    else:
        lines.append("| combined_score | fusion | seed | val_rot_x_deg | val_trans_l2_mm | run_folder |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in df.sort_values("combined_score").iterrows():
            lines.append(
                f"| {row['combined_score']:.4f} | {row['fusion']} | {row['seed']} | "
                f"{row['val_rot_x_deg']:.4f} | {row['val_trans_l2_mm']:.4f} | {row['run_folder']} |"
            )
        lines.append("")

        lines.append("## Per-variant summary (mean +/- std across seeds)\n")
        lines.append("| fusion | n_seeds | seeds | mean_combined_score | std_combined_score | mean_val_rot_x_deg | mean_val_trans_l2_mm |")
        lines.append("|---|---|---|---|---|---|---|")
        for fusion in FUSION_VARIANTS:
            fdf = df[df["fusion"] == fusion]
            if fdf.empty:
                lines.append(f"| {fusion} | 0 | - | - | - | - | - |")
                continue
            seeds = sorted(fdf["seed"].dropna().astype(int).unique().tolist())
            lines.append(
                f"| {fusion} | {len(seeds)} | {seeds} | {fdf['combined_score'].mean():.4f} | "
                f"{fdf['combined_score'].std():.4f} | {fdf['val_rot_x_deg'].mean():.4f} | "
                f"{fdf['val_trans_l2_mm'].mean():.4f} |"
            )
        lines.append("")

    lines.append("## Excluded runs (did not reach the required epoch count, or failed to parse)\n")
    if excluded_df.empty:
        lines.append("None.\n")
    else:
        lines.append(f"Required epochs: {required_epochs}\n")
        lines.append("| fusion | reason | run_folder |")
        lines.append("|---|---|---|")
        for _, row in excluded_df.iterrows():
            lines.append(f"| {row.get('fusion', '-')} | {row['reason']} | {row['run_folder']} |")
        lines.append("")

    lines.append("## Notes: config drift within a fusion variant\n")
    lines.append(
        "The runs below share a `model.fusion` value but differ in other config keys besides "
        "`training.seed`. This means seeds are not an apples-to-apples comparison for that variant "
        "— check whether the drift is intentional before trusting the aggregate.\n"
    )
    if not notes:
        lines.append("None detected — every variant's included runs are identical apart from seed.\n")
    else:
        for note in notes:
            lines.append(f"### {note['fusion']}\n")
            lines.append(f"Runs compared: {', '.join(Path(rf).name for rf in note['run_folders'])}\n")
            lines.append("| config key | values (one per run, same order as above) |")
            lines.append("|---|---|")
            for key, values in note["diffs"]:
                lines.append(f"| {key} | {values} |")
            lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_bar_combined_score(df: pd.DataFrame, output_dir: Path):
    variants = [v for v in FUSION_VARIANTS if v in df["fusion"].unique()]
    if not variants:
        return

    means = [df[df["fusion"] == v]["combined_score"].mean() for v in variants]
    stds = [df[df["fusion"] == v]["combined_score"].std() if (df["fusion"] == v).sum() > 1 else 0.0 for v in variants]
    colors = [VARIANT_COLORS[v] for v in variants]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(variants))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylabel("Combined score: Val Rot X (deg) + Val Trans L2 (mm)")
    ax.set_title("Gated-concat fusion variants: combined score\n(error bars = std across seeds)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "bar_combined_score.png", dpi=150)
    plt.close(fig)

    fig = go.Figure()
    fig.add_bar(
        x=variants, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=colors,
    )
    fig.update_layout(
        title="Gated-concat fusion variants: combined score (lower is better)",
        yaxis_title="Combined score: Val Rot X (deg) + Val Trans L2 (mm)",
        template="plotly_white",
    )
    fig.write_html(output_dir / "bar_combined_score.html")


def plot_val_loss_curves(df: pd.DataFrame, curves: dict, output_dir: Path):
    variants = [v for v in FUSION_VARIANTS if v in df["fusion"].unique()]
    if not variants:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    fig_plotly = go.Figure()
    seen_labels = set()

    for _, row in df.iterrows():
        run_df = curves[row["run_folder"]]
        color = VARIANT_COLORS[row["fusion"]]
        label = row["fusion"] if row["fusion"] not in seen_labels else None
        seen_labels.add(row["fusion"])

        ax.plot(run_df["epoch"], run_df["val_loss"], color=color, alpha=0.7, label=label)
        fig_plotly.add_trace(go.Scatter(
            x=run_df["epoch"], y=run_df["val_loss"],
            mode="lines", line=dict(color=color),
            name=row["fusion"],
            legendgroup=row["fusion"],
            showlegend=row["fusion"] not in [t.name for t in fig_plotly.data[:-1]],
            hovertemplate=f"{row['run_name']} (seed={row['seed']})<br>epoch=%{{x}}<br>val_loss=%{{y:.5f}}<extra></extra>",
        ))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val loss")
    ax.set_title("Validation loss curves by fusion variant")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="fusion")
    fig.tight_layout()
    fig.savefig(output_dir / "val_loss_curves.png", dpi=150)
    plt.close(fig)

    fig_plotly.update_layout(
        title="Validation loss curves by fusion variant",
        xaxis_title="Epoch", yaxis_title="Val loss",
        template="plotly_white",
    )
    fig_plotly.write_html(output_dir / "val_loss_curves.html")


def plot_scatter_rot_vs_trans(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        return
    variants = [v for v in FUSION_VARIANTS if v in df["fusion"].unique()]

    fig, ax = plt.subplots(figsize=(7, 6))
    for v in variants:
        vdf = df[df["fusion"] == v]
        ax.scatter(
            vdf["val_trans_l2_mm"], vdf["val_rot_x_deg"],
            color=VARIANT_COLORS[v], s=90, edgecolor="black", linewidth=0.5,
            alpha=0.85, label=v,
        )
    ax.set_xlabel("Val Trans L2 Norm (mm)")
    ax.set_ylabel("Val Rot X Error (deg)")
    ax.set_title("Error trade-off by fusion variant\n(last-5-epoch mean)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="fusion")
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_rot_vs_trans.png", dpi=150)
    plt.close(fig)

    fig_plotly = go.Figure()
    for v in variants:
        vdf = df[df["fusion"] == v]
        fig_plotly.add_trace(go.Scatter(
            x=vdf["val_trans_l2_mm"], y=vdf["val_rot_x_deg"],
            mode="markers", name=v,
            marker=dict(color=VARIANT_COLORS[v], size=12, line=dict(width=1, color="black")),
            text=[f"{rn} (seed={s})" for rn, s in zip(vdf["run_name"], vdf["seed"])],
            hovertemplate="%{text}<br>trans_l2=%{x:.4f} mm<br>rot_x=%{y:.4f} deg<extra></extra>",
        ))
    fig_plotly.update_layout(
        title="Error trade-off by fusion variant (last-5-epoch mean)",
        xaxis_title="Val Trans L2 Norm (mm)", yaxis_title="Val Rot X Error (deg)",
        template="plotly_white",
    )
    fig_plotly.write_html(output_dir / "scatter_rot_vs_trans.html")


def plot_metric_breakdown(df: pd.DataFrame, output_dir: Path):
    variants = [v for v in FUSION_VARIANTS if v in df["fusion"].unique()]
    if not variants:
        return

    rot_means = [df[df["fusion"] == v]["val_rot_x_deg"].mean() for v in variants]
    trans_means = [df[df["fusion"] == v]["val_trans_l2_mm"].mean() for v in variants]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(variants))
    colors = [VARIANT_COLORS[v] for v in variants]

    axes[0].bar(x, rot_means, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variants, rotation=20, ha="right")
    axes[0].set_ylabel("Val Rot X Error (deg)")
    axes[0].set_title("Rotation error")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    axes[1].bar(x, trans_means, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variants, rotation=20, ha="right")
    axes[1].set_ylabel("Val Trans L2 Norm (mm)")
    axes[1].set_title("Translation error")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Rotation vs translation error by fusion variant (mean across seeds)")
    fig.tight_layout()
    fig.savefig(output_dir / "metric_breakdown.png", dpi=150)
    plt.close(fig)

    from plotly.subplots import make_subplots
    fig_plotly = make_subplots(rows=1, cols=2, subplot_titles=("Rotation error", "Translation error"))
    fig_plotly.add_bar(x=variants, y=rot_means, marker_color=colors, row=1, col=1, showlegend=False)
    fig_plotly.add_bar(x=variants, y=trans_means, marker_color=colors, row=1, col=2, showlegend=False)
    fig_plotly.update_yaxes(title_text="Val Rot X Error (deg)", row=1, col=1)
    fig_plotly.update_yaxes(title_text="Val Trans L2 Norm (mm)", row=1, col=2)
    fig_plotly.update_layout(
        title="Rotation vs translation error by fusion variant (mean across seeds)",
        template="plotly_white",
    )
    fig_plotly.write_html(output_dir / "metric_breakdown.html")


def main():
    parser = argparse.ArgumentParser(description="Compare gated_concat fusion variants across seeds")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory containing run folders (searched recursively)")
    parser.add_argument("--required_epochs", type=int, default=150, help="Runs logging fewer than this many epochs are excluded")
    parser.add_argument("--tail_epochs", type=int, default=5, help="Number of final epochs to average for scoring")
    parser.add_argument("--output_dir", type=str, default="runs/gated_concat_variant_analysis")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = (REPO_ROOT / runs_dir) if (REPO_ROOT / runs_dir).exists() else runs_dir.resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df, excluded_df, curves, configs = build_dataframe(runs_dir, args.required_epochs, args.tail_epochs)

    if df.empty and excluded_df.empty:
        print(
            f"No runs matching fusion in {FUSION_VARIANTS} found under {runs_dir}. "
            "This script identifies runs by reading model.fusion from each run's config.yml "
            "(not from the folder name)."
        )
        return

    df.to_csv(output_dir / "comparison_table.csv", index=False)
    excluded_df.to_csv(output_dir / "excluded_runs.csv", index=False)
    print(f"Included runs: {len(df)}  |  Excluded runs: {len(excluded_df)}")
    print(f"Wrote comparison_table.csv and excluded_runs.csv to {output_dir}")

    notes = compute_config_drift_notes(df, configs)

    write_summary(df, excluded_df, notes, output_dir, args.required_epochs)
    print(f"Wrote summary.md to {output_dir}")

    if not df.empty:
        plot_bar_combined_score(df, output_dir)
        plot_val_loss_curves(df, curves, output_dir)
        plot_scatter_rot_vs_trans(df, output_dir)
        plot_metric_breakdown(df, output_dir)
        print(f"\nPlots (png + html) written to {output_dir}:")
        for f in sorted(output_dir.glob("*.png")):
            print(f"  - {f.name}")
    else:
        print("\nNo included runs after epoch-count filtering; no plots generated.")


if __name__ == "__main__":
    main()

