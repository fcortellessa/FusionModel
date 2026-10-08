#!/usr/bin/env python3
"""
Predicted vs ground-truth roll, and test error vs roll and object position, per fusion candidate.

Motivation: a few test samples dominate the rot_x error (the top 5% of ~325 samples carry 30-56%
of it) and the same samples are hard for every model. This looks at *where* in pose space the
errors sit and whether predictions are compressed toward the mean roll.

All ground-truth rolls in this dataset lie in a narrow band (~165-180 deg; a few are stored as
-179.9 and are unwrapped to ~180.1), so the pred-vs-gt panels are dominated by the identity line;
the signed-error-vs-roll panels are the informative ones (a slope there == compression/expansion
of the predicted roll range).

Reads the per-sample test results (results/results_*.csv written by scripts/test.py) of the runs
under --run_dir and groups by the same 7 fusion candidates as final_ablation_candidate_analysis.py
(which it reuses). Run it once per rot_scaling_factor folder:

    python scripts/evaluate/roll_position_error_analysis.py --run_dir runs/final_ablation
    python scripts/evaluate/roll_position_error_analysis.py --run_dir runs/final_ablation_rot_scaling_200

Default output: <run_dir>/overview/sample_level_analysis/

Outputs (png + svg unless noted):
    roll_pred_vs_gt              predicted vs ground-truth roll per candidate (identity line, median-trend slope)
    roll_signed_error_vs_gt      signed roll error vs ground-truth roll (binned median + IQR, median-trend slope)
    roll_abs_error_by_gt_bin     median / mean / p90 |roll error| per equal-count gt-roll bin, per candidate
    error_vs_position            per-sample error (mean over runs) vs ground-truth y / z position
    hard_samples_pose_map        where the hardest test samples sit in (roll, y) and (roll, z)
    error_position_correlations.csv   Spearman of |errors| vs gt roll / y / z, per candidate
    hardest_samples.csv          top-N hardest samples per split (mean |roll error| over that split's runs)

Definitions: signed roll error = val_signed_rot_x (pred - gt, wrap-safe as computed by test.py);
predicted roll is plotted as gt + signed error so a wrap at +-180 could never split the cloud.
Positions are the ground-truth translation from the results csv, in mm. Points in the
per-candidate panels pool every run of that candidate (same test samples repeat across runs), so
they are not independent draws; per-sample plots average over runs first.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_ablation_candidate_analysis as fca  # noqa: E402  (also applies the candidate-6 override)

scat = fca.scat
REPO_ROOT = fca.REPO_ROOT


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_samples(run_dirs):
    folders = scat.discover_run_folders(run_dirs)
    if not folders:
        sys.exit(f"No run folders with results/results_*.csv found under {run_dirs}")
    per_run_df, ps, excluded_df, configs = scat.collect_runs(folders)
    if ps.empty:
        sys.exit("No runs with usable per-sample test results.")
    if not excluded_df.empty:
        print(f"Excluded {len(excluded_df)} run(s):\n{excluded_df.to_string(index=False)}")

    needed = {"gt_roll_deg", "val_signed_rot_x", "val_abs_rot_x", "gt_trans_y", "gt_trans_z",
              "val_abs_y", "val_abs_z", "sample_id"}
    missing = needed - set(ps.columns)
    if missing:
        sys.exit(f"results csv is missing columns {missing}")

    ps = ps.copy()
    ps["candidate"] = ps["candidate"].astype(str)
    ps["split"] = ps["run_folder"].map(lambda rf: configs[rf].get("data.split_seed"))
    # Roll is stored in (-180, 180], but this dataset sits around +-180 (~165-180 deg): a few samples are
    # stored as -179.9 (same orientation as +180.1) and would stretch the axis over 360 deg and fake a
    # slope of 1. Unwrap around the circular mean so the range is contiguous.
    th = np.radians(ps["gt_roll_deg"])
    ref = np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean()))
    ps["gt_roll"] = ref + ((ps["gt_roll_deg"] - ref + 180.0) % 360.0 - 180.0)
    ps["err_roll"] = ps["val_signed_rot_x"]
    ps["pred_roll"] = ps["gt_roll"] + ps["err_roll"]
    ps["abs_roll"] = ps["val_abs_rot_x"]
    ps["gt_y_mm"] = ps["gt_trans_y"] * 1000.0
    ps["gt_z_mm"] = ps["gt_trans_z"] * 1000.0
    ps["abs_y_mm"] = ps["val_abs_y"] * 1000.0
    ps["abs_z_mm"] = ps["val_abs_z"] * 1000.0
    return ps


def per_sample_means(ps):
    """One row per test sample (split, sample_id): ground truth + mean |error| over all runs of that split."""
    keys = ["split", "sample_id"]
    return (ps.groupby(keys)
            .agg(gt_roll=("gt_roll", "first"), gt_y_mm=("gt_y_mm", "first"), gt_z_mm=("gt_z_mm", "first"),
                 abs_roll=("abs_roll", "mean"), abs_roll_max=("abs_roll", "max"),
                 signed_roll=("err_roll", "mean"), abs_y_mm=("abs_y_mm", "mean"), abs_z_mm=("abs_z_mm", "mean"),
                 n_runs=("run_name", "nunique"))
            .reset_index())


def binned(df, xcol, ycol, edges, stats=("median", "mean", "p90")):
    """Per equal-count bin: x centre (median x) and the requested stats of y."""
    idx = np.clip(np.digitize(df[xcol].to_numpy(), edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        y = df[ycol].to_numpy()[m]
        r = {"x": float(np.median(df[xcol].to_numpy()[m])), "n": int(m.sum())}
        if "median" in stats:
            r["median"] = float(np.median(y))
        if "mean" in stats:
            r["mean"] = float(np.mean(y))
        if "p90" in stats:
            r["p90"] = float(np.quantile(y, 0.9))
        if "q25" in stats:
            r["q25"] = float(np.quantile(y, 0.25))
            r["q75"] = float(np.quantile(y, 0.75))
        rows.append(r)
    return pd.DataFrame(rows)


def equal_count_edges(values, n_bins):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def median_trend_slope(df, xcol, ycol, edges):
    """Slope of the line through the per-bin medians. Unlike a least-squares fit over all points it is not
    dragged around by the handful of samples that are off by 5-10 deg in every model."""
    b = binned(df, xcol, ycol, edges, stats=("median",))
    return float(np.polyfit(b["x"], b["median"], 1)[0]) if len(b) >= 2 else float("nan")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _grid(n_panels, ncols=4, panel=(4.6, 4.2)):
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False)
    return fig, list(axes.flat)


def plot_pred_vs_gt(ps, out_dir, cands, names, colors, edges):
    fig, axes = _grid(len(cands) + 1)
    lo, hi = ps["gt_roll"].min(), ps["gt_roll"].max()
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    slopes = {}
    for ax, cand in zip(axes, cands + ["all"]):
        d = ps if cand == "all" else ps[ps["candidate"] == cand]
        color = "#444444" if cand == "all" else colors[cand]
        ax.scatter(d["gt_roll"], d["pred_roll"], s=5, alpha=0.15, color=color, linewidth=0)
        ax.plot(lim, lim, "k--", lw=1)
        slope = median_trend_slope(d, "gt_roll", "pred_roll", edges)
        slopes[cand] = slope
        ax.text(0.04, 0.95, f"median-trend slope {slope:.3f}\nmedian |err| {d['abs_roll'].median():.2f}°",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0))
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title("all candidates" if cand == "all" else names[cand], fontsize=9, fontweight="bold")
        ax.set_xlabel("ground-truth roll [deg]")
        ax.set_ylabel("predicted roll [deg]")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(cands) + 1:]:
        ax.axis("off")
    fig.suptitle("Predicted vs ground-truth roll (all runs of a candidate pooled; median-trend slope <1 = compression toward the mean roll)",
                 fontsize=11)
    fca.save_fig(fig, out_dir, "roll_pred_vs_gt")
    return slopes


def plot_signed_error_vs_gt(ps, out_dir, cands, names, colors, edges):
    fig, axes = _grid(len(cands) + 1)
    ylim = float(np.quantile(ps["err_roll"].abs(), 0.99)) * 1.4
    for ax, cand in zip(axes, cands + ["all"]):
        d = ps if cand == "all" else ps[ps["candidate"] == cand]
        color = "#444444" if cand == "all" else colors[cand]
        ax.scatter(d["gt_roll"], d["err_roll"], s=5, alpha=0.15, color=color, linewidth=0)
        b = binned(d, "gt_roll", "err_roll", edges, stats=("median", "q25"))
        ax.plot(b["x"], b["median"], "k-", lw=2, marker="o", ms=4)
        ax.fill_between(b["x"], b["q25"], b["q75"], color="black", alpha=0.12)
        ax.axhline(0, color="black", ls="--", lw=1, alpha=0.6)
        slope = median_trend_slope(d, "gt_roll", "err_roll", edges)
        clipped = int((d["err_roll"].abs() > ylim).sum())
        ax.text(0.04, 0.95, f"median bias trend {slope * 10:+.2f}° per +10° roll\n{clipped} pts beyond ±{ylim:.1f}° (clipped)",
                transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0))
        ax.set_ylim(-ylim, ylim)
        ax.set_title("all candidates" if cand == "all" else names[cand], fontsize=9, fontweight="bold")
        ax.set_xlabel("ground-truth roll [deg]")
        ax.set_ylabel("signed roll error  pred - gt [deg]")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(cands) + 1:]:
        ax.axis("off")
    fig.suptitle("Signed roll error vs ground-truth roll (black = binned median, band = IQR; downward trend = compression)",
                 fontsize=11)
    fca.save_fig(fig, out_dir, "roll_signed_error_vs_gt")


def plot_abs_error_by_gt_bin(ps, out_dir, cands, names, colors, edges):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, stat, title in zip(axes, ("median", "mean", "p90"), ("median", "mean", "90th percentile")):
        for cand in cands:
            b = binned(ps[ps["candidate"] == cand], "gt_roll", "abs_roll", edges)
            ax.plot(b["x"], b[stat], color=colors[cand], lw=2, marker="o", ms=4, label=names[cand])
        ax.set_title(f"{title} |roll error| per gt-roll bin", fontsize=11, fontweight="bold")
        ax.set_xlabel("ground-truth roll [deg] (equal-count bins)")
        ax.set_ylabel("|roll error| [deg]")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    fca.save_fig(fig, out_dir, "roll_abs_error_by_gt_bin")


def plot_error_vs_position(samples, out_dir, n_bins):
    panels = [
        ("gt_y_mm", "abs_roll", "|roll error| [deg]", "ground-truth y [mm]", True),
        ("gt_z_mm", "abs_roll", "|roll error| [deg]", "ground-truth z [mm]", True),
        ("gt_y_mm", "abs_y_mm", "|y error| [mm]", "ground-truth y [mm]", False),
        ("gt_z_mm", "abs_z_mm", "|z error| [mm]", "ground-truth z [mm]", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for ax, (xcol, ycol, ylabel, xlabel, logy) in zip(axes.flat, panels):
        ax.scatter(samples[xcol], samples[ycol], s=9, alpha=0.3, color="#4C72B0", linewidth=0)
        b = binned(samples, xcol, ycol, equal_count_edges(samples[xcol], n_bins))
        ax.plot(b["x"], b["median"], "k-", lw=2, marker="o", ms=4, label="median")
        ax.plot(b["x"], b["p90"], color="#C44E52", lw=2, marker="s", ms=4, label="p90")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both" if logy else "major")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Per-sample error (mean over the runs of a split) vs ground-truth position; equal-count bins", fontsize=12)
    fca.save_fig(fig, out_dir, "error_vs_position")


def plot_hard_sample_map(samples, out_dir, top_n):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    norm = LogNorm(vmin=max(samples["abs_roll"].min(), 1e-3), vmax=samples["abs_roll"].max())
    hardest = samples.nlargest(top_n, "abs_roll")
    for ax, ycol, ylabel in zip(axes, ("gt_y_mm", "gt_z_mm"), ("ground-truth y [mm]", "ground-truth z [mm]")):
        sc = ax.scatter(samples["gt_roll"], samples[ycol], c=samples["abs_roll"], cmap="viridis", norm=norm,
                        s=16, alpha=0.8, linewidth=0)
        ax.scatter(hardest["gt_roll"], hardest[ycol], s=70, facecolor="none", edgecolor="red", lw=1.3,
                   label=f"{top_n} hardest samples")
        ax.set_xlabel("ground-truth roll [deg]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.colorbar(sc, ax=axes, label="mean |roll error| over runs [deg] (log)")
    fig.suptitle("Where the hard test samples sit in pose space (all splits pooled, one dot per test sample)", fontsize=12)
    fig.savefig(out_dir / "hard_samples_pose_map.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "hard_samples_pose_map.svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_correlations(ps, out_dir, cands):
    pairs = [("abs_roll", "gt_roll"), ("abs_roll", "gt_y_mm"), ("abs_roll", "gt_z_mm"),
             ("abs_y_mm", "gt_y_mm"), ("abs_z_mm", "gt_z_mm")]
    rows = []
    for cand in cands + ["all"]:
        d = ps if cand == "all" else ps[ps["candidate"] == cand]
        row = {"candidate": cand, "n_rows": len(d)}
        for e, g in pairs:
            row[f"spearman({e},{g})"] = d[e].corr(d[g], method="spearman")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "error_position_correlations.csv", index=False)
    return df


def write_hardest(samples, out_dir, top_n):
    rows = []
    for split, g in samples.groupby("split"):
        top = g.nlargest(top_n, "abs_roll").copy()
        top["rank_in_split"] = range(1, len(top) + 1)
        rows.append(top)
    df = pd.concat(rows, ignore_index=True)[[
        "split", "rank_in_split", "sample_id", "abs_roll", "abs_roll_max", "signed_roll",
        "gt_roll", "gt_y_mm", "gt_z_mm", "abs_y_mm", "abs_z_mm", "n_runs"]]
    df.to_csv(out_dir / "hardest_samples.csv", index=False)
    return df


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", nargs="+", required=True,
                        help="Run folder(s) / container dir(s); use one rot_scaling_factor folder at a time.")
    parser.add_argument("--output_dir", default=None,
                        help="Default: <run_dir>/overview/sample_level_analysis (only when a single --run_dir is given).")
    parser.add_argument("--n_bins", type=int, default=6, help="Equal-count bins for the binned statistics (default 6).")
    parser.add_argument("--top_n", type=int, default=15, help="Hardest samples listed per split (default 15).")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif len(args.run_dir) == 1:
        out_dir = Path(args.run_dir[0]) / "overview" / "sample_level_analysis"
    else:
        sys.exit("--output_dir is required when more than one --run_dir is given")
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = load_samples(args.run_dir)
    cands = scat.ordered_candidates(ps["candidate"].unique().tolist())
    names = fca.candidate_display_names(ps, cands)
    fallback = {}
    colors = {c: scat.get_color(c, fallback) for c in cands}
    samples = per_sample_means(ps)
    edges = equal_count_edges(samples["gt_roll"], args.n_bins)
    print(f"{ps['run_name'].nunique()} runs, {len(samples)} distinct test samples "
          f"({samples.groupby('split').size().to_dict()} per split)")

    slopes = plot_pred_vs_gt(ps, out_dir, cands, names, colors, edges)
    plot_signed_error_vs_gt(ps, out_dir, cands, names, colors, edges)
    plot_abs_error_by_gt_bin(ps, out_dir, cands, names, colors, edges)
    plot_error_vs_position(samples, out_dir, args.n_bins)
    plot_hard_sample_map(samples, out_dir, args.top_n)
    corr = write_correlations(ps, out_dir, cands)
    hard = write_hardest(samples, out_dir, args.top_n)

    print("\nMedian-trend slope of predicted vs ground-truth roll (1.0 = no compression; least-squares would be outlier-driven):")
    for c, s in slopes.items():
        print(f"  {names.get(c, c):60s} {s:.3f}")
    print("\nSpearman correlations, all candidates pooled:")
    print(corr[corr["candidate"] == "all"].drop(columns=["candidate", "n_rows"]).round(2).to_string(index=False))
    top = samples.nlargest(args.top_n, "abs_roll")
    print(f"\nHardest {args.top_n} samples overall: gt roll {top['gt_roll'].min():.1f}-{top['gt_roll'].max():.1f} deg "
          f"(all samples: {samples['gt_roll'].min():.1f}-{samples['gt_roll'].max():.1f}), "
          f"y {top['gt_y_mm'].min():.1f}..{top['gt_y_mm'].max():.1f} mm, z {top['gt_z_mm'].min():.1f}..{top['gt_z_mm'].max():.1f} mm")

    print(f"\nWrote outputs to {out_dir}:")
    for f in sorted(out_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
