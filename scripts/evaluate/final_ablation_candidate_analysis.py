#!/usr/bin/env python3
"""
Analysis plots for the final ablation, split by the 7 fusion candidates.

Adapts plot functions from three existing scripts so every plot is grouped by
the same 7 candidates (same order, colors and labels everywhere):

    scatter_test_results_by_fusion.py  -> test-set scatter (by run / per sample) + summary table
                                          (reused directly: candidate classification, collection, plots)
    q1_input_stream_fusion_analysis.py -> convergence curves, signed errors over epochs,
                                          rotation-vs-translation scatter (rewritten: the originals
                                          group by the raw `model.fusion` string with a hardcoded
                                          color dict, which can't separate dual fusion from
                                          gated_concat and paints unknown groups black)
    q2_gating_granularity_analysis.py  -> gate trajectories over training, gate saturation,
                                          per-pyramid-level gate heatmap (rewritten per candidate)
    + new: train-vs-val overfitting ratio with the best-epoch marker.

The 7 candidates (model.fusion, model.dual_fusion):
    1 concat | 2 gated | 3 gated_concat | 4 residual_gated_concat |
    5 channelwise_gated_concat_local | 6 channelwise_residual_gated_concat_local |
    7 gated_concat + dual_fusion
Candidate 6 differs from scatter_test_results_by_fusion.py, whose #6 is the
`..._hybrid` variant; the final ablation uses the `..._local` variant, so it is
overridden here (the other script is left untouched).

Run it once per rot_scaling_factor folder so the two scales are never pooled:
    python scripts/evaluate/final_ablation_candidate_analysis.py --run_dir runs/final_ablation
    python scripts/evaluate/final_ablation_candidate_analysis.py --run_dir runs/final_ablation_rot_scaling_200

Needs test results (`scripts/test_multiple_runs.py`) since the candidate table is built from
results/results_*.csv; runs without them are listed in excluded_runs.csv and skipped.

Notes on reading the plots:
  - "Best epoch" is the checkpoint epoch: argmin of val_loss_q + val_loss_y + val_loss_z (the
    fixed-scale composite train.py uses for checkpoint selection), not the uncertainty-weighted val_loss.
  - Early stopping ends runs at different epochs; curves only show epochs where at least
    --min_run_fraction of a candidate's runs are still present, to avoid survivor bias in the tail.
  - Train metrics are averaged over the epoch in train mode (dropout/augmentation), val in eval
    mode, so the val/train ratio is a trend indicator, not an exact generalisation gap.
  - Variation across runs mixes different data splits (data.split_seed) -- bands are not seed noise.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scatter_test_results_by_fusion as scat  # noqa: E402

REPO_ROOT = scat.REPO_ROOT

# ---------------------------------------------------------------------------
# Candidate definition (override #6 of the scatter script's candidate set)
# ---------------------------------------------------------------------------

LOCAL_RES_LABEL = "6. Channelwise local residual sigmoid gated concatenation"


def _override_candidate_6():
    old_label = scat.CANDIDATE_ORDER[5]
    scat.CANDIDATE_ORDER[5] = LOCAL_RES_LABEL
    scat.CANDIDATE_MAP = {
        k: v for k, v in scat.CANDIDATE_MAP.items() if v != old_label
    }
    scat.CANDIDATE_MAP[("channelwise_residual_gated_concat_local", False)] = LOCAL_RES_LABEL
    scat.CANDIDATE_COLORS[LOCAL_RES_LABEL] = scat.CANDIDATE_COLORS.pop(old_label)
    scat.CANDIDATE_SHORT_NAME[LOCAL_RES_LABEL] = "channelwise_residual_gated_concat_local"
    scat.CANDIDATE_SHORT_NAME.pop(old_label, None)


_override_candidate_6()

# (train col, val col, title, unit, multiplier). y/z are stored in meters -> plotted in mm.
METRICS = {
    "rot_x": ("train_abs_rot_x", "val_abs_rot_x", "Rotation error X (roll)", "deg", 1.0),
    "y": ("train_abs_y", "val_abs_y", "Translation error Y", "mm", 1000.0),
    "z": ("train_abs_z", "val_abs_z", "Translation error Z", "mm", 1000.0),
}
SIGNED = [
    ("val_signed_rot_x", "Val signed rot X", "deg", 1.0),
    ("val_signed_y", "Val signed Y", "mm", 1000.0),
    ("val_signed_z", "Val signed Z", "mm", 1000.0),
]
STREAM_COLORS = {"rgb": "#2ca02c", "tactile": "#d62728"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_pair(s):
    """'[0.4,0.6]' -> (0.4, 0.6); anything unparsable -> (nan, nan)."""
    if not isinstance(s, str):
        return np.nan, np.nan
    try:
        vals = [float(x) for x in s.strip("[] ").split(",") if x.strip()]
    except ValueError:
        return np.nan, np.nan
    return (vals[0] if len(vals) > 0 else np.nan, vals[1] if len(vals) > 1 else np.nan)


def _split_pair_col(df, src, dst_rgb, dst_tac):
    if src not in df.columns:
        df[dst_rgb] = np.nan
        df[dst_tac] = np.nan
        return
    pairs = df[src].map(_parse_pair)
    df[dst_rgb] = pairs.map(lambda p: p[0])
    df[dst_tac] = pairs.map(lambda p: p[1])


def load_traces(per_run_df, configs):
    """Concatenate every included run's metrics.csv, tagged with candidate/seed/split_seed."""
    frames = []
    for _, r in per_run_df.iterrows():
        path = Path(r["run_folder"]) / "metrics.csv"
        if not path.exists():
            print(f"Warning: no metrics.csv for {r['run_name']}, skipped in trace plots")
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["run_name"] = r["run_name"]
        df["candidate"] = str(r["candidate"])
        df["seed"] = r["seed"]
        df["split_seed"] = configs[r["run_folder"]].get("data.split_seed")
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    t = pd.concat(frames, ignore_index=True)
    t["val_fixed"] = t["val_loss_q"] + t["val_loss_y"] + t["val_loss_z"]
    t["best_epoch"] = t.groupby("run_name")["val_fixed"].transform(
        lambda s: t.loc[s.idxmin(), "epoch"]
    )
    t["is_best"] = t["epoch"] == t["best_epoch"]
    t["val_trans_yz_mm"] = np.sqrt(t["val_abs_y"] ** 2 + t["val_abs_z"] ** 2) * 1000.0

    _split_pair_col(t, "fusion_weights", "gate_rgb", "gate_tactile")
    _split_pair_col(t, "train_fusion_gate_fraction_lt_0_1", "lt01_rgb", "lt01_tactile")
    _split_pair_col(t, "train_fusion_gate_fraction_gt_0_9", "gt09_rgb", "gt09_tactile")
    for lvl in ("p3", "p4", "p5"):
        _split_pair_col(t, f"train_fusion_gate_{lvl}_mean", f"{lvl}_rgb", f"{lvl}_tactile")
    return t


def candidate_display_names(traces, cands):
    names = {}
    for c in cands:
        n = traces.loc[traces["candidate"] == c, "run_name"].nunique()
        short = scat.CANDIDATE_SHORT_NAME.get(c, c)
        num = c.split(".")[0] if c[:1].isdigit() else "-"
        names[c] = f"{num}. {short} (n={n})"
    return names


def epoch_band(traces, cand, col, min_frac):
    d = traces[traces["candidate"] == cand]
    n_runs = d["run_name"].nunique()
    g = d.groupby("epoch")[col].agg(["mean", "std", "count"]).reset_index()
    return g[g["count"] >= max(1, min_frac * n_runs)]


def save_fig(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots adapted from q1_input_stream_fusion_analysis.py
# ---------------------------------------------------------------------------

def plot_convergence(traces, out_dir, cands, names, colors, min_frac):
    """Train / val error over epochs per candidate, linear (with std band) and log-y."""
    for key, (tcol, vcol, title, unit, mult) in METRICS.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        for row, log in enumerate((False, True)):
            for col, (c, split) in enumerate(((tcol, "Training"), (vcol, "Validation"))):
                ax = axes[row, col]
                for cand in cands:
                    b = epoch_band(traces, cand, c, min_frac)
                    m, s = b["mean"] * mult, b["std"] * mult
                    ax.plot(b["epoch"], m, color=colors[cand], lw=2, label=names[cand])
                    # std band dips <=0 on near-zero errors, undefined on a log axis
                    if not log and s.notna().any():
                        ax.fill_between(b["epoch"], m - s.fillna(0), m + s.fillna(0),
                                        color=colors[cand], alpha=0.12)
                ax.set_title(f"{split} {title}{' (log)' if log else ''}", fontsize=11, fontweight="bold")
                ax.set_ylabel(f"[{unit}]")
                if log:
                    ax.set_yscale("log")
                    ax.set_xlabel("Epoch")
                ax.grid(True, alpha=0.3, which="both" if log else "major")
        axes[0, 0].legend(fontsize=8)
        save_fig(fig, out_dir, f"convergence_{key}")


def plot_signed_errors(traces, out_dir, cands, names, colors, min_frac):
    """Val signed error (bias) over epochs per candidate."""
    fig, axes = plt.subplots(1, len(SIGNED), figsize=(6 * len(SIGNED), 5))
    for ax, (col, title, unit, mult) in zip(np.atleast_1d(axes), SIGNED):
        if col not in traces.columns:
            ax.set_title(f"{title}: column missing")
            continue
        for cand in cands:
            b = epoch_band(traces, cand, col, min_frac)
            m, s = b["mean"] * mult, b["std"] * mult
            ax.plot(b["epoch"], m, color=colors[cand], lw=2, label=names[cand])
            ax.fill_between(b["epoch"], m - s.fillna(0), m + s.fillna(0), color=colors[cand], alpha=0.1)
        ax.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
        ax.set_title(f"{title} (+ over / - under)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"[{unit}]")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    save_fig(fig, out_dir, "signed_errors_over_epochs")


def plot_rot_vs_trans_at_best_epoch(traces, out_dir, cands, names, colors):
    """Rot X vs translation (y/z L2) of the checkpoint epoch, one point per run (label = split seed).
    Uses the best epoch rather than a tail average since that is the checkpoint test.py evaluates."""
    best = traces[traces["is_best"]].drop_duplicates("run_name")
    fig, ax = plt.subplots(figsize=(9, 7))
    for cand in cands:
        d = best[best["candidate"] == cand]
        ax.scatter(d["val_trans_yz_mm"], d["val_abs_rot_x"], s=100, color=colors[cand],
                   edgecolor="black", lw=0.6, alpha=0.85, label=names[cand])
        for _, r in d.iterrows():
            ax.annotate(str(int(r["split_seed"])) if pd.notna(r["split_seed"]) else "?",
                        (r["val_trans_yz_mm"], r["val_abs_rot_x"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Val translation L2 (y/z) at best epoch [mm]")
    ax.set_ylabel("Val rotation error X at best epoch [deg]")
    ax.set_title("Rotation vs translation per fusion candidate (labels = split seed)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(fig, out_dir, "rot_vs_trans_val_best_epoch")


# ---------------------------------------------------------------------------
# New: overfitting
# ---------------------------------------------------------------------------

def plot_overfitting_ratio(traces, out_dir, cands, names, colors, min_frac):
    """Per-run val/train error ratio over epochs, mean per candidate. Dotted vertical
    line = median best (checkpoint) epoch of that candidate."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5))
    t = traces.copy()
    for ax, (key, (tcol, vcol, title, _unit, _mult)) in zip(np.atleast_1d(axes), METRICS.items()):
        t[f"ratio_{key}"] = t[vcol] / t[tcol].replace(0, np.nan)
        for cand in cands:
            b = epoch_band(t, cand, f"ratio_{key}", min_frac)
            ax.plot(b["epoch"], b["mean"], color=colors[cand], lw=2, label=names[cand])
            med_best = t[(t["candidate"] == cand) & t["is_best"]].drop_duplicates("run_name")["epoch"].median()
            ax.axvline(med_best, color=colors[cand], ls=":", alpha=0.6)
        ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
        ax.set_title(f"{title}: val / train", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("val error / train error")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    save_fig(fig, out_dir, "overfitting_val_train_ratio")


# ---------------------------------------------------------------------------
# Plots adapted from q2_gating_granularity_analysis.py
# ---------------------------------------------------------------------------

def plot_gate_trajectories(traces, out_dir, cands, names, colors, min_frac):
    """Gate weight (RGB vs tactile) over training, one panel per gated candidate.
    Dual fusion additionally shows its rotation / translation branch gate means (dashed)."""
    gated = [c for c in cands if traces.loc[traces["candidate"] == c, "gate_rgb"].notna().any()]
    if not gated:
        print("No gate columns found; skipping gate plots")
        return
    ncols = 3
    nrows = int(np.ceil(len(gated) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    for ax, cand in zip(axes.flat, gated):
        for stream, col in (("rgb", "gate_rgb"), ("tactile", "gate_tactile")):
            b = epoch_band(traces, cand, col, min_frac)
            m, s = b["mean"], b["std"].fillna(0)
            ax.plot(b["epoch"], m, lw=2, color=STREAM_COLORS[stream], label=stream)
            ax.fill_between(b["epoch"], (m - s).clip(lower=0), (m + s).clip(upper=1),
                            color=STREAM_COLORS[stream], alpha=0.15)
        for branch, col, ls in (("rotation branch", "train_rotation_fusion_gate_mean", "--"),
                                ("translation branch", "train_translation_fusion_gate_mean", ":")):
            if col in traces.columns and traces.loc[traces["candidate"] == cand, col].notna().any():
                b = epoch_band(traces, cand, col, min_frac)
                ax.plot(b["epoch"], b["mean"], lw=2, color="black", ls=ls, label=branch)
        ax.set_title(names[cand], fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("gate weight (mean ± std over runs, clipped to [0, 1])")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(gated):]:
        ax.axis("off")
    save_fig(fig, out_dir, "gate_trajectories")


def plot_gate_saturation(traces, out_dir, cands, names, colors):
    """Fraction of gate values < 0.1 (closed) and > 0.9 (open) at the best epoch, per stream.
    Dual fusion has no per-stream fraction columns and is left out."""
    best = traces[traces["is_best"]].drop_duplicates("run_name")
    rows = [c for c in cands if best.loc[best["candidate"] == c, "lt01_rgb"].notna().any()]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(len(rows))
    w = 0.38
    for ax, (title, pre) in zip(axes, (("Fraction of gates < 0.1 (closed)", "lt01"),
                                       ("Fraction of gates > 0.9 (open)", "gt09"))):
        for i, stream in enumerate(("rgb", "tactile")):
            g = [best.loc[best["candidate"] == c, f"{pre}_{stream}"] for c in rows]
            ax.bar(x + (i - 0.5) * w, [v.mean() for v in g], w, yerr=[v.std() for v in g],
                   color=STREAM_COLORS[stream], alpha=0.85, capsize=3, label=stream)
        ax.set_xticks(x)
        ax.set_xticklabels([names[c] for c in rows], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    save_fig(fig, out_dir, "gate_saturation_best_epoch")


def plot_gate_level_heatmap(traces, out_dir, cands, names):
    """Mean gate per pyramid level (P3/P4/P5) x stream at the best epoch, for candidates
    that log per-level gates (channelwise ones)."""
    best = traces[traces["is_best"]].drop_duplicates("run_name")
    cols = [f"{lvl}_{s}" for lvl in ("p3", "p4", "p5") for s in ("rgb", "tactile")]
    rows = [c for c in cands if best.loc[best["candidate"] == c, cols].notna().any().any()]
    if not rows:
        return
    data = np.array([[best.loc[best["candidate"] == c, col].mean() for col in cols] for c in rows])
    fig, ax = plt.subplots(figsize=(1.6 * len(cols), 1.0 + 1.0 * len(rows)))
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=max(0.1, np.nanmax(data)), aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([names[c] for c in rows], fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    color="white" if data[i, j] < 0.5 * np.nanmax(data) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean gate")
    ax.set_title("Gate per pyramid level x stream (best epoch)", fontsize=11, fontweight="bold")
    save_fig(fig, out_dir, "gate_level_heatmap_best_epoch")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_trace_summary(traces, out_dir, cands):
    """Per candidate: runs, epochs run, best epoch, epochs trained past the best, val error at best."""
    best = traces[traces["is_best"]].drop_duplicates("run_name").set_index("run_name")
    n_epochs = traces.groupby("run_name")["epoch"].max()
    rows = []
    for c in cands:
        runs = best.index[best["candidate"] == c]
        rows.append({
            "candidate": c,
            "n_runs": len(runs),
            "epochs_run_mean": n_epochs[runs].mean(),
            "best_epoch_median": best.loc[runs, "epoch"].median(),
            "epochs_after_best_mean": (n_epochs[runs] - best.loc[runs, "epoch"]).mean(),
            "val_rot_x_deg_at_best": best.loc[runs, "val_abs_rot_x"].mean(),
            "val_y_mm_at_best": best.loc[runs, "val_abs_y"].mean() * 1000,
            "val_z_mm_at_best": best.loc[runs, "val_abs_z"].mean() * 1000,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "candidate_trace_summary.csv", index=False)
    print("\nPer-candidate training summary (checkpoint = best epoch):")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def config_drift(per_run_df, configs):
    """Config keys that differ between runs of the same candidate. Unlike the scatter script's
    version, data.split_seed is ignored too (it varies by design across the final ablation), as are
    the per-split translation normalisation stats derived from it."""
    ignore = ("training.seed", "data.split_seed", "data.translation_norm_stats")
    lines = []
    for cand, g in per_run_df.groupby("candidate", observed=True):
        if len(g) < 2:
            continue
        flat = [configs[rf] for rf in g["run_folder"]]
        for key in sorted(set().union(*(f.keys() for f in flat))):
            if key.startswith(ignore):
                continue
            vals = [f.get(key, "<missing>") for f in flat]
            if len(set(vals)) > 1:
                lines.append(f"- {cand}: `{key}` = {sorted(set(map(str, vals)))} "
                             f"({', '.join(g['run_name'].tolist())})")
    return lines


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", nargs="+", required=True,
                        help="Run folder(s) / container dir(s). Use one rot_scaling_factor folder at a time.")
    parser.add_argument("--output_dir", default=None,
                        help="Default: runs/candidate_analysis/<first run_dir name> (kept outside the run "
                             "container so test_multiple_runs.py never mistakes it for a run folder).")
    parser.add_argument("--rot_metric", choices=["val_abs_rot_z", "val_abs_rot_x"], default="val_abs_rot_x",
                        help="Rotation column for the test scatter plots (default: roll-only rot_x).")
    parser.add_argument("--min_run_fraction", type=float, default=0.5,
                        help="Show an epoch only if this fraction of a candidate's runs reached it (default 0.5).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else \
        REPO_ROOT / "runs" / "candidate_analysis" / Path(args.run_dir[0]).name
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_folders = scat.discover_run_folders(args.run_dir)
    if not run_folders:
        sys.exit(f"No run folders with results/results_*.csv found under {args.run_dir}")
    per_run_df, per_sample_df, excluded_df, configs = scat.collect_runs(run_folders)
    if per_run_df.empty:
        sys.exit("No runs with usable test results.")

    per_run_df.to_csv(out_dir / "test_comparison_by_run.csv", index=False)
    excluded_df.to_csv(out_dir / "excluded_runs.csv", index=False)
    print(f"Included runs: {len(per_run_df)} | excluded: {len(excluded_df)}")

    rot_col_run = "rot_z_geodesic_deg" if args.rot_metric == "val_abs_rot_z" else "rot_x_deg"
    rot_label = "Rot Error (geodesic, deg)" if args.rot_metric == "val_abs_rot_z" else "Rot Error X / roll (deg)"

    # --- test-set plots + table (reused directly from the scatter script) ---
    scat.plot_scatter_by_run(per_run_df, out_dir, rot_col_run, rot_label)
    scat.plot_scatter_per_sample(per_sample_df, out_dir, args.rot_metric, rot_label)
    summary = scat.flag_comparability(scat.build_summary_table(per_run_df))
    table_md = scat.write_summary_table(summary, out_dir, args.run_dir)
    if table_md:
        print("\nTest-set summary (mean ± std across runs):\n")
        print(table_md)

    missing = [c for c in scat.CANDIDATE_ORDER if c not in set(per_run_df["candidate"].astype(str))]
    if missing:
        print(f"\n⚠️  Candidates with no runs in {args.run_dir}: {missing}")
    extra = sorted(set(per_run_df["candidate"].astype(str)) - set(scat.CANDIDATE_ORDER))
    if extra:
        print(f"⚠️  Runs outside the 7 candidates (plotted under their own label): {extra}")

    drift = config_drift(per_run_df, configs)
    if drift:
        (out_dir / "config_drift_notes.md").write_text(
            "# Config differences between runs of the same candidate (split_seed/seed ignored)\n\n"
            + "\n".join(drift) + "\n")
        print(f"⚠️  {len(drift)} config difference(s) within candidates -- see config_drift_notes.md")

    # --- trace plots (metrics.csv) ---
    traces = load_traces(per_run_df, configs)
    if traces.empty:
        print("No metrics.csv traces found; skipping trace plots")
    else:
        cands = scat.ordered_candidates(traces["candidate"].unique().tolist())
        names = candidate_display_names(traces, cands)
        fallback = {}
        colors = {c: scat.get_color(c, fallback) for c in cands}
        mf = args.min_run_fraction

        plot_convergence(traces, out_dir, cands, names, colors, mf)
        plot_signed_errors(traces, out_dir, cands, names, colors, mf)
        plot_rot_vs_trans_at_best_epoch(traces, out_dir, cands, names, colors)
        plot_overfitting_ratio(traces, out_dir, cands, names, colors, mf)
        plot_gate_trajectories(traces, out_dir, cands, names, colors, mf)
        plot_gate_saturation(traces, out_dir, cands, names, colors)
        plot_gate_level_heatmap(traces, out_dir, cands, names)
        write_trace_summary(traces, out_dir, cands)

    print(f"\nWrote outputs to {out_dir}:")
    for f in sorted(out_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
