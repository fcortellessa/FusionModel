#!/usr/bin/env python3
"""
Interaction of `model.rot_scaling_factor` and uncertainty task weighting.

Every run in the final ablation trains with `task_weighting: uncertainty`
(exp(-log_var) * loss + log_var per task, log_var initialised at 0). The logged
`loss_q` already carries the fixed rot_scaling_factor s, so the weight the
rotation task effectively puts on the raw geodesic error is

    eff_rot   = s * exp(-log_var_rot)
    eff_trans =     exp(-log_var_trans)
    eff_ratio = eff_rot / eff_trans

Minising exp(-lv) * L + lv over lv gives lv* = log(L). If the module has settled,
log_var_rot should therefore sit at log(loss_q) (an offset of log(s) between the two
scales) and eff_rot should be scale-independent; the scale would then only change how
long the module takes to get there from lv = 0. The plots test that:

    uncertainty_over_epochs_by_scale.{png,svg}
        pooled over all runs (median + IQR), one color per scale: log_var_rot/trans,
        effective weight ratio, effective rot/trans weights, and the equilibrium gap
        lv - log(train loss) (0 = settled).
    uncertainty_logvar_by_candidate.{png,svg}
        log_var_rot / log_var_trans over epochs per fusion candidate, one column per scale.
    uncertainty_endpoint_by_candidate.{png,svg}
        effective weights at the checkpoint (best) epoch, per candidate and scale, one dot per run.
    uncertainty_vs_performance.{png,svg}
        effective weight ratio at the best epoch vs val rot_x / translation error, per run.
    paired_scale_effect.{png,svg} + paired_scale_effect.csv
        val error at the best epoch for the same (candidate, split seed) under scale 200 vs 1.0
        (runs sharing a cell are averaged); only cells present in both scales are drawn.
    uncertainty_summary.csv
        per (scale, candidate): log_vars/effective weights at best and last epoch and how many
        epochs until lv tracks log(loss) within --tol.

Scale is read from each run's config (`model.rot_scaling_factor`; unset = the quaternion
default of 200 -- this script assumes the quaternion head, as in the final ablation).
Candidates are the same 7 as final_ablation_candidate_analysis.py (which it reuses).

Usage (both scale folders, default):
    python scripts/evaluate/uncertainty_scale_analysis.py \\
        --run_dir runs/final_ablation runs/final_ablation_rot_scaling_200
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
import final_ablation_candidate_analysis as fca  # noqa: E402  (also applies the candidate-6 override)

scat = fca.scat
REPO_ROOT = fca.REPO_ROOT

DEFAULT_SCALE = 200.0  # quaternion total_loss default when model.rot_scaling_factor is unset
SCALE_COLORS = {200.0: "#d95f02", 1.0: "#1b9e77"}
FALLBACK_SCALE_COLORS = ["#7570b3", "#e7298a", "#66a61e"]


def scale_label(s):
    return f"rot_scaling_factor={s:g}"


def scale_color(s):
    if s not in SCALE_COLORS:
        SCALE_COLORS[s] = FALLBACK_SCALE_COLORS[len(SCALE_COLORS) % len(FALLBACK_SCALE_COLORS)]
    return SCALE_COLORS[s]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def build_traces(run_dirs):
    run_folders = scat.discover_run_folders(run_dirs)
    if not run_folders:
        sys.exit(f"No run folders with results/results_*.csv found under {run_dirs}")
    per_run_df, _, excluded_df, configs = scat.collect_runs(run_folders)
    if per_run_df.empty:
        sys.exit("No runs with usable test results.")
    if not excluded_df.empty:
        print(f"Excluded {len(excluded_df)} run(s):\n{excluded_df.to_string(index=False)}")

    traces = fca.load_traces(per_run_df, configs)
    scale_by_run = {}
    for _, r in per_run_df.iterrows():
        cfg = configs[r["run_folder"]]
        if cfg.get("model.head_type", "quaternion") != "quaternion":
            print(f"Warning: {r['run_name']} is not a quaternion-head run; effective-weight formula may not apply")
        s = cfg.get("model.rot_scaling_factor")
        scale_by_run[r["run_name"]] = DEFAULT_SCALE if s is None else float(s)
    traces["scale"] = traces["run_name"].map(scale_by_run)

    needed = {"uncertainty_log_var_rot", "uncertainty_log_var_trans", "loss_q", "loss_y", "loss_z"}
    missing = needed - set(traces.columns)
    if missing:
        sys.exit(f"metrics.csv lacks uncertainty columns {missing}; were these runs trained with task_weighting?")
    traces = traces.dropna(subset=["uncertainty_log_var_rot", "uncertainty_log_var_trans"]).copy()

    traces["eff_rot"] = traces["scale"] * np.exp(-traces["uncertainty_log_var_rot"])
    traces["eff_trans"] = np.exp(-traces["uncertainty_log_var_trans"])
    traces["eff_ratio"] = traces["eff_rot"] / traces["eff_trans"]
    # lv* = log(L) minimises exp(-lv) * L + lv; gap = 0 means the module has settled
    traces["gap_rot"] = traces["uncertainty_log_var_rot"] - np.log(traces["loss_q"])
    traces["gap_trans"] = traces["uncertainty_log_var_trans"] - np.log(traces["loss_y"] + traces["loss_z"])
    return traces


def band(df, col, min_frac):
    """Median + IQR per epoch over the runs in df, only where >= min_frac of the runs are still present."""
    n_runs = df["run_name"].nunique()
    g = df.groupby("epoch")[col].agg(
        med="median", lo=lambda s: s.quantile(0.25), hi=lambda s: s.quantile(0.75), count="count"
    ).reset_index()
    return g[g["count"] >= max(1, min_frac * n_runs)]


def _draw(ax, b, color, label=None, ls="-", fill=True):
    ax.plot(b["epoch"], b["med"], color=color, lw=2, ls=ls, label=label)
    if fill:
        ax.fill_between(b["epoch"], b["lo"], b["hi"], color=color, alpha=0.15)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_over_epochs_by_scale(traces, out_dir, min_frac):
    scales = sorted(traces["scale"].unique(), reverse=True)
    panels = [
        ("uncertainty_log_var_rot", "log_var_rot", False),
        ("uncertainty_log_var_trans", "log_var_trans", False),
        ("eff_ratio", "effective weight ratio  rot / trans  (log)", True),
        ("eff_rot", "effective rotation weight  s·exp(-log_var_rot)  (log)", True),
        ("eff_trans", "effective translation weight  exp(-log_var_trans)  (log)", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for ax, (col, title, log) in zip(axes.flat, panels):
        for s in scales:
            d = traces[traces["scale"] == s]
            _draw(ax, band(d, col, min_frac), scale_color(s),
                  f"{scale_label(s)} (n={d['run_name'].nunique()})")
        ax.set_title(title, fontsize=10, fontweight="bold")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3, which="both" if log else "major")
    axes[0, 0].legend(fontsize=9)

    ax = axes[1, 2]
    for s in scales:
        d = traces[traces["scale"] == s]
        _draw(ax, band(d, "gap_rot", min_frac), scale_color(s), f"rot, s={s:g}")
        _draw(ax, band(d, "gap_trans", min_frac), scale_color(s), f"trans, s={s:g}", ls="--", fill=False)
    ax.axhline(0, color="black", lw=1, alpha=0.6)
    ax.set_title("equilibrium gap  log_var - log(train loss)   (0 = settled)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Uncertainty weighting vs rot_scaling_factor (all runs pooled; median, IQR band)", fontsize=12)
    fca.save_fig(fig, out_dir, "uncertainty_over_epochs_by_scale")


def plot_logvar_by_candidate(traces, out_dir, cands, names, colors, min_frac):
    scales = sorted(traces["scale"].unique(), reverse=True)
    fig, axes = plt.subplots(2, len(scales), figsize=(7 * len(scales), 9), sharey="row", squeeze=False)
    for j, s in enumerate(scales):
        d_s = traces[traces["scale"] == s]
        for i, (col, title) in enumerate((("uncertainty_log_var_rot", "log_var_rot"),
                                          ("uncertainty_log_var_trans", "log_var_trans"))):
            ax = axes[i, j]
            for c in cands:
                d = d_s[d_s["candidate"] == c]
                if d.empty:
                    continue
                _draw(ax, band(d, col, min_frac), colors[c],
                      f"{names[c].rsplit(' (n=', 1)[0]} (n={d['run_name'].nunique()})", fill=False)
            ax.set_title(f"{title}, {scale_label(s)}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fca.save_fig(fig, out_dir, "uncertainty_logvar_by_candidate")


def plot_endpoint_by_candidate(traces, out_dir, cands, names):
    best = traces[traces["is_best"]].drop_duplicates("run_name")
    scales = sorted(best["scale"].unique(), reverse=True)
    panels = [("eff_rot", "effective rotation weight"), ("eff_trans", "effective translation weight"),
              ("eff_ratio", "effective ratio rot / trans")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    x = np.arange(len(cands))
    w = 0.36
    for ax, (col, title) in zip(axes, panels):
        for k, s in enumerate(scales):
            off = (k - (len(scales) - 1) / 2) * w
            for i, c in enumerate(cands):
                v = best.loc[(best["candidate"] == c) & (best["scale"] == s), col].to_numpy()
                if len(v) == 0:
                    continue
                ax.scatter(i + off + rng.uniform(-0.05, 0.05, len(v)), v, s=28, color=scale_color(s), alpha=0.7,
                           label=scale_label(s) if i == 0 or not ax.get_legend_handles_labels()[0] else None)
                ax.hlines(np.median(v), i + off - 0.13, i + off + 0.13, color="black", lw=2)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([names[c].rsplit(" (n=", 1)[0] for c in cands], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{title} at best epoch (bar = median)", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0].legend(uniq.values(), uniq.keys(), fontsize=8)
    fca.save_fig(fig, out_dir, "uncertainty_endpoint_by_candidate")


def plot_vs_performance(traces, out_dir):
    best = traces[traces["is_best"]].drop_duplicates("run_name")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (ycol, ylabel) in zip(axes, (("val_abs_rot_x", "val rot X at best epoch [deg]"),
                                         ("val_trans_yz_mm", "val translation y/z L2 at best epoch [mm]"))):
        for s in sorted(best["scale"].unique(), reverse=True):
            d = best[best["scale"] == s]
            ax.scatter(d["eff_ratio"], d[ycol], s=45, color=scale_color(s), alpha=0.75,
                       edgecolor="black", lw=0.4, label=f"{scale_label(s)} (n={len(d)})")
        ax.set_xscale("log")
        ax.set_xlabel("effective weight ratio rot / trans at best epoch (log)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fca.save_fig(fig, out_dir, "uncertainty_vs_performance")


def plot_paired_scale_effect(traces, out_dir, cands, colors):
    """Same (candidate, split seed) under the two scales, val error at the best epoch."""
    scales = sorted(traces["scale"].unique(), reverse=True)
    if len(scales) != 2:
        print(f"Paired plot needs exactly 2 scales, found {scales}; skipping")
        return
    hi, lo = scales  # e.g. 200, 1.0
    best = traces[traces["is_best"]].drop_duplicates("run_name").copy()
    rows = []
    fig, axes = plt.subplots(1, len(fca.METRICS), figsize=(6 * len(fca.METRICS), 5.5))
    for ax, (key, (_t, vcol, title, unit, mult)) in zip(np.atleast_1d(axes), fca.METRICS.items()):
        best["_v"] = best[vcol] * mult
        piv = best.pivot_table(index=["candidate", "split_seed"], columns="scale", values="_v", aggfunc="mean")
        piv = piv.dropna(subset=[hi, lo])
        for (cand, split), r in piv.iterrows():
            ax.plot([0, 1], [r[hi], r[lo]], color=colors[cand], alpha=0.7, marker="o", ms=4)
            rows.append({"metric": key, "unit": unit, "candidate": cand, "split_seed": split,
                         f"val_scale_{hi:g}": r[hi], f"val_scale_{lo:g}": r[lo], "delta_lo_minus_hi": r[lo] - r[hi]})
        d = (piv[lo] - piv[hi]).to_numpy()
        ax.set_xticks([0, 1])
        ax.set_xticklabels([scale_label(hi), scale_label(lo)])
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylabel(f"[{unit}]")
        ax.set_title(f"{title}\n{len(d)} pairs, mean Δ({lo:g}-{hi:g}) = {d.mean():+.3f} {unit}, "
                     f"{(d < 0).sum()} better at s={lo:g}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], color=colors[c], lw=2, label=fca_name(c)) for c in cands]
    np.atleast_1d(axes)[0].legend(handles=handles, fontsize=7)
    fca.save_fig(fig, out_dir, "paired_scale_effect")
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "paired_scale_effect.csv", index=False)


def fca_name(c):
    return f"{c.split('.')[0]}. {scat.CANDIDATE_SHORT_NAME.get(c, c)}" if c[:1].isdigit() else c


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def epochs_to_settle(run_df, col, tol):
    """First epoch from which |gap| stays < tol until the end; NaN if it never does."""
    d = run_df.sort_values("epoch")
    bad = d.loc[d[col].abs() >= tol, "epoch"]
    if bad.empty:
        return float(d["epoch"].iloc[0])
    nxt = d.loc[d["epoch"] > bad.max(), "epoch"]
    return float(nxt.iloc[0]) if not nxt.empty else np.nan


def write_summary(traces, out_dir, cands, tol):
    best = traces[traces["is_best"]].drop_duplicates("run_name").set_index("run_name")
    last = traces.sort_values("epoch").groupby("run_name").tail(1).set_index("run_name")
    settle = pd.DataFrame({
        "settle_rot": traces.groupby("run_name").apply(lambda g: epochs_to_settle(g, "gap_rot", tol)),
        "settle_trans": traces.groupby("run_name").apply(lambda g: epochs_to_settle(g, "gap_trans", tol)),
    })
    rows = []
    for s in sorted(traces["scale"].unique(), reverse=True):
        for c in cands:
            runs = best.index[(best["candidate"] == c) & (best["scale"] == s)]
            if len(runs) == 0:
                continue
            rows.append({
                "scale": s, "candidate": c, "n_runs": len(runs),
                "best_epoch_median": best.loc[runs, "epoch"].median(),
                "log_var_rot_best": best.loc[runs, "uncertainty_log_var_rot"].mean(),
                "log_var_trans_best": best.loc[runs, "uncertainty_log_var_trans"].mean(),
                "eff_rot_best_median": best.loc[runs, "eff_rot"].median(),
                "eff_trans_best_median": best.loc[runs, "eff_trans"].median(),
                "eff_ratio_best_median": best.loc[runs, "eff_ratio"].median(),
                "eff_ratio_last_median": last.loc[runs, "eff_ratio"].median(),
                "gap_rot_last_mean": last.loc[runs, "gap_rot"].mean(),
                "gap_trans_last_mean": last.loc[runs, "gap_trans"].mean(),
                f"epochs_to_settle_rot(tol={tol})": settle.loc[runs, "settle_rot"].mean(),
                f"epochs_to_settle_trans(tol={tol})": settle.loc[runs, "settle_trans"].mean(),
                "runs_never_settled_rot": int(settle.loc[runs, "settle_rot"].isna().sum()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "uncertainty_summary.csv", index=False)

    pooled = []
    for s in sorted(traces["scale"].unique(), reverse=True):
        runs = best.index[best["scale"] == s]
        pooled.append({
            "scale": s, "n_runs": len(runs),
            "log_var_rot_last": last.loc[runs, "uncertainty_log_var_rot"].mean(),
            "eff_rot_last_median": last.loc[runs, "eff_rot"].median(),
            "eff_trans_last_median": last.loc[runs, "eff_trans"].median(),
            "eff_ratio_last_median": last.loc[runs, "eff_ratio"].median(),
            "gap_rot_last": last.loc[runs, "gap_rot"].mean(),
            f"settle_rot(tol={tol})": settle.loc[runs, "settle_rot"].mean(),
            "never_settled_rot": int(settle.loc[runs, "settle_rot"].isna().sum()),
        })
    print("\nPooled by scale (last epoch; 'settle' = mean epoch from which |log_var - log loss| stays < tol):")
    print(pd.DataFrame(pooled).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", nargs="+",
                        default=["runs/final_ablation", "runs/final_ablation_rot_scaling_200"],
                        help="Run container dirs; scale is read per run, so they can be mixed.")
    parser.add_argument("--output_dir", default="runs/candidate_analysis/uncertainty_scale")
    parser.add_argument("--min_run_fraction", type=float, default=0.5,
                        help="Show an epoch only if this fraction of the runs in a curve reached it.")
    parser.add_argument("--tol", type=float, default=0.5,
                        help="|log_var - log(train loss)| below which the weighting counts as settled (default 0.5).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = build_traces(args.run_dir)
    cands = scat.ordered_candidates(traces["candidate"].unique().tolist())
    names = fca.candidate_display_names(traces, cands)
    fallback = {}
    colors = {c: scat.get_color(c, fallback) for c in cands}
    print(f"Runs per scale: {traces.groupby('scale')['run_name'].nunique().to_dict()}")

    plot_over_epochs_by_scale(traces, out_dir, args.min_run_fraction)
    plot_logvar_by_candidate(traces, out_dir, cands, names, colors, args.min_run_fraction)
    plot_endpoint_by_candidate(traces, out_dir, cands, names)
    plot_vs_performance(traces, out_dir)
    plot_paired_scale_effect(traces, out_dir, cands, colors)
    write_summary(traces, out_dir, cands, args.tol)

    print(f"\nWrote outputs to {out_dir}:")
    for f in sorted(out_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
