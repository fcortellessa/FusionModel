#!/usr/bin/env python3
"""
Sample-level error structure of the final ablation: which test samples are hard, for whom, and what
that means for comparing fusion candidates.

Background (see roll_position_error_analysis.py): a handful of test samples carry a large share of the
rot_x error, the same samples are hard for every model, and the split seed explains most of the
between-run variance. The plots here are built around those facts instead of run means:

  1. error_heatmap_split_<N>       test samples (rows, hardest first) x runs (columns, grouped by
                                   candidate); log color = |roll error| (left) and y/z translation L2
                                   (right). Vertical streaks = samples that are hard for every model.
     error_heatmap_summary.csv     per split: how much of the tail is shared between runs.
  2. montage_hardest / montage_typical   RGB + DIGIT image pair of the hardest (and, for contrast,
                                   median-difficulty) test samples with ground-truth / predicted roll,
                                   translation error and grasp-group context.
     hardest_samples_with_files.csv       the top-30 hardest samples with their timestamps + image paths.
  2b. hard_samples_roll_swap       predicted vs ground-truth roll with an identity and a mirror line: are the
                                   hard samples a swap between the two ends of the roll range?
  4. ecdf_errors_by_candidate      log-x ECDFs of |roll|, translation L2, |y|, |z| error per candidate
                                   (suits heavy tails better than means / boxplots).
     success_rate_table.csv        fraction of samples under fixed error thresholds, per candidate.
  5a. split_centered_run_metrics   per-run metric as % above/below the average run of the SAME split
                                   (removes split difficulty), one dot per run colored by split.
  5b. paired_diff_vs_concat        per-sample paired difference vs concat on the same test samples
                                   (per candidate+split the runs are averaged first), hard samples
                                   (top 5% by mean error in their split) in red.
     paired_vs_concat.csv          mean / median / bootstrap CI of the paired differences, incl. the
                                   mean without the hard samples and the per-split means.

Mapping test-sample ids to image files (needed only for the montage). test.py's `sample_id` is the
position in os.listdir() of Data_split/test/test_preprocessing, which only exists for the last
split that was built. So each split's test set is re-derived from its seed with the split code's own
functions (group_triplets_by_pose + _assign_groups_to_ratios, nothing is copied or modified), listed
through a scratch directory to reproduce the listdir order, and every split is verified against the
ground-truth y/z stored in the results csvs. A split that does not verify is skipped in the montage.
(On the machine this was written on, the reconstructed order was identical to the real directory's
for the split that still exists on disk, and matched the results' ground truth on all five splits.)

Usage (once per rot_scaling_factor folder):
    python scripts/evaluate/sample_level_error_analysis.py --run_dir runs/final_ablation
    python scripts/evaluate/sample_level_error_analysis.py --run_dir runs/final_ablation_rot_scaling_200
Default output: <run_dir>/overview/sample_level_analysis/   (--skip_montage to avoid reading images)
"""

import argparse
import ast
import os
import random
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_ablation_candidate_analysis as fca  # noqa: E402  (also applies the candidate-6 override)
import roll_position_error_analysis as rpea  # noqa: E402

scat = fca.scat
REPO_ROOT = fca.REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REFERENCE_CANDIDATE = scat.CANDIDATE_ORDER[0]  # concat: the baseline in CLAUDE.md
SPLIT_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
ROT_THRESHOLDS_DEG = [0.1, 0.25, 0.5, 1, 2, 5]
MM_THRESHOLDS = [0.1, 0.25, 0.5, 1, 2]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(run_dir):
    ps = rpea.load_samples([run_dir])
    ps["split"] = ps["split"].astype(int)
    ps["trans_mm"] = np.hypot(ps["abs_y_mm"], ps["abs_z_mm"])
    cands = scat.ordered_candidates(ps["candidate"].unique().tolist())
    names = fca.candidate_display_names(ps, cands)
    fallback = {}
    colors = {c: scat.get_color(c, fallback) for c in cands}
    return ps, cands, names, colors


def cand_number(c):
    return c.split(".")[0] if c[:1].isdigit() else "-"


# ---------------------------------------------------------------------------
# 1. error heatmaps
# ---------------------------------------------------------------------------

def plot_error_heatmaps(ps, out_dir, cands, names, colors):
    rows = []
    for split, g in ps.groupby("split"):
        runs = g[["run_name", "candidate"]].drop_duplicates()
        runs["ord"] = runs["candidate"].map({c: i for i, c in enumerate(cands)})
        runs = runs.sort_values(["ord", "run_name"]).reset_index(drop=True)
        cols = runs["run_name"].tolist()
        rot = g.pivot_table(index="sample_id", columns="run_name", values="abs_roll")[cols]
        trn = g.pivot_table(index="sample_id", columns="run_name", values="trans_mm")[cols]
        order = rot.mean(axis=1).sort_values(ascending=False).index
        rot, trn = rot.loc[order], trn.loc[order]

        fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.05 * len(cols) * 2), 11), sharey=True)
        for ax, mat, title, unit in ((axes[0], rot, "|roll error| [deg]", "deg"),
                                     (axes[1], trn, "translation L2 (y/z) [mm]", "mm")):
            v = mat.to_numpy()
            vmin = max(float(np.nanquantile(v, 0.02)), 1e-3)
            vmax = float(np.nanquantile(v, 0.999))
            im = ax.imshow(v, aspect="auto", interpolation="nearest", cmap="viridis",
                           norm=LogNorm(vmin=vmin, vmax=vmax))
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([cand_number(c) for c in runs["candidate"]], fontsize=9)
            for tick, c in zip(ax.get_xticklabels(), runs["candidate"]):
                tick.set_color(colors[c])
                tick.set_fontweight("bold")
            top5 = int(np.ceil(0.05 * len(mat)))
            ax.axhline(top5 - 0.5, color="red", lw=1, ls="--")
            ax.set_title(f"{title}\n(rows sorted by mean error over runs; red line = top 5%)", fontsize=10)
            ax.set_xlabel("run (label = candidate number)")
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label=f"error [{unit}] (log)")
        axes[0].set_ylabel(f"test sample, hardest first (n={len(rot)})")
        fig.legend(handles=[Patch(color=colors[c], label=names[c].rsplit(" (n=", 1)[0]) for c in cands],
                   loc="lower center", ncol=2, fontsize=8, frameon=False)
        fig.suptitle(f"Per-sample test error, split seed {split} (same test samples for every run)", fontsize=12)
        fig.tight_layout(rect=[0, 0.07, 1, 0.97])
        fig.savefig(out_dir / f"error_heatmap_split_{split}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        # how much of the tail is shared? (tail = above that run's own 95th percentile)
        thr = rot.quantile(0.95)
        above = rot.gt(thr, axis=1)
        frac = above.mean(axis=1)
        top_n = int(np.ceil(0.05 * len(rot)))
        rows.append({
            "split": split, "n_runs": len(cols), "n_samples": len(rot),
            "top5pct_share_of_mean_error": rot.mean(axis=1).iloc[:top_n].sum() / rot.mean(axis=1).sum(),
            "top5pct_samples_in_tail_of_>=80%_runs": float((frac.iloc[:top_n] >= 0.8).mean()),
            "samples_in_tail_of_>=80%_runs": int((frac >= 0.8).sum()),
            "samples_in_tail_of_exactly_1_run": int((above.sum(axis=1) == 1).sum()),
            "median_pairwise_spearman_across_runs": float(np.nanmedian(
                rot.corr(method="spearman").to_numpy()[np.triu_indices(len(cols), 1)])),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "error_heatmap_summary.csv", index=False)
    print("\nShared-tail summary (tail = above a run's own 95th percentile of |roll error|):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


# ---------------------------------------------------------------------------
# 4. ECDFs + success rates
# ---------------------------------------------------------------------------

def plot_ecdf(ps, out_dir, cands, names, colors):
    panels = [("abs_roll", "|roll error| [deg]", ROT_THRESHOLDS_DEG),
              ("trans_mm", "translation L2 (y/z) error [mm]", MM_THRESHOLDS),
              ("abs_y_mm", "|y error| [mm]", MM_THRESHOLDS),
              ("abs_z_mm", "|z error| [mm]", MM_THRESHOLDS)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rows = []
    for ax, (col, label, thresholds) in zip(axes.flat, panels):
        allv = ps[col].to_numpy()
        xmin = max(float(np.quantile(allv[allv > 0], 0.001)), 1e-3)
        for cand in cands:
            v = np.sort(ps.loc[ps["candidate"] == cand, col].to_numpy())
            y = np.arange(1, len(v) + 1) / len(v)
            ax.step(np.maximum(v, xmin), y, where="post", color=colors[cand], lw=1.8, label=names[cand])
            for t in thresholds:
                rows.append({"candidate": cand, "metric": col, "threshold": t,
                             "fraction_le_threshold": float((v <= t).mean()), "n_samples": len(v)})
        for t in thresholds:
            ax.axvline(t, color="gray", lw=0.6, ls=":")
        for q in (0.5, 0.9, 0.99):
            ax.axhline(q, color="gray", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(xmin, allv.max() * 1.1)
        ax.set_ylim(0, 1.005)
        ax.set_xlabel(label + " (log)")
        ax.set_ylabel("fraction of test samples with error <= x")
        ax.grid(True, alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Test-error ECDFs per candidate (all runs and splits of a candidate pooled; dotted = thresholds / 50-90-99%)",
                 fontsize=11)
    fca.save_fig(fig, out_dir, "ecdf_errors_by_candidate")

    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "success_rate_table.csv", index=False)
    pv = tab[tab["metric"] == "abs_roll"].pivot(index="candidate", columns="threshold", values="fraction_le_threshold")
    pv.index = [cand_number(c) + ". " + scat.CANDIDATE_SHORT_NAME.get(c, c) for c in pv.index]
    print("\nFraction of test samples with |roll error| <= threshold [deg]:")
    print(pv.to_string(float_format=lambda x: f"{x:.3f}"))


# ---------------------------------------------------------------------------
# 5a. split-centred run metrics
# ---------------------------------------------------------------------------

def plot_split_centered(ps, out_dir, cands, names, colors):
    runm = (ps.groupby(["split", "candidate", "run_name"])
            .agg(rot_mean=("abs_roll", "mean"), rot_median=("abs_roll", "median"),
                 y_mean=("abs_y_mm", "mean"), z_mean=("abs_z_mm", "mean"))
            .reset_index())
    panels = [("rot_mean", "mean |roll error|"), ("rot_median", "median |roll error|"),
              ("y_mean", "mean |y error|"), ("z_mean", "mean |z error|")]
    splits = sorted(runm["split"].unique())
    split_color = {s: SPLIT_COLORS[i % len(SPLIT_COLORS)] for i, s in enumerate(splits)}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    print("\nCandidate mean of (run metric vs average run of its split, %); negative = better than the split average:")
    summary = {}
    for ax, (col, title) in zip(axes.flat, panels):
        runm[col + "_rel"] = 100.0 * (runm[col] / runm.groupby("split")[col].transform("mean") - 1.0)
        for i, cand in enumerate(cands):
            d = runm[runm["candidate"] == cand]
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d[col + "_rel"], s=55,
                       c=[split_color[s] for s in d["split"]], edgecolor="black", lw=0.5, zorder=3)
            ax.hlines(d[col + "_rel"].mean(), i - 0.3, i + 0.3, color="black", lw=2.5, zorder=4)
            summary.setdefault(cand, {})[title] = d[col + "_rel"].mean()
        ax.axhline(0, color="gray", lw=1)
        ax.set_xticks(range(len(cands)))
        ax.set_xticklabels([f"{cand_number(c)}. {scat.CANDIDATE_SHORT_NAME.get(c, c)}\n(n={int((runm['candidate'] == c).sum())})"
                            for c in cands], rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("% vs average run of the same split")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    axes[0, 0].legend(handles=[Patch(color=split_color[s], label=f"split {s}") for s in splits], fontsize=8)
    fig.suptitle("Run metrics with split difficulty removed (black bar = candidate mean)", fontsize=12)
    fca.save_fig(fig, out_dir, "split_centered_run_metrics")
    print(pd.DataFrame(summary).T.rename(index=lambda c: cand_number(c) + ". " + scat.CANDIDATE_SHORT_NAME.get(c, c))
          .to_string(float_format=lambda x: f"{x:+.1f}"))


# ---------------------------------------------------------------------------
# 5b. paired per-sample differences vs concat
# ---------------------------------------------------------------------------

def _boot_ci(d, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def plot_paired_vs_concat(ps, out_dir, cands, names, colors):
    if REFERENCE_CANDIDATE not in cands:
        print(f"No {REFERENCE_CANDIDATE} runs; skipping paired comparison")
        return
    metrics = [("abs_roll", "Δ |roll error| [deg]", 0.05), ("abs_y_mm", "Δ |y error| [mm]", 0.02),
               ("abs_z_mm", "Δ |z error| [mm]", 0.02)]
    others = [c for c in cands if c != REFERENCE_CANDIDATE]
    splits = sorted(ps["split"].unique())
    split_color = {s: SPLIT_COLORS[i % len(SPLIT_COLORS)] for i, s in enumerate(splits)}
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    out_rows = []
    for ax, (col, ylabel, linthresh) in zip(axes, metrics):
        per = ps.groupby(["split", "candidate", "sample_id"])[col].mean().unstack("candidate")
        overall = ps.groupby(["split", "sample_id"])[col].mean()
        hard_flag = overall.groupby(level="split").rank(pct=True) > 0.95
        ticklabels = []
        for i, cand in enumerate(others):
            if cand not in per.columns:
                ticklabels.append(cand_number(cand))
                continue
            d = (per[cand] - per[REFERENCE_CANDIDATE]).dropna()
            if d.empty:
                ticklabels.append(f"{cand_number(cand)}\n(no pairs)")
                continue
            hard = hard_flag.reindex(d.index.droplevel("sample_id").to_frame().assign(
                sample_id=d.index.get_level_values("sample_id")).set_index(["split", "sample_id"]).index).to_numpy()
            dv = d.to_numpy()
            x = i + rng.uniform(-0.28, 0.28, len(dv))
            ax.scatter(x[~hard], dv[~hard], s=6, color="#999999", alpha=0.35, linewidth=0, zorder=2)
            ax.scatter(x[hard], dv[hard], s=16, color="#d62728", alpha=0.85, linewidth=0, zorder=3,
                       label="hard samples (top 5% in their split)" if i == 0 else None)
            mean, med = float(dv.mean()), float(np.median(dv))
            lo, hi = _boot_ci(dv)
            ax.errorbar([i], [mean], yerr=[[mean - lo], [hi - mean]], fmt="D", color="black", ms=8, capsize=4, zorder=5,
                        label="mean Δ (95% bootstrap CI over samples)" if i == 0 else None)
            ax.plot([i], [med], "^", color="#1f77b4", ms=8, zorder=5, label="median Δ" if i == 0 else None)
            per_split = {}
            for s in splits:
                m = d.index.get_level_values("split") == s
                if m.any():
                    per_split[s] = float(dv[m].mean())
                    ax.plot([i + 0.4], [per_split[s]], "s", color=split_color[s], ms=6, zorder=5)
            dx = dv[~hard]
            lo_x, hi_x = _boot_ci(dx) if len(dx) > 1 else (np.nan, np.nan)
            share_better = 100.0 * float((dv < 0).mean())
            out_rows.append({"metric": col, "candidate": cand, "reference": REFERENCE_CANDIDATE, "n_pairs": len(dv),
                             "n_splits": len(per_split), "mean_delta": mean, "ci95_lo": lo, "ci95_hi": hi,
                             "median_delta": med, "pct_samples_better_than_ref": share_better,
                             "mean_delta_excl_top5pct": float(dx.mean()), "ci95_lo_excl": lo_x, "ci95_hi_excl": hi_x,
                             **{f"mean_delta_split_{s}": v for s, v in per_split.items()}})
            ticklabels.append(f"{cand_number(cand)}. {scat.CANDIDATE_SHORT_NAME.get(cand, cand)}\n"
                              f"mean {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]\nexcl. hard {dx.mean():+.3f}\n"
                              f"{share_better:.0f}% better, n={len(dv)}")
        ax.axhline(0, color="black", lw=1)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_xticks(range(len(others)))
        ax.set_xticklabels(ticklabels, fontsize=7)
        ax.set_ylabel(ylabel + "  (+ = worse than concat, symlog)")
        ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.legend(handles=axes[0].get_legend_handles_labels()[0] + [
        Patch(color=split_color[s], label=f"split {s} mean") for s in splits],
        fontsize=8, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 0.972), frameon=False)
    fig.suptitle("Per-sample paired difference vs concat on the same test samples "
                 "(runs of a candidate averaged within each split first; squares = per-split means)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fca.save_fig(fig, out_dir, "paired_diff_vs_concat")
    df = pd.DataFrame(out_rows)
    df.to_csv(out_dir / "paired_vs_concat.csv", index=False)
    print("\nPaired difference vs concat (+ = worse), 95% bootstrap CI over test samples:")
    show = df[["metric", "candidate", "n_pairs", "mean_delta", "ci95_lo", "ci95_hi", "median_delta",
               "pct_samples_better_than_ref", "mean_delta_excl_top5pct"]].copy()
    show["candidate"] = show["candidate"].map(lambda c: cand_number(c) + ". " + scat.CANDIDATE_SHORT_NAME.get(c, c))
    print(show.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


# ---------------------------------------------------------------------------
# 2b. are the hard samples a swap between the two ends of the roll range?
# ---------------------------------------------------------------------------

def plot_roll_swap(ps, out_dir, top_frac=0.05):
    """Predicted vs ground-truth roll (mean over runs) per test sample; the top-5% hardest per split in red.
    Mirror line: pred = 2*mid - gt (mid = middle of the observed roll range). A 'swap' failure sits on it."""
    s = rpea.per_sample_means(ps)
    s["pred_roll"] = s["gt_roll"] + s["signed_roll"]
    mid = 0.5 * (s["gt_roll"].min() + s["gt_roll"].max())
    s["hard"] = s.groupby("split")["abs_roll"].rank(pct=True) > (1 - top_frac)
    h = s[s["hard"]]
    slope = float(np.polyfit(h["gt_roll"] - mid, h["signed_roll"], 1)[0])
    d_ident = (h["pred_roll"] - h["gt_roll"]).abs()
    d_mirror = (h["pred_roll"] - (2 * mid - h["gt_roll"])).abs()
    share_mirror = float((d_mirror < d_ident).mean())
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.scatter(s.loc[~s["hard"], "gt_roll"], s.loc[~s["hard"], "pred_roll"], s=8, color="#999999", alpha=0.4, linewidth=0,
               label="other samples")
    ax.scatter(h["gt_roll"], h["pred_roll"], s=28, color="#d62728", edgecolor="black", lw=0.4,
               label=f"hardest {int(top_frac * 100)}% per split (n={len(h)})")
    lim = np.array([s["gt_roll"].min() - 1, s["gt_roll"].max() + 1])
    ax.plot(lim, lim, "k--", lw=1, label="identity (perfect)")
    ax.plot(lim, 2 * mid - lim, color="#d62728", ls="--", lw=1, label=f"mirror through {mid:.1f}° (swap between the range ends)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ground-truth roll [deg]")
    ax.set_ylabel("predicted roll [deg] (mean over the runs of the split)")
    ax.set_title(f"Hard samples vs identity / mirror line\n"
                 f"error-vs-offset slope of hard samples {slope:+.2f} (0 = unrelated to position, -2 = perfect mirror); "
                 f"{100 * share_mirror:.0f}% closer to the mirror line than to identity", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fca.save_fig(fig, out_dir, "hard_samples_roll_swap")
    print(f"\nHard samples (top {int(top_frac * 100)}% per split): error-vs-offset slope {slope:+.2f}; "
          f"{100 * share_mirror:.0f}% lie closer to the mirror line than to the identity line (mid roll {mid:.1f}°)")


# ---------------------------------------------------------------------------
# 2. montage of hard samples (needs sample_id -> image mapping)
# ---------------------------------------------------------------------------

def _transf_matrix():
    """The fixed hub-frame calibration matrix, read from preprocess_data.py's source without executing it."""
    tree = ast.parse((REPO_ROOT / "data_processing" / "preprocess_data.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "transf_matrix" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("transf_matrix not found in preprocess_data.py")


def build_sample_index(ps, data_root, ratios=(0.8, 0.15, 0.05)):
    """{split seed: [timestamp per sample_id]} in the os.listdir order test.py used, verified against the
    ground-truth y/z in the results. Splits that do not verify are omitted. Also returns {ts: group frames}."""
    from data_processing.GripFusionNet_split import group_triplets_by_pose, _assign_groups_to_ratios
    from data_processing.preprocess_data import _hub_frame_pose
    from utils.GripFusionNet_utils import get_triplets

    tm = _transf_matrix()
    groups = group_triplets_by_pose(get_triplets(data_root), data_root)
    group_of = {t[0]: gi for gi, g in enumerate(groups) for t in g}
    scratch_root = REPO_ROOT / "runs" / "candidate_analysis"
    index = {}
    for seed in sorted(ps["split"].unique()):
        _, _, test = _assign_groups_to_ratios(groups, *ratios, random.Random(int(seed)))
        stamps = [t[0] for t in test]
        scratch = scratch_root / f"_order_scratch_{seed}"
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            for ts in stamps:
                (scratch / f"{ts}.pt").touch()
            order = [f[:-3] for f in os.listdir(scratch) if f.endswith(".pt")]
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        run = ps.loc[ps["split"] == seed, "run_name"].iloc[0]
        g = ps[ps["run_name"] == run].sort_values("sample_id")
        if len(g) != len(order):
            print(f"Split {seed}: results have {len(g)} samples but the reconstructed test set has {len(order)}; skipped")
            continue
        poses = np.array([_hub_frame_pose(os.path.join(data_root, f"{ts}.json"), tm) for ts in order])
        ok = (np.allclose(poses[:, 1], g["gt_trans_y"].to_numpy(), atol=1e-6)
              and np.allclose(poses[:, 2], g["gt_trans_z"].to_numpy(), atol=1e-6))
        print(f"Split {seed}: reconstructed test order {'verified' if ok else 'DOES NOT MATCH'} against results ground truth")
        if ok:
            index[int(seed)] = order
    return index, group_of, groups


def _pair_image(ts, data_root, height=240):
    from PIL import Image
    ims = []
    for suffix in ("", "_digit"):
        im = Image.open(os.path.join(data_root, f"{ts}{suffix}.png")).convert("RGB")
        ims.append(im.resize((int(im.width * height / im.height), height)))
    canvas = Image.new("RGB", (ims[0].width + ims[1].width + 8, height), (255, 255, 255))
    canvas.paste(ims[0], (0, 0))
    canvas.paste(ims[1], (ims[0].width + 8, 0))
    return canvas


def plot_montage(rows, out_dir, name, title, data_root, ncols=4):
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)
    for ax, (_, r) in zip(axes.flat, rows.iterrows()):
        ax.imshow(_pair_image(r["timestamp"], data_root))
        ax.axis("off")
        ax.set_title(f"split {int(r['split'])} id {int(r['sample_id'])}:  gt {r['gt_roll']:.1f}°  pred {r['pred_roll']:.1f}°  "
                     f"({r['signed_roll']:+.1f}°)\n"
                     f"trans err {r['trans_mm']:.2f} mm | y {r['gt_y_mm']:.1f} z {r['gt_z_mm']:.1f}\n"
                     f"group: {int(r['group_frames'])} frame(s), siblings' err "
                     f"{('%.2f°' % r['sibling_mean_err']) if pd.notna(r['sibling_mean_err']) else 'n/a'}",
                     fontsize=7)
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    fig.suptitle(title + "   (left: RGB, right: DIGIT)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def montage_section(ps, out_dir, data_root, top_k):
    index, group_of, groups = build_sample_index(ps, data_root)
    if not index:
        print("No split could be verified; skipping montage")
        return
    samples = rpea.per_sample_means(ps)
    samples = samples[samples["split"].isin(index)].copy()
    samples["timestamp"] = [index[int(s)][int(i)] for s, i in zip(samples["split"], samples["sample_id"])]
    trans = ps.groupby(["split", "sample_id"])["trans_mm"].mean().rename("trans_mm").reset_index()
    samples = samples.merge(trans, on=["split", "sample_id"])
    samples["pred_roll"] = samples["gt_roll"] + samples["signed_roll"]
    samples["group"] = samples["timestamp"].map(group_of)
    samples["group_frames"] = samples["group"].map(lambda gi: len(groups[gi]))
    gsum = samples.groupby("group")["abs_roll"].transform("sum")
    gn = samples.groupby("group")["abs_roll"].transform("count")
    samples["sibling_mean_err"] = np.where(gn > 1, (gsum - samples["abs_roll"]) / (gn - 1).clip(lower=1), np.nan)

    hardest = samples.nlargest(top_k, "abs_roll")
    lo, hi = samples["abs_roll"].quantile([0.45, 0.55])
    typical = samples[(samples["abs_roll"] >= lo) & (samples["abs_roll"] <= hi)].sample(top_k, random_state=0)
    plot_montage(hardest, out_dir, "montage_hardest",
                 f"{top_k} hardest test samples (highest mean |roll error| over all runs of their split)", data_root)
    plot_montage(typical, out_dir, "montage_typical",
                 f"{top_k} typical test samples for contrast (45-55th percentile of mean |roll error|)", data_root)

    top = samples.nlargest(30, "abs_roll").copy()
    top["rgb_file"] = [os.path.join(data_root, f"{t}.png") for t in top["timestamp"]]
    top["digit_file"] = [os.path.join(data_root, f"{t}_digit.png") for t in top["timestamp"]]
    top[["split", "sample_id", "timestamp", "abs_roll", "abs_roll_max", "signed_roll", "gt_roll", "pred_roll",
         "trans_mm", "gt_y_mm", "gt_z_mm", "group_frames", "sibling_mean_err", "n_runs", "rgb_file", "digit_file"]] \
        .to_csv(out_dir / "hardest_samples_with_files.csv", index=False)
    print(f"\nHardest {top_k}: {int((hardest['group_frames'] > 1).sum())} belong to multi-frame grasp groups; "
          f"siblings' mean |roll err| = {hardest['sibling_mean_err'].mean():.2f}° "
          f"(all-sample mean {samples['abs_roll'].mean():.2f}°)")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True, help="One rot_scaling_factor run folder / container dir.")
    parser.add_argument("--output_dir", default=None, help="Default: <run_dir>/overview/sample_level_analysis")
    parser.add_argument("--data_root", default=str(REPO_ROOT / "Data"), help="Folder with the raw triplets (montage only).")
    parser.add_argument("--top_k", type=int, default=12, help="Samples per montage (default 12).")
    parser.add_argument("--skip_montage", action="store_true", help="Skip the image montage (no image / split reconstruction).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.run_dir) / "overview" / "sample_level_analysis"
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ps, cands, names, colors = load(args.run_dir)
    print(f"{ps['run_name'].nunique()} runs, splits {sorted(ps['split'].unique())}")

    plot_error_heatmaps(ps, out_dir, cands, names, colors)
    plot_ecdf(ps, out_dir, cands, names, colors)
    plot_split_centered(ps, out_dir, cands, names, colors)
    plot_paired_vs_concat(ps, out_dir, cands, names, colors)
    plot_roll_swap(ps, out_dir)
    if not args.skip_montage:
        montage_section(ps, out_dir, args.data_root, args.top_k)

    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
