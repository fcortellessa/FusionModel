#!/usr/bin/env python3
"""
RGB-only vs tactile-only vs RGB+tactile fusion, paired by split seed.

Reads the CSV written by scripts/evaluate/input_ablation_test.py (condition == "normal") and
answers: for the *same* train/val/test split, is the fusion model actually better than the
RGB-only model?

Why paired: every run has its own split seed, so each seed has a different (small) test set;
absolute errors differ between seeds far more than between models. Comparing model means over
different seeds mixes that split noise in. Here, for each split seed present in both groups,
we take (fusion - single-stream) and summarise those differences.

Groups are (streams, fusion, pre_head_channel_divisor). If a group has several runs for the same
split seed they are averaged first. Lower is better for every metric; a negative difference
means fusion is better.

Usage:
    python scripts/evaluate/modality_comparison.py --csv results/input_ablation_test.csv
    python scripts/evaluate/modality_comparison.py --csv results/input_ablation_test.csv \
        --noise_deg 0.1
"""
import argparse
import sys

import numpy as np
import pandas as pd

METRICS = [("val_abs_rot_x", "rot_x [deg]", 1.0), ("val_abs_y", "y [mm]", 1e3), ("val_abs_z", "z [mm]", 1e3)]


def group_name(r):
    div = r.get("pre_head_channel_divisor")
    div = "" if pd.isna(div) else f" div{int(div)}"
    return f"{r['streams']} | {r['fusion']}{div}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--noise_deg", type=float, default=None,
                    help="optional rot_x noise floor (deg) to compare mean differences against")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    normal = df[df.condition == "normal"].copy()
    normal["group"] = normal.apply(group_name, axis=1)
    const = df[df.condition == "const_pose"].copy()
    for col, _, _ in METRICS:
        normal[col] = normal[col].astype(float)

    per = normal.groupby(["group", "streams", "split_seed"])[[m for m, _, _ in METRICS]].mean().reset_index()
    print("=== per split seed (mean over runs in the group) ===")
    for col, label, scale in METRICS:
        t = per.pivot_table(index="split_seed", columns="group", values=col) * scale
        c = const.groupby("split_seed")[col].mean() * scale
        t["const_pose (oracle)"] = c
        print(f"\n{label}")
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print(t.round(3).to_string())

    print("\n=== paired difference vs single-stream baselines (fusion - baseline; negative = fusion better) ===")
    single = {g: s for g, s in per[["group", "streams"]].drop_duplicates().itertuples(index=False)
              if "+" not in s}
    fusion = {g for g, s in per[["group", "streams"]].drop_duplicates().itertuples(index=False) if "+" in s}
    if not single or not fusion:
        print("need at least one single-stream group and one two-stream group in the CSV")
        return
    rows = []
    for fg in sorted(fusion):
        for sg in sorted(single):
            a_ = per[per.group == fg].set_index("split_seed")
            b_ = per[per.group == sg].set_index("split_seed")
            seeds = sorted(set(a_.index) & set(b_.index))
            if not seeds:
                continue
            for col, label, scale in METRICS:
                d = (a_.loc[seeds, col] - b_.loc[seeds, col]) * scale
                rows.append(dict(fusion=fg, baseline=sg, metric=label, n_seeds=len(seeds),
                                 mean_diff=d.mean(), std_diff=d.std(ddof=1) if len(d) > 1 else np.nan,
                                 min_diff=d.min(), max_diff=d.max(),
                                 fusion_wins=f"{int((d < 0).sum())}/{len(d)}",
                                 seeds=",".join(str(s) for s in seeds)))
    if not rows:
        print("no split seed is shared between a fusion group and a single-stream group")
        return
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 60):
        print(out.round(3).to_string(index=False))
    if a.noise_deg is not None:
        r = out[out.metric.str.startswith("rot_x")]
        print(f"\nrot_x differences within +-{a.noise_deg} deg are inside the stated seed noise:")
        for _, x in r.iterrows():
            print(f"  {x.fusion}  vs  {x.baseline}: mean {x.mean_diff:+.3f} deg "
                  f"-> {'within noise' if abs(x.mean_diff) <= a.noise_deg else 'outside noise'}")
    print("\nNote: with 4-5 seeds a mean difference is only suggestive; look at min/max_diff and "
          "fusion_wins -- a consistent sign across seeds is stronger evidence than the mean.")


if __name__ == "__main__":
    sys.exit(main())
