"""Scatter + summary stats for the per-head checkpoint: both streams vs RGB only vs tactile only.

For one fusion method (default ``concat``) the runs are split into three arms by
``model.streams`` (RGB + tactile / RGB only / tactile only) and the *per-head*
checkpoint (``checkpoint_best_per_head.pth``) is compared on the val and test split:

    left  panel: rotation error about x (deg)  vs  translation L2 error (mm)
    right panel: translation y error (mm)      vs  translation z error (mm)

One point = one run (mean over that run's samples), labelled with its split seed; the large marker
is the arm's mean over runs (split seeds) with +-1 std error bars. Colours come from
scripts/evaluate/fusion_colors.py (two-stream arm = the fusion method's colour, single-stream = greys). Printed/saved table, per arm / split / metric, over all samples of
all runs of that arm pooled: mean +- std, median, worst (max), plus the std of the per-run means
(seed-to-seed spread).

Inputs per run (discovered by scanning ``--run_dir`` roots for config.yml):
    test: <run>/results/per_head/results_*.csv        (written by scripts/test.py)
    val : <run>/results_val_per_head/results_val.csv  (written by infer_split_gpu.py --checkpoint
          checkpoint_best_per_head.pth; with --compute_val this script runs it for missing runs.
          NOTE: that rebuilds Data/Data_split for each run's split seed, like test.py does.)

Translation L2 = sqrt(err_y^2 + err_z^2) (``--trans_metric l2_xyz`` for the csv's val_abs_trans).
x is never predicted and its ~1.4 mm calibration residual would dominate the xyz norm.

  python scripts/evaluate/scatter_per_stream_per_head.py --fusion concat --run_dir runs
  python scripts/evaluate/scatter_per_stream_per_head.py --fusion se_channel_gated_concat \\
      --run_dir runs/final_per_stream --compute_val
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_colors import color_for  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Arm label, colour key in fusion_colors.COLORS, marker. The two-stream arm takes the colour of the
# fusion method under test (set in main(); concat -> dark charcoal); single-stream arms are the greys.
# Marker shapes stay different per arm as a secondary encoding.
ARMS = {
    "both": ["RGB + tactile", "concat", "o"],
    "rgb": ["RGB only", "rgb_only", "s"],
    "tactile": ["Tactile only", "digit_only", "^"],
}


def arm_of(streams) -> str | None:
    s = set(streams)
    if s == {"rgb_image", "digit_image"}:
        return "both"
    if s == {"rgb_image"}:
        return "rgb"
    if s == {"digit_image"}:
        return "tactile"
    return None


def latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def discover(roots, fusion, divisor):
    runs = {}
    for root in roots:
        for cfg_path in Path(root).rglob("config.yml"):
            if "tensorboard" in cfg_path.parts or "_cleanup_overview" in cfg_path.parts:
                continue
            run = cfg_path.parent
            try:
                cfg = yaml.safe_load(cfg_path.read_text())
                m, t, d = cfg["model"], cfg["training"], cfg["data"]
            except Exception:
                continue
            arm = arm_of(m.get("streams", []))
            if m.get("fusion") != fusion or arm is None or m.get("dual_fusion"):
                continue
            if divisor is not None and m.get("pre_head_channel_divisor") != divisor:
                continue
            test_csv = latest(str(run / "results" / "per_head" / "results_*.csv"))
            if test_csv is None:
                continue
            split_seed = d.get("split_seed", t.get("seed"))
            key = (arm, split_seed, t.get("seed"))
            # same arm+seeds trained more than once: keep the newest run folder
            if key not in runs or run.name > runs[key]["run"].name:
                runs[key] = dict(run=run, arm=arm, split_seed=split_seed, test_csv=test_csv)
    return list(runs.values())


def load_errors(path: str, trans_metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    trans = df["val_abs_trans"] if trans_metric == "l2_xyz" else np.hypot(df["val_abs_y"], df["val_abs_z"])
    return pd.DataFrame({
        "rot_x_deg": df["val_abs_rot_x"],
        "trans_mm": trans * 1e3,
        "y_mm": df["val_abs_y"] * 1e3,
        "z_mm": df["val_abs_z"] * 1e3,
    })


METRICS = [("rot_x_deg", "Rot x (deg)"), ("trans_mm", "Trans L2 (mm)"), ("y_mm", "Trans y (mm)"), ("z_mm", "Trans z (mm)")]


def summarise(samples: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    """samples[(arm, split, run_id)] -> per-sample errors. Returns one row per arm/split/metric."""
    rows = []
    for arm in ARMS:
        for split in ("val", "test"):
            frames = {k: v for k, v in samples.items() if k[0] == arm and k[1] == split}
            if not frames:
                continue
            pooled = pd.concat(frames.values(), ignore_index=True)
            for col, label in METRICS:
                per_run = np.array([f[col].mean() for f in frames.values()])
                rows.append(dict(
                    arm=ARMS[arm][0], split=split, metric=label, n_runs=len(frames), n_samples=len(pooled),
                    mean=pooled[col].mean(), std=pooled[col].std(), median=pooled[col].median(),
                    worst=pooled[col].max(), run_mean_std=per_run.std(ddof=1) if len(per_run) > 1 else np.nan,
                ))
    return pd.DataFrame(rows)


def plot(samples, seed_of, out_base: Path, title_suffix: str = ""):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
    panels = [("rot_x_deg", "trans_mm", "Rot x abs error (deg)", "Trans L2 abs error (mm)"),
              ("y_mm", "z_mm", "Trans y abs error (mm)", "Trans z abs error (mm)")]
    for r, split in enumerate(("val", "test")):
        for c, (xc, yc, xl, yl) in enumerate(panels):
            ax = axes[r, c]
            for arm, (label, color_key, marker) in ARMS.items():
                color = color_for(color_key)
                runs = [(k[2], f[xc].mean(), f[yc].mean()) for k, f in samples.items() if k[0] == arm and k[1] == split]
                if not runs:
                    continue
                pts = np.array([[x, y] for _, x, y in runs])
                ax.scatter(pts[:, 0], pts[:, 1], s=95, color=color, marker=marker, alpha=0.9,
                           edgecolor="black", linewidth=0.6, label=f"{label} (n={len(pts)})", zorder=3)
                for rid, x, y in runs:   # label = split seed, as in the final_ablation_candidates scatter
                    ax.annotate(str(seed_of[rid]), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, zorder=4)
                mu = pts.mean(0)
                sd = pts.std(0, ddof=1) if len(pts) > 1 else np.zeros(2)
                ax.errorbar(mu[0], mu[1], xerr=sd[0], yerr=sd[1], color=color, marker=marker, markersize=15,
                            markeredgecolor="black", markeredgewidth=1.4, capsize=5, linewidth=2.2, zorder=5)
            ax.set_title(f"{split.capitalize()} set: {yl.split(' abs')[0]} vs {xl.split(' abs')[0]}", fontsize=11, loc="left")
            ax.set_xlabel(f"{split.capitalize()} {xl}")
            ax.set_ylabel(f"{split.capitalize()} {yl}")
            ax.grid(linestyle="--", alpha=0.45, linewidth=0.7)
            if r == 0 and c == 0:
                ax.legend(frameon=True, fontsize=9, title="stream arm")
    fig.suptitle("Per-head checkpoint, mean over samples per run (labels = split seed; "
                 f"large marker = mean +/- std over split seeds){title_suffix}", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", nargs="+", default=["runs"])
    ap.add_argument("--fusion", default="concat")
    ap.add_argument("--divisor", type=int, default=None, help="require model.pre_head_channel_divisor == this")
    ap.add_argument("--trans_metric", choices=["l2_yz", "l2_xyz"], default="l2_yz")
    ap.add_argument("--all_seeds", action="store_true",
                    help="default keeps only split seeds present in every arm that has runs")
    ap.add_argument("--compute_val", action="store_true", help="run per-head inference on the val split if missing")
    ap.add_argument("--output_dir", default="runs/scatter_per_stream_per_head")
    a = ap.parse_args()

    runs = discover([REPO / r if not Path(r).is_absolute() else Path(r) for r in a.run_dir], a.fusion, a.divisor)
    if not runs:
        sys.exit(f"No runs with fusion={a.fusion} and per-head test results found under {a.run_dir}")
    by_arm = {arm: sorted({r["split_seed"] for r in runs if r["arm"] == arm}) for arm in ARMS}
    print(f"fusion={a.fusion}  split seeds per arm: " + "; ".join(f"{ARMS[k][0]}: {v or 'NO RUNS'}" for k, v in by_arm.items()))
    present = [v for v in by_arm.values() if v]
    if not a.all_seeds and len(present) > 1:
        common = set.intersection(*map(set, present))
        runs = [r for r in runs if r["split_seed"] in common]
        print(f"keeping split seeds common to all arms: {sorted(common)}")

    if a.compute_val:
        todo = [r for r in runs if not (r["run"] / "results_val_per_head" / "results_val.csv").exists()]
        if todo:
            cmd = [sys.executable, str(REPO / "scripts/evaluate/infer_split_gpu.py"), "--split", "val",
                   "--checkpoint", "checkpoint_best_per_head.pth", "--runs"] + [str(r["run"]) for r in todo]
            subprocess.run(cmd, check=True, cwd=REPO)

    ARMS["both"][1] = a.fusion  # two-stream arm uses the colour of the fusion method under test
    ARMS["both"][0] = f"{a.fusion} (RGB + tactile)" if a.fusion != "concat" else "Concat (RGB + tactile)"
    samples, seed_of, missing_val = {}, {}, []
    for r in runs:
        rid = f"{r['run'].parent.name}/{r['run'].name}"
        seed_of[rid] = r["split_seed"]
        samples[(r["arm"], "test", rid)] = load_errors(r["test_csv"], a.trans_metric)
        val_csv = r["run"] / "results_val_per_head" / "results_val.csv"
        if val_csv.exists():
            samples[(r["arm"], "val", rid)] = load_errors(str(val_csv), a.trans_metric)
        else:
            missing_val.append(rid)
    if missing_val:
        print(f"\n{len(missing_val)} runs have no per-head val results (use --compute_val): val omitted for them")

    out = Path(a.output_dir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    table = summarise(samples)
    table.to_csv(out / f"summary_{a.fusion}.csv", index=False)
    plot(samples, seed_of, out / f"scatter_{a.fusion}")

    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.4f}")
    for split in ("val", "test"):
        for _col, label in METRICS:
            sub = table[(table.split == split) & (table.metric == label)].drop(columns=["split", "metric"])
            if sub.empty:
                continue
            print(f"\n=== {split.upper()}  {label}  (pooled over samples; run_mean_std = std of per-run means) ===")
            print(sub.to_string(index=False))
    print(f"\nWrote {out}/scatter_{a.fusion}.png|svg and summary_{a.fusion}.csv")


if __name__ == "__main__":
    main()
