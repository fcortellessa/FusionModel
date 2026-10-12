#!/usr/bin/env python3
"""
Does a trained model actually use each input stream? -- input-level ablation.

For every run (checkpoint_best.pth) the chosen split is evaluated under these conditions:

  normal        the model as trained
  zero_digit    tactile image replaced by an all-zero tensor. Images are normalised with the
                train mean/std, so zero == the mean tactile image (no contact information)
  shuf_digit    tactile image replaced by a *different* sample's tactile image (cyclic shift of
                N//2 over the split): keeps the input distribution, breaks the link to its own
                RGB frame and pose
  zero_rgb      same as zero_digit, for the RGB stream
  shuf_rgb      same as shuf_digit, for the RGB stream
  const_pose    reference, not a model: predict the split's own median y / z / roll for every
                sample (an *oracle* constant predictor: the best any constant can do). A model
                that does not clearly beat this has not learnt the target.

Only conditions on streams the run actually has are evaluated (a tactile-only run gets
normal / zero_digit / shuf_digit; an RGB-only run gets normal / zero_rgb / shuf_rgb).

How to read it (differences vs `normal`, compared against seed-to-seed noise, ~+-0.1 deg rot):
  * zero_digit ~= shuf_digit ~= normal      -> the model ignores tactile.
  * shuf_digit worse than zero_digit        -> it uses tactile and is hurt by wrong tactile.
  * a stream's removal blows the error up   -> that stream carries the signal.
For the single-stream runs the shuf_* rows are a control: if shuffling the only input does not
degrade a model, that model is not using its input.

Usage:
    python scripts/evaluate/input_ablation_test.py --split test \
        --sweeps runs/se_gca runs/pre_head_channel_divisor_vs_se_channel
    python scripts/evaluate/input_ablation_test.py --split val --runs runs/<sweep>/<run_dir> [...]
    python scripts/evaluate/input_ablation_test.py --split test --sweeps runs/se_gca --list

IMPORTANT -- shared data folder. Like scripts/test.py, evaluating a run needs Data/Data_split to
be the split with that run's split seed, and switching split seed *rebuilds* Data/Data_split
(train, val and test). That would corrupt a training job that is reading it. So by default this
script only evaluates runs whose split already matches what is on disk and SKIPS the rest
(printing why). Pass --allow_rebuild only when no training / testing job is running; runs are
then grouped by split seed so each split is rebuilt at most once.

Output: results/input_ablation_<split>.csv (one row per run x condition). Feed it to
scripts/evaluate/modality_comparison.py for the paired RGB-only / tactile-only / fusion table.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_split_gpu as I  # noqa: E402  (same folder; provides REPO, build_model, errors_frame)

sys.path.insert(0, I.REPO)
from data_processing.dataloader_preprocessed import PreprocessedTactileDataset, tactile_collate_fn  # noqa: E402
from data_processing.GripFusionNet_split import split_manifest_matches  # noqa: E402
from data_processing.preprocess_data import ensure_test_split_current  # noqa: E402
from utils.utils import (get_optimizer, load_checkpoint, load_config, resolve_data_root,  # noqa: E402
                         resolve_group_aware_split, resolve_split_seed)

RGB, DIGIT = "rgb_image", "digit_image"
METRICS = ["val_abs_rot_x", "val_abs_rot_z", "val_abs_y", "val_abs_z", "val_abs_trans"]


class ShiftedView(Dataset):
    """ds[(i + shift) % N]: pairs every sample with a different one, deterministically."""

    def __init__(self, ds, shift):
        self.ds, self.shift = ds, shift

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[(i + self.shift) % len(self.ds)]


def split_on_disk_matches(cfg, data_root):
    split_root = os.path.join(data_root, "Data_split")
    ok = split_manifest_matches(data_root, split_root, seed=resolve_split_seed(cfg),
                                group_aware=resolve_group_aware_split(cfg))
    return ok and os.path.isdir(os.path.join(split_root, "test", "test_preprocessing"))


def summarise(df, cond, run_meta):
    row = dict(run_meta, condition=cond, n=len(df))
    for m in METRICS:
        row[m] = float(df[m].mean()) if m in df else np.nan
    return row


def const_pose_frame(gt_df):
    """Oracle constant predictor on the evaluated split: median y/z/roll of its own ground truth."""
    out = pd.DataFrame(index=gt_df.index)
    out["val_abs_y"] = (gt_df["gt_trans_y"] - gt_df["gt_trans_y"].median()).abs()
    out["val_abs_z"] = (gt_df["gt_trans_z"] - gt_df["gt_trans_z"].median()).abs()
    d = np.deg2rad(gt_df["gt_roll_deg"].to_numpy() - np.median(gt_df["gt_roll_deg"].to_numpy()))
    out["val_abs_rot_x"] = np.abs(np.degrees((d + np.pi) % (2 * np.pi) - np.pi))
    return out


def sanity_check_vs_test_py(run_dir, split, normal_df, tol_deg=1e-3, tol_m=1e-6):
    """Compare our `normal` condition with the run's own scripts/test.py output, if there is one.

    The checkpoint, split and error formulas are the same, so the per-sample errors must agree to
    float precision. This confirms that THIS script's data loading / model construction /
    checkpoint loading reproduce the official evaluation, so the ablation rows (which only change
    the inputs) can be trusted. Only possible for split == test and runs that have a results CSV.
    """
    out = {"sanity": "no_reference"}
    ref_files = sorted(glob.glob(os.path.join(run_dir, "results", "results_2*.csv")))
    if split != "test" or not ref_files:
        return out
    ref = pd.read_csv(ref_files[-1])
    if len(ref) != len(normal_df):
        return {"sanity": f"MISMATCH n={len(normal_df)} vs ref n={len(ref)}"}
    diffs = {}
    for col, tol in (("val_abs_rot_x", tol_deg), ("val_abs_y", tol_m), ("val_abs_z", tol_m)):
        # order-insensitive: sort the per-sample errors before comparing
        d = np.abs(np.sort(normal_df[col].to_numpy()) - np.sort(ref[col].to_numpy())).max()
        diffs[col] = (float(d), d <= tol)
    ok = all(v[1] for v in diffs.values())
    out.update(sanity="OK" if ok else "MISMATCH",
               sanity_max_diff_rot_x=diffs["val_abs_rot_x"][0],
               sanity_max_diff_y=diffs["val_abs_y"][0], sanity_max_diff_z=diffs["val_abs_z"][0],
               sanity_ref_file=os.path.basename(ref_files[-1]))
    return out


def run_one(run_dir, split, device, batch_size):
    cfg = load_config(os.path.join(run_dir, "config.yml"))
    streams = list(cfg["model"]["streams"])
    data_root = resolve_data_root(cfg, I.REPO)
    ensure_test_split_current(cfg, data_root)  # no-op when the split on disk already matches
    folder = os.path.join(data_root, "Data_split", split, f"{split}_preprocessing")
    ds = PreprocessedTactileDataset(folder)
    kw = dict(batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=tactile_collate_fn)
    loader = DataLoader(ds, **kw)
    loader_shift = DataLoader(ShiftedView(ds, max(1, len(ds) // 2)), **kw)

    model = I.build_model(cfg)
    opt = get_optimizer(model_or_params=model, optimizer_name=cfg["training"]["optimizer"], lr=cfg["training"]["lr"])
    load_checkpoint(model, opt, os.path.join(run_dir, "checkpoint_best.pth"), device="cpu", load_optimizer=False)
    model.to(device).eval()

    conds = ["normal"]
    if DIGIT in streams:
        conds += ["zero_digit", "shuf_digit"]
    if RGB in streams:
        conds += ["zero_rgb", "shuf_rgb"]
    preds = {c: [] for c in conds}
    gts = []
    with torch.no_grad():
        for sample, other in zip(loader, loader_shift):
            x = {k: v.to(device) for k, v in sample.items() if k != "pose"}
            o = {k: v.to(device) for k, v in other.items() if k != "pose"}
            gts.append(sample["pose"].numpy())
            for c in conds:
                xi = dict(x)
                if c == "zero_digit":
                    xi[DIGIT] = torch.zeros_like(x[DIGIT])
                elif c == "shuf_digit":
                    xi[DIGIT] = o[DIGIT]
                elif c == "zero_rgb":
                    xi[RGB] = torch.zeros_like(x[RGB])
                elif c == "shuf_rgb":
                    xi[RGB] = o[RGB]
                out, _ = model(xi)
                preds[c].append(out["pose"].cpu().numpy())
    gt = np.concatenate(gts)

    meta = dict(run=os.path.basename(run_dir), sweep=os.path.basename(os.path.dirname(run_dir)), split=split,
                split_seed=resolve_split_seed(cfg), train_seed=cfg["training"].get("seed"),
                streams="+".join(s.replace("_image", "") for s in streams), fusion=cfg["model"]["fusion"],
                pre_head_channel_divisor=cfg["model"].get("pre_head_channel_divisor"))
    rows, frames = [], {}
    for c in conds:
        frames[c] = I.errors_frame(np.concatenate(preds[c]), gt, cfg)
        rows.append(summarise(frames[c], c, meta))
    rows.append(summarise(const_pose_frame(frames["normal"]), "const_pose", meta))
    rows[0].update(sanity_check_vs_test_py(run_dir, split, frames["normal"]))
    return rows


def collect_runs(args):
    runs = [os.path.abspath(r) for r in args.runs]
    for s in args.sweeps:
        runs += [os.path.abspath(d) for d in sorted(glob.glob(os.path.join(s, "2*"))) if os.path.isdir(d)]
    keep = []
    for r in runs:
        if not os.path.exists(os.path.join(r, "config.yml")) or not os.path.exists(os.path.join(r, "checkpoint_best.pth")):
            print(f"[skip] {os.path.relpath(r, I.REPO)}: no config.yml / checkpoint_best.pth (unfinished run?)")
            continue
        idle_min = (time.time() - max(os.path.getmtime(os.path.join(r, f))
                                      for f in ("metrics.csv", "checkpoint_best.pth")
                                      if os.path.exists(os.path.join(r, f)))) / 60.0
        if idle_min < args.min_idle_minutes:
            print(f"[skip] {os.path.relpath(r, I.REPO)}: modified {idle_min:.0f} min ago -- still training? "
                  f"(its checkpoint_best.pth is a partial model; --min_idle_minutes 0 to override)")
            continue
        keep.append(r)
    return keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--sweeps", nargs="*", default=[])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--allow_rebuild", action="store_true",
                    help="rebuild Data/Data_split when a run's split seed differs from the one on disk "
                         "(only when no training/testing job is using it)")
    ap.add_argument("--min_idle_minutes", type=float, default=20.0,
                    help="skip runs whose metrics.csv/checkpoint changed more recently than this (still training)")
    ap.add_argument("--split_seeds", nargs="*", type=int, default=None,
                    help="only evaluate runs whose split seed is in this list")
    ap.add_argument("--list", action="store_true", help="only list runs, split seeds and whether they can run now")
    ap.add_argument("--output_csv", default=None)
    a = ap.parse_args()

    runs = collect_runs(a)
    info = []
    for r in runs:
        cfg = load_config(os.path.join(r, "config.yml"))
        info.append((r, cfg, resolve_split_seed(cfg), split_on_disk_matches(cfg, resolve_data_root(cfg, I.REPO))))
    if a.split_seeds:
        info = [t for t in info if t[2] in set(a.split_seeds)]
    info.sort(key=lambda t: (str(t[2]), t[0]))
    print(f"{len(info)} runs with checkpoints")
    if a.list:
        for r, cfg, ss, ok in info:
            print(f"  split_seed={ss}  streams={cfg['model']['streams']}  fusion={cfg['model']['fusion']}  "
                  f"{'matches split on disk' if ok else 'needs split rebuild'}  {os.path.relpath(r, I.REPO)}")
        sys.exit(0)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_csv = a.output_csv or os.path.join(I.REPO, "results", f"input_ablation_{a.split}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    all_rows, skipped = [], []
    for i, (r, cfg, ss, ok) in enumerate(info, 1):
        if not ok and not a.allow_rebuild:
            skipped.append((r, ss))
            print(f"[{i}/{len(info)}] SKIP split_seed={ss} (differs from Data_split on disk; "
                  f"use --allow_rebuild when nothing is training): {os.path.relpath(r, I.REPO)}", flush=True)
            continue
        t0 = time.time()
        rows = run_one(r, a.split, dev, a.batch_size)
        all_rows += rows
        n = next(x for x in rows if x["condition"] == "normal")
        print(f"[{i}/{len(info)}] split_seed={ss} {rows[0]['streams']:<10} rot_x normal={n['val_abs_rot_x']:.3f}  "
              + "  ".join(f"{x['condition']}={x['val_abs_rot_x']:.3f}" for x in rows if x["condition"] != "normal")
              + f"  sanity={n.get('sanity')}  ({time.time() - t0:.0f}s)  {os.path.basename(r)}", flush=True)
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)  # incremental: survives an interruption

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}\n")
        cols = ["val_abs_rot_x", "val_abs_y", "val_abs_z"]
        g = df.groupby(["streams", "fusion", "condition"])[cols].agg(["mean", "std", "count"]).round(4)
        with pd.option_context("display.width", 250, "display.max_rows", 200):
            print(g)
    if skipped:
        print(f"\n{len(skipped)} runs skipped (split rebuild needed); split seeds: "
              f"{sorted({str(s) for _, s in skipped})}")
