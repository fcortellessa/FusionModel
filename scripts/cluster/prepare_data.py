"""One-time data preparation for the compact (cluster / array-job) pipeline. Run from the repo root.

    # 1. freeze the splits used so far (never overwritten afterwards)
    python scripts/cluster/prepare_data.py freeze-splits --seeds 36 37 38 39 40 41 42 43 44 45
    # 2. build the shared uint8 cache (~4 GB, split-independent)
    python scripts/cluster/prepare_data.py build-cache
    # 3. checks
    python scripts/cluster/prepare_data.py verify-splits   # frozen splits == the splits existing runs were tested on
    python scripts/cluster/prepare_data.py verify-cache    # compact tensors == legacy .pt tensors (bit-identical)

Everything is written under the data root (FUSIONMODEL_DATA_ROOT, else <repo>/Data): splits/, compact_cache/.
Nothing here touches Data/Data_split or any run folder.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from data_processing.compact_cache import (  # noqa: E402
    HUB_TRANSF_MATRIX,
    CompactCacheDataset,
    build_compact_cache,
    split_filename,
    write_frozen_split,
)
from data_processing.preprocess_data import _hub_frame_pose  # noqa: E402
from utils.utils import DATA_ROOT_ENV, resolve_group_aware_split, resolve_split_seed  # noqa: E402

TRANS_KEYS = (("mean_y", "trans_mean_y"), ("std_y", "trans_std_y"), ("mean_z", "trans_mean_z"), ("std_z", "trans_std_z"))


def data_root_default():
    return os.path.abspath(os.environ.get(DATA_ROOT_ENV) or os.path.join(REPO_ROOT, "Data"))


# ------------------------------------------------------------------------------------------------
# existing runs
# ------------------------------------------------------------------------------------------------
def iter_runs(runs_dir):
    seen = set()
    for cfg_path in glob.glob(os.path.join(runs_dir, "**", "config.yml"), recursive=True):
        run = os.path.realpath(os.path.dirname(cfg_path))
        if run in seen or "tensorboard" in run:
            continue
        seen.add(run)
        try:
            cfg = yaml.safe_load(open(cfg_path))
        except Exception:
            continue
        if not isinstance(cfg, dict) or "data" not in cfg:
            continue
        yield run, cfg


def legacy_translation_stats(runs_dir, seed):
    """The exact y/z stats existing group-aware runs on this split trained with (from their config snapshots)."""
    found = Counter()
    for _, cfg in iter_runs(runs_dir):
        if resolve_split_seed(cfg) != seed or not resolve_group_aware_split(cfg):
            continue
        t = cfg.get("data", {}).get("translation_norm_stats")
        if t and all(k in t for k, _ in TRANS_KEYS):
            found[tuple(float(t[k]) for k, _ in TRANS_KEYS)] += 1
    if not found:
        return None, 0, 0
    values, count = found.most_common(1)[0]
    return {dst: v for (_, dst), v in zip(TRANS_KEYS, values)}, count, len(found)


def legacy_image_stats(data_root, seed):
    """Image stats are only kept on disk for the split currently materialised in Data_split."""
    manifest = os.path.join(data_root, "Data_split", "split_manifest.json")
    stats_path = os.path.join(data_root, "Data_split", "normalization_stats.json")
    if not (os.path.exists(manifest) and os.path.exists(stats_path)):
        return None
    m = json.load(open(manifest))
    if m.get("seed") != seed or not m.get("group_aware", False):
        return None
    s = json.load(open(stats_path))
    return {k: s[k] for k in ("rgb_mean", "rgb_std", "digit_mean", "digit_std")}


def _roll_deg(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


def verify_split_against_runs(data_root, runs_dir, split, max_runs=None):
    """Compare the frozen test set with the per-sample ground truth stored in existing runs' test results."""
    seed = split["split_seed"]
    poses = np.array([_hub_frame_pose(os.path.join(data_root, f"{t}.json"), HUB_TRANSF_MATRIX) for t in split["test"]])
    ref = np.stack([poses[:, 1], poses[:, 2], _roll_deg(poses[:, 3:])], 1)
    ok, bad, skipped = [], [], 0
    for run, cfg in iter_runs(runs_dir):
        if resolve_split_seed(cfg) != seed or not resolve_group_aware_split(cfg):
            continue
        if "group_aware_split" not in cfg.get("data", {}) and "group_aware_split" not in cfg:
            skipped += 1  # predates group-aware splitting: trained on the legacy per-frame split
            continue
        res = sorted(glob.glob(os.path.join(run, "results", "results_*.csv")))
        if not res:
            continue
        import pandas as pd

        df = pd.read_csv(res[-1])
        if not {"gt_trans_y", "gt_trans_z"} <= set(df.columns):
            continue  # results file from an older test.py without per-sample ground truth
        has_roll = "gt_roll_deg" in df.columns  # older result files lack it: compare y/z only
        cols = ["gt_trans_y", "gt_trans_z"] + (["gt_roll_deg"] if has_roll else [])
        got = df[cols].to_numpy(dtype=np.float64)
        r = ref if has_roll else ref[:, :2]
        got = got[np.lexsort(got.T[::-1])]
        r = r[np.lexsort(r.T[::-1])]
        if len(got) != len(r):
            bad.append((run, f"test size {len(got)} != frozen {len(r)}"))
            continue
        dy = np.abs(got[:, :2] - r[:, :2]).max()
        droll = np.abs(((got[:, 2] - r[:, 2]) + 180) % 360 - 180).max() if has_roll else 0.0
        (ok if dy < 1e-6 and droll < 1e-3 else bad).append(
            (run, f"max |Δy,z|={dy:.2e} m" + (f", max |Δroll|={droll:.2e} deg" if has_roll else " (no roll column)")))
        if max_runs and len(ok) + len(bad) >= max_runs:
            break
    return ok, bad, skipped


# ------------------------------------------------------------------------------------------------
# commands
# ------------------------------------------------------------------------------------------------
def cmd_freeze(args):
    out_dir = args.out_dir or os.path.join(args.data_root, "splits")
    for seed in args.seeds:
        path = os.path.join(out_dir, split_filename(seed, True))
        if os.path.exists(path) and not args.overwrite:
            print(f"[skip] {path} already frozen")
            continue
        override, source = {}, {}
        trans, n_runs, n_variants = legacy_translation_stats(args.runs_dir, seed)
        if trans:
            override.update(trans)
            source.update({k: f"copied from {n_runs} existing run config(s)" + (f" ({n_variants} variants seen!)" if n_variants > 1 else "")
                           for k in trans})
        img = legacy_image_stats(args.data_root, seed)
        if img:
            override.update(img)
            source.update({k: "copied from Data/Data_split/normalization_stats.json (legacy cache of this split)" for k in img})
        path, payload = write_frozen_split(args.data_root, out_dir, seed, group_aware=True, stats_override=override,
                                           stats_source=source, overwrite=args.overwrite)
        n = {k: len(payload[k]) for k in ("train", "val", "test")}
        print(f"[frozen] seed {seed}: {n} -> {path}")
        print(f"         translation stats: {'legacy runs' if trans else 'recomputed'}; image stats: {'legacy cache' if img else 'recomputed'}")
        ok, bad, skipped = verify_split_against_runs(args.data_root, args.runs_dir, payload)
        print(f"         check vs existing runs: {len(ok)} match, {len(bad)} mismatch, {skipped} pre-group-aware skipped")
        for run, msg in bad[:5]:
            print(f"           MISMATCH {os.path.relpath(run, REPO_ROOT)}: {msg}")


def cmd_verify_splits(args):
    split_dir = args.out_dir or os.path.join(args.data_root, "splits")
    total_bad = 0
    for path in sorted(glob.glob(os.path.join(split_dir, "split_*.json"))):
        split = json.load(open(path))
        ok, bad, skipped = verify_split_against_runs(args.data_root, args.runs_dir, split)
        total_bad += len(bad)
        print(f"{os.path.basename(path)}: {len(ok)} runs match, {len(bad)} mismatch, "
              f"{skipped} pre-group-aware runs skipped")
        for run, msg in bad[:5]:
            print(f"   MISMATCH {os.path.relpath(run, REPO_ROOT)}: {msg}")
    sys.exit(1 if total_bad else 0)


def cmd_build_cache(args):
    out_dir = args.cache_dir or os.path.join(args.data_root, "compact_cache")
    meta = build_compact_cache(args.data_root, out_dir, workers=args.workers, overwrite=args.overwrite)
    size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1e9
    print(f"[built] {meta['num_samples']} samples, boxes {meta['boxes']}, {size:.2f} GB -> {out_dir}")


def cmd_verify_cache(args):
    """Bit-compare compact tensors with the legacy .pt cache of the split currently in Data/Data_split,
    using that legacy cache's own stats (isolates the pipeline from any stats recomputation)."""
    split_root = os.path.join(args.data_root, "Data_split")
    manifest = json.load(open(os.path.join(split_root, "split_manifest.json")))
    legacy = json.load(open(os.path.join(split_root, "normalization_stats.json")))
    cache_dir = args.cache_dir or os.path.join(args.data_root, "compact_cache")
    rng = np.random.default_rng(0)
    worst, n_checked, n_identical = 0.0, 0, 0
    for name in ("test", "val", "train"):
        ts = manifest["splits"][name]
        if name == "train" and args.n_train is not None:
            ts = list(rng.choice(ts, size=min(args.n_train, len(ts)), replace=False))
        ds = CompactCacheDataset(cache_dir, ts, legacy)
        for i, t in enumerate(ts):
            ref = torch.load(os.path.join(split_root, name, f"{name}_preprocessing", f"{t}.pt"))
            rgb, digit, pose = ds[i]
            same = torch.equal(rgb, ref["rgb"]) and torch.equal(digit, ref["digit"]) and torch.equal(pose, ref["pose"])
            n_identical += same
            n_checked += 1
            worst = max(worst, float((rgb - ref["rgb"]).abs().max()), float((digit - ref["digit"]).abs().max()),
                        float((pose - ref["pose"]).abs().max()))
        print(f"{name}: checked {len(ts)}")
    print(f"split seed {manifest['seed']}: {n_identical}/{n_checked} samples bit-identical, max |diff| = {worst:.3g}")
    sys.exit(0 if n_identical == n_checked else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=data_root_default())
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze-splits")
    f.add_argument("--seeds", type=int, nargs="+", required=True)
    f.add_argument("--runs_dir", default=os.path.join(REPO_ROOT, "runs"))
    f.add_argument("--out_dir", default=None)
    f.add_argument("--overwrite", action="store_true")
    v = sub.add_parser("verify-splits")
    v.add_argument("--runs_dir", default=os.path.join(REPO_ROOT, "runs"))
    v.add_argument("--out_dir", default=None)
    b = sub.add_parser("build-cache")
    b.add_argument("--cache_dir", default=None)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("verify-cache")
    c.add_argument("--cache_dir", default=None)
    c.add_argument("--n_train", type=int, default=300, help="train samples to check (val+test are checked fully)")
    args = ap.parse_args()
    {"freeze-splits": cmd_freeze, "verify-splits": cmd_verify_splits, "build-cache": cmd_build_cache,
     "verify-cache": cmd_verify_cache}[args.cmd](args)


if __name__ == "__main__":
    main()
