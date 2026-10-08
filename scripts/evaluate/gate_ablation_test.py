#!/usr/bin/env python3
"""
Do the fusion gates do per-sample work? -- mean-gate / shuffled-gate ablation.

For a trained gated-concat-family model (gated_concat, residual_gated_concat, channelwise_*,
se_channel_*, ..., i.e. every fusion whose output is concat_s(gate_s * feature_s) or
concat_s((1 + gate_s) * feature_s) with gate weights of shape (B, streams) or (B, streams, dim)),
evaluate checkpoint_best.pth on a split four ways:

  normal      the model as trained
  recompute   gates replaced by their own values -- a self-check that the output is rebuilt
              correctly from the gates (must match `normal` to float precision)
  mean_gate   every sample gets the dataset-mean gate (per stream / per element): keeps the
              gate's average level, removes ALL per-sample variation
  shuffled    every sample gets a different sample's gate (cyclic shift inside the batch):
              keeps the gate's distribution, breaks the link to its own sample

If `mean_gate` and `shuffled` give (nearly) the same error as `normal`, the gate carries no
useful per-sample information: it is acting as a constant, i.e. a learned global scale. A gate
doing real work makes `normal` clearly better than both. Compare the differences against the
seed-to-seed noise of the metric (about +-0.1 deg rotation in the final ablation).
`gate_between_sample_std` is the direct measure of per-sample variation: the std of each gate
value across samples, averaged over gates (0 = the gate is a constant).

Usage:
    python scripts/evaluate/gate_ablation_test.py --split test --sweeps runs/final_ablation
    python scripts/evaluate/gate_ablation_test.py --split val --runs runs/<sweep>/<run_dir> [...]
Writes one row per run to --output_csv (default results/gate_ablation_<split>.csv) and prints a
per-fusion summary. Note: like scripts/test.py it rebuilds Data/Data_split for each run's split
seed, so do not run it while a training job is reading that folder.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_split_gpu as I  # noqa: E402  (same folder; provides REPO, build_model, errors_frame)

sys.path.insert(0, I.REPO)
from data_processing.dataloader_preprocessed import PreprocessedTactileDataset, tactile_collate_fn  # noqa: E402
from data_processing.preprocess_data import ensure_test_split_current  # noqa: E402
from utils.utils import get_optimizer, load_checkpoint, load_config, resolve_data_root, resolve_split_seed  # noqa: E402


class GateSwap:
    """Forward hook on a fusion module: records gate statistics and can rebuild the output from
    modified gates. mode: 'observe' (no change), 'recompute', 'mean_gate', 'shuffled'."""

    def __init__(self, residual):
        self.residual = residual
        self.mode = "observe"
        self.mean_gate = None       # (S,) or (S, dim), set between passes
        self.reset_stats()

    def reset_stats(self):
        self.n = 0
        self.sum = None
        self.sumsq = None

    def _rebuild(self, features_list, w):
        # w: (B, S) or (B, S, dim)
        if w.dim() == 2:
            w = w.unsqueeze(-1)
        parts = [((1.0 + w[:, s]) if self.residual else w[:, s]) * f for s, f in enumerate(features_list)]
        return torch.cat(parts, dim=1)

    def __call__(self, module, args, output):
        features_list = args[0]
        out, w = output
        if w.dim() not in (2, 3) or (w.dim() == 3 and w.shape[-1] != features_list[0].shape[1]):
            raise RuntimeError(f"unsupported gate weight shape {tuple(w.shape)}")
        w_d = w.detach().double()
        s, ss = w_d.sum(0), (w_d ** 2).sum(0)
        self.sum = s if self.sum is None else self.sum + s
        self.sumsq = ss if self.sumsq is None else self.sumsq + ss
        self.n += w.shape[0]
        if self.mode == "observe":
            return None
        if self.mode == "recompute":
            w_new = w
        elif self.mode == "mean_gate":
            w_new = self.mean_gate.to(w.dtype).unsqueeze(0).expand_as(w)
        elif self.mode == "shuffled":
            w_new = torch.roll(w, shifts=1, dims=0) if w.shape[0] > 1 else w
        else:
            raise ValueError(self.mode)
        return self._rebuild(features_list, w_new), w

    def finish_pass(self):
        mean = self.sum / self.n
        var = (self.sumsq / self.n - mean ** 2).clamp_min(0)
        return mean.float(), var.sqrt().mean().item()   # per-gate mean, mean of per-gate across-sample std


def fusion_modules(model):
    mods = [getattr(model, n) for n in ("fusion", "fusion_rotation", "fusion_translation") if hasattr(model, n)]
    return [m for m in mods if isinstance(m, torch.nn.Module)]


def evaluate(model, loader, device, cfg):
    preds, gts = [], []
    with torch.no_grad():
        for sample in loader:
            inputs = {k: v.to(device) for k, v in sample.items() if k != "pose"}
            out, _ = model(inputs)
            preds.append(out["pose"].cpu().numpy())
            gts.append(sample["pose"].numpy())
    df = I.errors_frame(np.concatenate(preds), np.concatenate(gts), cfg)
    return df.val_abs_rot_x.mean(), df.val_abs_y.mean() * 1e3, df.val_abs_z.mean() * 1e3


def run_one(run_dir, split, device):
    cfg = load_config(os.path.join(run_dir, "config.yml"))
    fusion = cfg["model"]["fusion"]
    if "concat" not in fusion or fusion == "concat" or "scale_gated" in fusion:
        return None
    data_root = resolve_data_root(cfg, I.REPO)
    ensure_test_split_current(cfg, data_root)
    folder = os.path.join(data_root, "Data_split", split, f"{split}_preprocessing")
    loader = DataLoader(PreprocessedTactileDataset(folder), batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=tactile_collate_fn)
    model = I.build_model(cfg)
    opt = get_optimizer(model_or_params=model, optimizer_name=cfg["training"]["optimizer"], lr=cfg["training"]["lr"])
    load_checkpoint(model, opt, os.path.join(run_dir, "checkpoint_best.pth"), device="cpu", load_optimizer=False)
    model.to(device).eval()

    hooks = [GateSwap(residual="residual" in fusion) for _ in fusion_modules(model)]
    handles = [m.register_forward_hook(h) for m, h in zip(fusion_modules(model), hooks)]
    res = {}
    for h in hooks:
        h.mode = "observe"
    res["normal"] = evaluate(model, loader, device, cfg)
    stats = [h.finish_pass() for h in hooks]
    for h, (mean, _) in zip(hooks, stats):
        h.mean_gate = mean.to(device)
    between_std = float(np.mean([s for _, s in stats]))
    gate_mean = float(np.mean([m.mean().item() for m, _ in stats]))
    for mode in ("recompute", "mean_gate", "shuffled"):
        for h in hooks:
            h.mode = mode
            h.reset_stats()
        res[mode] = evaluate(model, loader, device, cfg)
    for hd in handles:
        hd.remove()

    row = {"run": os.path.basename(run_dir), "sweep": os.path.basename(os.path.dirname(run_dir)), "fusion": fusion,
           "dual_fusion": bool(cfg["model"].get("dual_fusion", False)), "split_seed": resolve_split_seed(cfg),
           "split": split, "gate_mean": gate_mean, "gate_between_sample_std": between_std}
    for mode, (rot, y, z) in res.items():
        row.update({f"{mode}_rot_x": rot, f"{mode}_y_mm": y, f"{mode}_z_mm": z})
    for mode in ("mean_gate", "shuffled"):
        row[f"{mode}_d_rot_pct"] = 100 * (res[mode][0] / res["normal"][0] - 1)
        row[f"{mode}_d_trans_pct"] = 100 * (np.hypot(res[mode][1], res[mode][2]) / np.hypot(res["normal"][1], res["normal"][2]) - 1)
    row["recompute_selfcheck_abs_diff_rot"] = abs(res["recompute"][0] - res["normal"][0])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--sweeps", nargs="*", default=[])
    ap.add_argument("--output_csv", default=None)
    a = ap.parse_args()

    runs = [os.path.abspath(r) for r in a.runs]
    for s in a.sweeps:
        runs += [os.path.abspath(d) for d in sorted(glob.glob(os.path.join(s, "2026*")))]
    runs = [r for r in runs if os.path.exists(os.path.join(r, "checkpoint_best.pth"))]
    runs.sort(key=lambda r: str(resolve_split_seed(load_config(os.path.join(r, "config.yml")))))  # few split rebuilds
    out_csv = a.output_csv or os.path.join(I.REPO, "results", f"gate_ablation_{a.split}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for i, r in enumerate(runs, 1):
        row = run_one(r, a.split, dev)
        if row is None:
            print(f"[{i}/{len(runs)}] skip (no concat-style gate): {os.path.basename(r)}", flush=True)
            continue
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)          # incremental, so a crash keeps progress
        print(f"[{i}/{len(runs)}] {row['fusion']:42s} seed {row['split_seed']}  gate mean {row['gate_mean']:.3f} "
              f"between-sample std {row['gate_between_sample_std']:.4f} | rot_x normal {row['normal_rot_x']:.3f} "
              f"mean_gate {row['mean_gate_d_rot_pct']:+.1f}% shuffled {row['shuffled_d_rot_pct']:+.1f}% "
              f"(selfcheck {row['recompute_selfcheck_abs_diff_rot']:.1e})", flush=True)

    if rows:
        df = pd.DataFrame(rows)
        df["fusion_label"] = df.fusion + np.where(df.dual_fusion, " +dual", "")
        summary = df.groupby("fusion_label").agg(
            n=("run", "count"), gate_between_sample_std=("gate_between_sample_std", "mean"),
            mean_gate_rot_pct=("mean_gate_d_rot_pct", "mean"), shuffled_rot_pct=("shuffled_d_rot_pct", "mean"),
            mean_gate_trans_pct=("mean_gate_d_trans_pct", "mean"), shuffled_trans_pct=("shuffled_d_trans_pct", "mean"))
        print("\nMean change in error when the per-sample gate is removed (positive = worse without it):")
        print(summary.round(3).to_string())
        print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
