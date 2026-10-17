#!/usr/bin/env python3
"""
Does the SE block (and, for reference, the token gate) of a trained se_gated_cross_attention do
per-sample work? -- mean / shuffled / removed ablation (same idea as gate_ablation_test.py, which only
covers the gated_concat family).

For each run (se_gated_cross_attention, shared or dual_fusion), evaluate checkpoint_best.pth on the test
split with:
  normal           the model as trained
  se_mean          SE weights replaced by their dataset mean (removes all per-sample variation)
  se_shuffled      each sample gets another sample's SE weights
  se_ones          SE weights set to 1 (removes the SE factor; the net is trained with it, so this only
                   shows the net adapted to its level, not that it helps)
  tokgate_mean / tokgate_shuffled   same for the token-wise gate logits
`se_between_sample_std` is the across-sample std of the SE weights (0 = a constant per-channel scale).

Only the 1D module (fusion/gated_cross_attention.py::SEGatedCrossAttentionFusion: `se_block` +
`spatial_gate`) is supported; the 2D module's SE branch outputs logits, probe it after training.

The test split on disk must already match the run's split seed (the script aborts instead of rebuilding
Data_split, unlike gate_ablation_test.py / test.py).

Usage:
    python scripts/evaluate/se_block_ablation_test.py --runs runs/<sweep>/<run_dir> [...] [--output_csv f.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_split_gpu as I
sys.path.insert(0, I.REPO)
from torch.utils.data import DataLoader
from data_processing.dataloader_preprocessed import PreprocessedTactileDataset, tactile_collate_fn
from data_processing.GripFusionNet_split import split_manifest_matches
from utils.utils import get_optimizer, load_checkpoint, load_config, resolve_data_root, resolve_split_seed, resolve_group_aware_split

class Swap:
    """Forward hook that records per-element across-sample stats and can replace the output."""
    def __init__(self): self.mode = "observe"; self.mean = None; self.reset()
    def reset(self): self.n = 0; self.s = None; self.ss = None
    def __call__(self, module, args, out):
        o = out.detach().double()
        s, ss = o.sum(0), (o ** 2).sum(0)
        self.s = s if self.s is None else self.s + s; self.ss = ss if self.ss is None else self.ss + ss; self.n += o.shape[0]
        if self.mode == "observe": return None
        if self.mode == "mean": return self.mean.to(out.dtype).unsqueeze(0).expand_as(out).clone()
        if self.mode == "shuffled": return torch.roll(out, 1, 0) if out.shape[0] > 1 else out
        if self.mode == "ones": return torch.ones_like(out)
        raise ValueError(self.mode)
    def finish(self):
        m = self.s / self.n; v = (self.ss / self.n - m ** 2).clamp_min(0)
        return m.float(), v.sqrt()

def evaluate(model, loader, dev, cfg):
    P, G = [], []
    with torch.no_grad():
        for b in loader:
            x = {k: v.to(dev) for k, v in b.items() if k != "pose"}
            out, _ = model(x); P.append(out["pose"].cpu().numpy()); G.append(b["pose"].numpy())
    df = I.errors_frame(np.concatenate(P), np.concatenate(G), cfg)
    return df.val_abs_rot_x.mean(), df.val_abs_y.mean() * 1e3, df.val_abs_z.mean() * 1e3

def probe(run_dir, dev):
    cfg = load_config(os.path.join(run_dir, "config.yml"))
    root = resolve_data_root(cfg, I.REPO)
    assert split_manifest_matches(root, os.path.join(root, "Data_split"), seed=resolve_split_seed(cfg),
                                  group_aware=resolve_group_aware_split(cfg)), "split on disk differs -> would rebuild; aborting"
    loader = DataLoader(PreprocessedTactileDataset(os.path.join(root, "Data_split", "test", "test_preprocessing")),
                        batch_size=16, shuffle=False, num_workers=2, collate_fn=tactile_collate_fn)
    model = I.build_model(cfg)
    opt = get_optimizer(model_or_params=model, optimizer_name=cfg["training"]["optimizer"], lr=cfg["training"]["lr"])
    load_checkpoint(model, opt, os.path.join(run_dir, "checkpoint_best.pth"), device="cpu", load_optimizer=False)
    model.to(dev).eval()
    fus = [getattr(model, n) for n in ("fusion", "fusion_rotation", "fusion_translation") if hasattr(model, n)]
    se = [Swap() for _ in fus]; tg = [Swap() for _ in fus]
    hs = [f.se_block.register_forward_hook(h) for f, h in zip(fus, se)] + \
         [f.spatial_gate.register_forward_hook(h) for f, h in zip(fus, tg)]
    res = {}
    for h in se + tg: h.mode = "observe"
    res["normal"] = evaluate(model, loader, dev, cfg)
    se_stats = [h.finish() for h in se]; tg_stats = [h.finish() for h in tg]
    for h, (m, _) in zip(se, se_stats): h.mean = m.to(dev)
    for h, (m, _) in zip(tg, tg_stats): h.mean = m.to(dev)
    for name, hooks, modes in [("se", se, ("mean", "shuffled", "ones")), ("tokgate", tg, ("mean", "shuffled"))]:
        for mode in modes:
            for h in se + tg: h.mode = "observe"
            for h in hooks: h.mode = mode
            res[f"{name}_{mode}"] = evaluate(model, loader, dev, cfg)
    for h in hs: h.remove()
    # SE output = sigmoid weights (B,256): across-sample std per channel vs across-channel spread of the mean
    ses = [(m.cpu().numpy(), s.cpu().numpy()) for m, s in se_stats]
    row = dict(run=os.path.basename(run_dir), dual=bool(cfg["model"].get("dual_fusion")), n_fusion=len(fus),
               se_weight_mean=float(np.mean([m.mean() for m, _ in ses])),
               se_between_sample_std=float(np.mean([s.mean() for _, s in ses])),
               se_between_channel_std_of_mean=float(np.mean([m.std() for m, _ in ses])),
               tokgate_logit_between_sample_std=float(np.mean([s.mean().item() for _, s in tg_stats])),
               tokgate_logit_between_token_channel_std_of_mean=float(np.mean([m.std().item() for m, _ in tg_stats])))
    for k, (r, y, z) in res.items():
        row[f"{k}_rot"], row[f"{k}_y"], row[f"{k}_z"] = r, y, z
    return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output_csv", default=os.path.join(I.REPO, "results", "se_block_ablation_test.csv"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.DataFrame([probe(os.path.abspath(r), dev) for r in a.runs])
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(df.T.to_string())
    os.makedirs(os.path.dirname(a.output_csv), exist_ok=True)
    df.to_csv(a.output_csv, index=False)
    print(f"wrote {a.output_csv}")
