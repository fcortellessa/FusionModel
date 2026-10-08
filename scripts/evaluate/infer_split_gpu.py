"""Per-sample inference of checkpoint_best.pth on the val (or test) split, on GPU.

Mirrors scripts/test.py::infer_and_evaluate (same model construction, same error formulas -- the
helper functions are imported from it) but: runs on CUDA, can read the val split, and writes
<run>/results_<split>/results_<split>.csv instead of touching <run>/results/.

  python scripts/evaluate/infer_split_gpu.py --split test --runs <run_dir> [...]     (sanity check vs test.py's csv)
  python scripts/evaluate/infer_split_gpu.py --split val  --sweeps runs/final_ablation runs/final_ablation_rot_scaling_200
"""
import argparse
import glob
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

spec = importlib.util.spec_from_file_location("repo_test_script", os.path.join(REPO, "scripts", "test.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)   # only defines functions (main is guarded) -- reused, not re-implemented

from data_processing.dataloader_preprocessed import PreprocessedTactileDataset, tactile_collate_fn
from data_processing.preprocess_data import ensure_test_split_current
from utils.normalization import canonicalize_translation_norm_stats
from utils.utils import load_checkpoint, load_config, resolve_data_root, resolve_split_seed, get_optimizer


def build_model(config):
    return T.PoseEstimationNet(
        model_name=config["model"]["backbone"],
        streams=config["model"]["streams"],
        fusion=config["model"]["fusion"],
        freeze_backbone=config["model"]["freeze_backbone"],
        rotation_parametrisation=config["model"]["head_type"],
        activation=config["model"]["activation"],
        rotation_layers=[config["model"]["mlp_quat_1"], config["model"]["mlp_quat_2"], config["model"]["mlp_quat_3"]],
        translation_layers=[config["model"]["mlp_trans_1"], config["model"]["mlp_trans_2"], config["model"]["mlp_trans_3"]],
        dropout_rot=config["training"]["dropout_rot"],
        dropout_trans=config["training"]["dropout_trans"],
        use_batchnorm=config["training"]["use_batchnorm"],
        translation_head_mode=config["model"].get("translation_head_mode", "separate"),
        spatial_grid_size=config["model"].get("spatial_grid_size", (10, 10)),
        task_adapter=config["model"].get("task_adapter", None),
        task_feature_adapter=T._is_task_feature_adapter_enabled(config),
        task_feature_adapter_type=T._get_task_feature_adapter_type(config),
        dual_fusion=bool(config["model"].get("dual_fusion", False)),
        gate_init_prob=config["model"].get("gate_init_prob", None),
        pre_head_layernorm=bool(config["model"].get("pre_head_layernorm", False)),
        head_bottleneck_rank=config["model"].get("head_bottleneck_rank", None),
        pre_head_channel_divisor=config["model"].get("pre_head_channel_divisor", None),
    )


def errors_frame(pred_pose, gt_pose, config):
    """Same formulas as test.py lines ~151-237."""
    gt_tr, pred_tr = gt_pose[:, :3].copy(), pred_pose[:, :3].copy()
    pred_rot, gt_rot = pred_pose[:, 3:7], gt_pose[:, 3:7]
    if config.get("data", {}).get("normalize_translation", True):
        st = canonicalize_translation_norm_stats(config.get("data", {}).get("translation_norm_stats", {}))
        pred_tr[:, 1] = pred_tr[:, 1] * st["std_y"] + st["mean_y"]
        pred_tr[:, 2] = pred_tr[:, 2] * st["std_z"] + st["mean_z"]
        gt_tr[:, 1] = gt_tr[:, 1] * st["std_y"] + st["mean_y"]
        gt_tr[:, 2] = gt_tr[:, 2] * st["std_z"] + st["mean_z"]
    signed = pred_tr - gt_tr
    absv = np.abs(signed)
    pred_roll, gt_roll = T._quaternion_roll(pred_rot), T._quaternion_roll(gt_rot)
    d = (pred_roll - gt_roll + np.pi) % (2.0 * np.pi) - np.pi
    signed_rot_x = np.degrees(d)
    return pd.DataFrame({
        "sample_id": np.arange(len(gt_pose)),
        "pred_trans_x": pred_tr[:, 0], "pred_trans_y": pred_tr[:, 1], "pred_trans_z": pred_tr[:, 2],
        "gt_trans_x": gt_tr[:, 0], "gt_trans_y": gt_tr[:, 1], "gt_trans_z": gt_tr[:, 2],
        "pred_roll_deg": np.degrees(pred_roll), "gt_roll_deg": np.degrees(gt_roll),
        "val_abs_trans": np.linalg.norm(absv, axis=1),
        "val_abs_x": absv[:, 0], "val_abs_y": absv[:, 1], "val_abs_z": absv[:, 2],
        "val_abs_rot_x": np.abs(signed_rot_x),
        "val_abs_rot_z": T._quaternion_geodesic_error_deg(pred_rot, gt_rot),
        "val_signed_x": signed[:, 0], "val_signed_y": signed[:, 1], "val_signed_z": signed[:, 2],
        "val_signed_rot_x": signed_rot_x,
    })


def run_one(run_dir, split, device):
    cfg = load_config(os.path.join(run_dir, "config.yml"))
    data_root = resolve_data_root(cfg, REPO)
    ensure_test_split_current(cfg, data_root)          # rebuilds Data_split for this run's split seed if needed
    folder = os.path.join(data_root, "Data_split", split, f"{split}_preprocessing")
    loader = DataLoader(PreprocessedTactileDataset(folder), batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=tactile_collate_fn)
    model = build_model(cfg)
    opt = get_optimizer(model_or_params=model, optimizer_name=cfg["training"]["optimizer"], lr=cfg["training"]["lr"])
    load_checkpoint(model, opt, os.path.join(run_dir, "checkpoint_best.pth"), device="cpu", load_optimizer=False)
    model.to(device).eval()
    preds, gts = [], []
    with torch.no_grad():
        for sample in loader:
            inputs = {k: v.to(device) for k, v in sample.items() if k != "pose"}
            out, _ = model(inputs)
            preds.append(out["pose"].cpu().numpy()); gts.append(sample["pose"].numpy())
    df = errors_frame(np.concatenate(preds), np.concatenate(gts), cfg)
    out_dir = os.path.join(run_dir, f"results_{split}")
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, f"results_{split}*.csv")):
        os.remove(old)
    df.to_csv(os.path.join(out_dir, f"results_{split}.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--sweeps", nargs="*", default=[])
    a = ap.parse_args()
    runs = [os.path.abspath(r) for r in a.runs]
    for s in a.sweeps:
        runs += [os.path.abspath(d) for d in sorted(glob.glob(os.path.join(s, "2026*")))]
    runs = [r for r in runs if os.path.exists(os.path.join(r, "checkpoint_best.pth"))]

    def seed_key(r):
        try:
            return str(resolve_split_seed(load_config(os.path.join(r, "config.yml"))))
        except Exception:
            return "z"
    runs.sort(key=seed_key)                              # same split seed consecutively -> few split rebuilds
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(runs)} runs, split={a.split}, device={dev}", flush=True)
    for i, r in enumerate(runs, 1):
        t0 = time.time()
        df = run_one(r, a.split, dev)
        print(f"[{i}/{len(runs)}] {os.path.relpath(r, REPO)}  n={len(df)}  rot_x={df.val_abs_rot_x.mean():.4f}  "
              f"y={df.val_abs_y.mean()*1e3:.4f}mm  z={df.val_abs_z.mean()*1e3:.4f}mm  ({time.time()-t0:.0f}s)", flush=True)
