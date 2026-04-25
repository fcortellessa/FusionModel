# test script
import argparse
import json
import time
import os
import sys
import torch
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from base_model.pose_estimation_model import PoseEstimationNet
from data_processing.dataloader_preprocessed import get_test_dataloader
from utils.utils import load_checkpoint, load_config, get_optimizer


config_path = "configs/gated.yml"
root_data_path = "Data/Data_split/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path, help="Path to the training config used for the run")
    parser.add_argument("--run_dir", type=str, default=None, help="Specific run directory to load instead of the latest run")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path to load")
    parser.add_argument("--results_dir", type=str, default=None, help="Override the directory where test CSVs are saved")
    args = parser.parse_args()

    if args.run_dir is not None:
        manifest_path = os.path.join(args.run_dir, "manifest.json")
        legacy_manifest_path = os.path.join(args.run_dir, "run_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        elif os.path.exists(legacy_manifest_path):
            with open(legacy_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {
                "config_snapshot_path": os.path.join(args.run_dir, "config.yml"),
                "checkpoint_path": os.path.join(args.run_dir, "checkpoint_best.pth"),
                "run_dir": args.run_dir,
                "results_dir": os.path.join(args.run_dir, "results"),
            }

            if not os.path.exists(manifest["config_snapshot_path"]):
                manifest["config_snapshot_path"] = os.path.join(args.run_dir, "config.resolved.yml")
            if not os.path.exists(manifest["checkpoint_path"]):
                manifest["checkpoint_path"] = os.path.join(args.run_dir, "checkpoints", "best.pth")
    else:
        if args.checkpoint is None:
            raise ValueError("Provide either --run_dir or --checkpoint.")
        manifest = {
            "config_snapshot_path": args.config,
            "checkpoint_path": args.checkpoint,
            "results_dir": "results",
        }

    cfg_path = manifest.get("config_snapshot_path") or args.config
    cfg = load_config(cfg_path)
    model_path = args.checkpoint or manifest["checkpoint_path"]
    results_dir = args.results_dir or manifest.get("results_dir") or os.path.join(os.path.dirname(model_path), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    model = PoseEstimationNet(
        model_name=cfg["model"]["backbone"],
        streams=cfg["model"]["streams"],
        fusion=cfg["model"]["fusion"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        rotation_parametrisation=cfg["model"]["head_type"],
        activation=cfg["model"]["activation"],
        rotation_layers=[cfg["model"]["mlp_quat_1"], cfg["model"]["mlp_quat_2"], cfg["model"]["mlp_quat_3"]],
        translation_layers=[cfg["model"]["mlp_trans_1"], cfg["model"]["mlp_trans_2"],cfg["model"]["mlp_trans_3"]],
        dropout_rot=cfg["training"]["dropout_rot"],
        dropout_trans=cfg["training"]["dropout_trans"],
        use_batchnorm=cfg["training"]["use_batchnorm"]
    )
    # load state dict
    test_loader = get_test_dataloader(
        root_dir=root_data_path,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )
    optimizer = get_optimizer(model_or_params=model, optimizer_name=cfg["training"]["optimizer"], lr=cfg["training"]["lr"])
    model.load_state_dict(load_checkpoint(model, optimizer, model_path, device="cpu")["model_state_dict"])
    print("Model loaded successfully!")

    # load test data
    # test_data = torch.load(test_data_path)

    # run inference
    model.eval()
    csv_results = {
        "sample_id": [],
        "pred_translation": [],
        "pred_rotation": [],
        "pred_quaternion": [],
        "gt_pose": []
    }
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            inputs = {k: v.unsqueeze(0) for k, v in sample.items() if k != "pose"}
            target = sample["pose"].unsqueeze(0)

            y, z, q = model(inputs)
            # Store results for CSV export
            csv_results["sample_id"].append(i)
            csv_results["pred_translation"].append(y.cpu().numpy())
            csv_results["pred_rotation"].append(z.cpu().numpy())
            csv_results["pred_quaternion"].append(q.cpu().numpy())
            csv_results["gt_pose"].append(target.cpu().numpy())

    # Save results to CSV
    df = pd.DataFrame(csv_results)
    df.to_csv(os.path.join(results_dir, f"results_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"), index=False)
