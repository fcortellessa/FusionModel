# scripts/train.py

import os
import sys
import csv
import time
import torch
import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from base_model.pose_estimation_model import PoseEstimationNet
from data_processing.preprocess_data import preprocess_data
from losses.se2_losses import mse_loss, total_loss_2D_vector
from losses.absolute_pose_error import absolute_pose_error_x_rot
from losses.original_loss import total_loss
from utils.utils2 import setup_logger, load_config, save_checkpoint, load_checkpoint

from data_processing.dataloader_preprocessed import get_dataloaders

logger, log_file = setup_logger()


def _format_fusion_weights(fusion_weights):
    if fusion_weights is None:
        return ""
    return "[" + ",".join(f"{weight:.4f}" for weight in fusion_weights.tolist()) + "]"


def get_optimizer(model, lr=0.001):
    return torch.optim.Adam(model.parameters(), lr=lr)

def train_loop(
    model,
    num_epochs,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    last_epoch=0,
    checkpointing=False,
    metrics_csv_path=None,
    run_metadata=None,
):
    best_loss = float("inf")
    csv_file = None
    csv_writer = None
    metadata = run_metadata or {}

    if metrics_csv_path is not None:
        csv_file = open(metrics_csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp",
                "epoch",
                "train_loss",
                "loss_q",
                "loss_y",
                "loss_z",
                "train_abs_trans",
                "train_abs_rot_x",
                "val_abs_trans",
                "val_abs_rot_x",
                "fusion_weights",
                "best_loss",
                "epoch_time_sec",
                "checkpoint_path",
                "model_name",
                "streams",
                "fusion",
                "head_type",
                "freeze_backbone",
                "lr",
                "batch_size",
                "seed",
            ],
        )
        csv_writer.writeheader()

    try:
        for epoch in range(last_epoch, num_epochs):
            epoch_start = time.perf_counter()
            model.train()
            train_metrics = {
                "loss": 0.0,
                "loss_q": 0.0,
                "loss_y": 0.0,
                "loss_z": 0.0,
                "abs_trans": 0.0,
                "abs_rot_x": 0.0,
            }

            fusion_weights = None
            tqdm_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)", leave=False, mininterval=1.0)
            for batch in tqdm_bar:
                optimizer.zero_grad()

                batch = {k: v.to(device) for k, v in batch.items()}
                pose = batch["pose"].to(device)

                pose_pred, fusion_weights = model(batch)    # fusion_weights is only returned for logging/debugging purposes

                loss, loss_q, loss_y, loss_z = criterion(pose, pose_pred["pose"])
                abs_trans_y, abs_trans_z, abs_trans, abs_rot_x = absolute_pose_error_x_rot(pose_pred["pose"].detach(), pose)

                # Update metrics
                current_vals = {
                    "loss": loss.item(),
                    "loss_q": loss_q.item(),
                    "loss_y": loss_y.item(),
                    "loss_z": loss_z.item(),
                    "abs_trans": abs_trans.item(),
                    "abs_rot_x": abs_rot_x.item(),
                }

                for key, value in current_vals.items():
                    train_metrics[key] += value

                # Display key weights/losses in the bar itself
                t_loss = train_metrics["loss"] / (tqdm_bar.n + 1)
                tqdm_bar.set_postfix(avg_loss=f"{t_loss:.4f}", q_wt=f"{loss_q.item():.4f}")

                # Debug prints to check if absolute errors are reasonable
                # tqdm_bar.write(f"Batch Absolute Translation Error (y/z): {abs_trans_y:.4f}, {abs_trans_z:.4f}, Total: {abs_trans:.4f} | Batch Absolute Rotation Error (X-axis): {abs_rot_x:.4f}")
                # print(f"Batch Absolute Translation Error (y/z): {abs_trans_y},{abs_trans_z}", f"target pose: {pose[0, 1:3]}, predicted pose: {pose_pred['pose'][0, 1:3]}")
                # print(f"Batch Absolute Rotation Error: {abs_rot_x}", f"target quat: {pose[0, 3:]}, predicted quat: {pose_pred['pose'][0, 3:]}")

                loss.backward()
                optimizer.step()

                tqdm_bar.set_postfix(loss=f"{loss.item():.4f}")

            num_train = len(train_loader)
            num_val = len(val_loader)
            fw_str = _format_fusion_weights(fusion_weights)

            epoch_avg_loss = train_metrics["loss"] / num_train
            avg_q = train_metrics["loss_q"] / num_train
            avg_y = train_metrics["loss_y"] / num_train
            avg_z = train_metrics["loss_z"] / num_train
            train_abs_trans = train_metrics["abs_trans"] / num_train
            train_abs_rot_x = train_metrics["abs_rot_x"] / num_train
            epoch_time = time.perf_counter() - epoch_start

            logger.info(
                f"epoch={epoch+1} train_loss={epoch_avg_loss:.4f} loss_q={avg_q:.4f} "
                f"loss_y={avg_y:.4f} loss_z={avg_z:.4f} train_abs_trans={train_abs_trans:.4f} "
                f"train_abs_rot_x={train_abs_rot_x:.4f} fusion_weights={fw_str} epoch_time_sec={epoch_time:.2f}"
            )

            checkpoint_path = ""
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                checkpoint_path = save_checkpoint(model, optimizer, epoch, epoch_avg_loss, checkpoint_dir="checkpoints/best")

            # Validation phase
            model.eval()
            val_abs_trans, val_abs_rot_x = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    pose_pred, _ = model(batch)
                    _, _, at, arx = absolute_pose_error_x_rot(pose_pred["pose"], batch["pose"])
                    val_abs_trans += at.item()
                    val_abs_rot_x += arx.item()

            val_abs_trans_avg = val_abs_trans / num_val
            val_abs_rot_x_avg = val_abs_rot_x / num_val

            logger.info(
                f"epoch={epoch+1}/{num_epochs} train_loss={epoch_avg_loss:.4f} train_abs_trans={train_abs_trans:.4f} "
                f"train_abs_rot_x={train_abs_rot_x:.4f} val_abs_trans={val_abs_trans_avg:.4f} "
                f"val_abs_rot_x={val_abs_rot_x_avg:.4f} fusion_weights={fw_str}"
            )

            logger.debug(f"Epoch {epoch+1} detailed: q={avg_q:.4f}, y={avg_y:.4f}")

            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "epoch": epoch + 1,
                        "train_loss": f"{epoch_avg_loss:.6f}",
                        "loss_q": f"{avg_q:.6f}",
                        "loss_y": f"{avg_y:.6f}",
                        "loss_z": f"{avg_z:.6f}",
                        "train_abs_trans": f"{train_abs_trans:.6f}",
                        "train_abs_rot_x": f"{train_abs_rot_x:.6f}",
                        "val_abs_trans": f"{val_abs_trans_avg:.6f}",
                        "val_abs_rot_x": f"{val_abs_rot_x_avg:.6f}",
                        "fusion_weights": fw_str,
                        "best_loss": f"{best_loss:.6f}",
                        "epoch_time_sec": f"{epoch_time:.2f}",
                        "checkpoint_path": checkpoint_path,
                        "model_name": metadata.get("model_name", ""),
                        "streams": metadata.get("streams", ""),
                        "fusion": metadata.get("fusion", ""),
                        "head_type": metadata.get("head_type", ""),
                        "freeze_backbone": metadata.get("freeze_backbone", ""),
                        "lr": metadata.get("lr", ""),
                        "batch_size": metadata.get("batch_size", ""),
                        "seed": metadata.get("seed", ""),
                    }
                )
    finally:
        if csv_file is not None:
            csv_file.close()

def main(config_path, checkpointing=False):
  
    cfg = load_config(config_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_csv_path = log_file.replace(".log", "_metrics.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess data (split and transform) if directory doesn't exist
    data_dir = os.path.join(base_dir, "../Data")
    preprocessed_dir = os.path.join(data_dir, "Data_split/train_preprocessed")
    if not os.path.exists(preprocessed_dir):
        preprocess_data(root_dir=data_dir)
    

    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=os.path.join(data_dir, "Data_split"),
        batch_size=3,
        shuffle=True
    )

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
    ).to(device)

    optimizer = get_optimizer(
        model,
        lr=cfg["training"]["lr"]
    )

    run_metadata = {
        "model_name": cfg["model"]["backbone"],
        "streams": cfg["model"]["streams"],
        "fusion": cfg["model"]["fusion"],
        "head_type": cfg["model"]["head_type"],
        "freeze_backbone": cfg["model"]["freeze_backbone"],
        "lr": cfg["training"]["lr"],
        "batch_size": 3,
        "seed": cfg.get("training", {}).get("seed", ""),
    }

    logger.info(
        f"run_start log_file={log_file} metrics_csv={metrics_csv_path} "
        f"model={run_metadata['model_name']} fusion={run_metadata['fusion']} "
        f"head_type={run_metadata['head_type']} batch_size={run_metadata['batch_size']} lr={run_metadata['lr']}"
    )

    loss_function = total_loss

    train_loop(
        model=model,
        num_epochs=cfg["training"]["epochs"],
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=loss_function,
        device=device, 
        checkpointing=checkpointing,
        metrics_csv_path=metrics_csv_path,
        run_metadata=run_metadata
    )


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)

    # args = parser.parse_args()

    # main(args.config)

    # main("D:/MA/GripFusionNet/GripFusionNet/MyModel/configs/default.yml")
    main("configs/gated.yml", checkpointing = False)