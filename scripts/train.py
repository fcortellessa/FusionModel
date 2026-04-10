# scripts/train.py

import os
import sys
import torch
import tqdm
import yaml
from pathlib import Path
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from base_model.pose_estimation_model import PoseEstimationNet
from data_processing.preprocess_data import preprocess_data
from losses.se2_losses import mse_loss, total_loss_2D_vector
from losses.absolute_pose_error import absolute_pose_error_x_rot
from losses.original_loss import total_loss
from utils.utils import setup_logger, load_config, save_checkpoint, load_checkpoint

from data_processing.dataloader_preprocessed import get_dataloaders

logger = logging.getLogger("GripFusionNet")
logger = setup_logger()


def get_optimizer(model, lr=0.001):
    return torch.optim.Adam(model.parameters(), lr=lr)

def train_loop(model, num_epochs, train_loader, val_loader, optimizer, criterion, device, last_epoch=0, saving_interval=1):
    for epoch in range(last_epoch, num_epochs):
        model.train()
        train_metrics = {
        "loss": 0.0, "loss_q": 0.0, "loss_y": 0.0, "loss_z": 0.0,
        "abs_trans": 0.0, "abs_rot_x": 0.0
        }

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
                "loss": loss.item(), "loss_q": loss_q.item(), 
                "loss_y": loss_y.item(), "abs_trans": abs_trans.item(), "abs_rot_x": abs_rot_x.item()
            }
            
            for k, v in current_vals.items():
                train_metrics[k] += v

            # Display key weights/losses in the bar itself
            t_loss = train_metrics["loss"] / (tqdm_bar.n + 1)
            tqdm_bar.set_postfix(avg_loss=f"{t_loss:.4f}", q_wt=f"{loss_q.item():.4f}")

            # Debug prints to check if absolute errors are reasonable
            # tqdm_bar.write(f"Batch Absolute Translation Error (y/z): {abs_trans_y:.4f}, {abs_trans_z:.4f}, Total: {abs_trans:.4f} | Batch Absolute Rotation Error (X-axis): {abs_rot_x:.4f}")
            # print(f"Batch Absolute Translation Error (y/z): {abs_trans_y},{abs_trans_z}", f"target pose: {pose[0, 1:3]}, predicted pose: {pose_pred['pose'][0, 1:3]}")
            # print(f"Batch Absolute Rotation Error: {abs_rot_x}", f"target quat: {pose[0, 3:]}, predicted quat: {pose_pred['pose'][0, 3:]}")

            loss.backward()
            optimizer.step()

            # Accumulate
            train_metrics["loss"] += loss.item()
            train_metrics["loss_q"] += loss_q.item()
            train_metrics["loss_y"] += loss_y.item()
            train_metrics["loss_z"] += loss_z.item()
            train_metrics["abs_trans"] += abs_trans.item()
            train_metrics["abs_rot_x"] += abs_rot_x.item()

            tqdm_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate average metrics
        if fusion_weights is not None:
            fusion_weights_str = [f"{w:.2f}" for w in fusion_weights.tolist()]
        else:
            fusion_weights_str = "None"
        tqdm_bar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "gates": f"{fusion_weights_str}" 
        })

        epoch_loss = train_metrics["loss"] / len(train_loader)    # Average loss per batch over the epoch = total loss for the epoch
        avg_q = train_metrics["loss_q"] / len(train_loader)
        logger.info(f"Epoch {epoch+1} Complete | Avg Loss: {epoch_loss:.4f} | Avg Quat Loss: {avg_q:.4f}")

        # --- SAVE CHECKPOINT ---
        # Save every n epochs or based on some condition (e.g., improvement)

        save_checkpoint(model, optimizer, epoch, epoch_loss, interval=saving_interval)

        # Save "best" model separately if you want
        best_loss = float('inf') 
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, epoch, epoch_loss, checkpoint_dir="checkpoints/best")

        # Validation phase
        model.eval()
        val_abs_trans, val_abs_rot_x = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pose_pred, fusion_weights = model(batch)
                _, _, at, arx = absolute_pose_error_x_rot(pose_pred["pose"], batch["pose"])
                val_abs_trans += at.item()
                val_abs_rot_x += arx.item()

        # Normalize and Log
        num_train = len(train_loader)
        num_val = len(val_loader)
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} Summary: "
            f"Loss: {train_metrics['loss']/num_train:.4f} | "
            f"Train TransYZ Err: {train_metrics['abs_trans']/num_train:.4f} | "
            f"Train RotX Err: {train_metrics['abs_rot_x']/num_train:.4f} | "
            f"Val TransYZ Err: {val_abs_trans/num_val:.4f} | "
            f"Val RotX Err: {val_abs_rot_x/num_val:.4f}"
        )
        
        # Detailed logging (useful for debugging specific loss components)
        logger.debug(f"Epoch {epoch+1} detailed: q={train_metrics['loss_q']/num_train:.4f}, y={train_metrics['loss_y']/num_train:.4f}")

def main(config_path):
    cfg = load_config(config_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
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

    loss_function = total_loss

    train_loop(
        model=model,
        num_epochs=cfg["training"]["epochs"],
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=loss_function,
        device=device
    )


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)

    # args = parser.parse_args()

    # main(args.config)

    main("configs/default.yml")