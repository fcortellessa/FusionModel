# scripts/train.py

import os
import sys
import csv
import time
import argparse
import subprocess
import platform
import socket
import torch
import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from base_model.pose_estimation_model import PoseEstimationNet
from data_processing.preprocess_data import preprocess_data
from losses.se2_losses import mse_loss, total_loss_2D_vector
from losses.absolute_pose_error import absolute_pose_error_x_rot
from losses.original_loss import total_loss
from utils.utils import setup_logger, load_config, save_checkpoint, load_checkpoint, get_optimizer

from data_processing.dataloader_preprocessed import get_dataloaders

logger, log_file = setup_logger()


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_tensorboard(logdir, port=6006):
    if SummaryWriter is None:
        logger.warning("TensorBoard logging is unavailable because torch.utils.tensorboard could not be imported.")
        return None

    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        logdir,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    # Platform-specific kwargs to suppress console windows and ensure proper process group handling
    kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
    
    if platform.system() == "Windows":
        # This constant only exists on Windows
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        pass

    try:
        logger.info(f"Attempting to start TensorBoard at http://127.0.0.1:{port}")
        return subprocess.Popen(command, **kwargs)
    except Exception as exc:
        logger.warning(f"Could not start TensorBoard automatically: {exc}")
        return None

def _format_fusion_weights(fusion_weights):
    if fusion_weights is None:
        return ""
    return "[" + ",".join(f"{weight:.4f}" for weight in fusion_weights.tolist()) + "]"


def _compute_model_norms(model):
    grad_sq_sum = 0.0
    param_sq_sum = 0.0

    for param in model.parameters():
        param_sq_sum += param.detach().pow(2).sum().item()
        if param.grad is not None:
            grad_sq_sum += param.grad.detach().pow(2).sum().item()

    return param_sq_sum ** 0.5, grad_sq_sum ** 0.5


def _get_lr_by_group(optimizer, group_name):
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    return float(optimizer.param_groups[0]["lr"])


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


class UncertaintyWeighting(torch.nn.Module):
    def __init__(self, init_log_var_rot=0.0, init_log_var_trans=0.0):
        super().__init__()
        self.log_var_rot = torch.nn.Parameter(torch.tensor(float(init_log_var_rot), dtype=torch.float32))
        self.log_var_trans = torch.nn.Parameter(torch.tensor(float(init_log_var_trans), dtype=torch.float32))

    def forward(self, loss_rot, loss_trans):
        weighted_rot = torch.exp(-self.log_var_rot) * loss_rot + self.log_var_rot
        weighted_trans = torch.exp(-self.log_var_trans) * loss_trans + self.log_var_trans
        return weighted_rot + weighted_trans


def _build_named_param_groups(model, base_lr):
    grouped_params = {
        "base": [],
        "head_rot": [],
        "head_y": [],
        "head_z": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("rot_head."):
            grouped_params["head_rot"].append(param)
        elif name.startswith("y_head."):
            grouped_params["head_y"].append(param)
        elif name.startswith("z_head."):
            grouped_params["head_z"].append(param)
        else:
            grouped_params["base"].append(param)

    param_groups = []
    for group_name in ["base", "head_rot", "head_y", "head_z"]:
        params = grouped_params[group_name]
        if not params:
            continue
        param_groups.append(
            {
                "name": group_name,
                "params": params,
                "lr": float(base_lr),
            }
        )

    return param_groups


def train_loop(
    model,
    num_epochs,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    uncertainty_module=None,
    last_epoch=0,
    checkpointing=False,
    checkpoint_path=None,
    metrics_csv_path=None,
    run_metadata=None,
    tensorboard_log_dir=None,
    head_lr_decay_cfg=None,
):
    best_val_loss = float("inf")
    csv_file = None
    csv_writer = None
    tb_writer = None
    metadata = run_metadata or {}
    global_batch_step = 0
    saved_checkpoint_path = ""

    head_lr_decay_cfg = head_lr_decay_cfg or {}
    head_lr_decay_enabled = bool(head_lr_decay_cfg.get("enabled", True))
    head_patience = int(head_lr_decay_cfg.get("patience", 3))
    head_reduce_factor = float(head_lr_decay_cfg.get("factor", 0.3))
    head_min_lr_mult = float(head_lr_decay_cfg.get("min_lr_mult", 0.05))
    base_group = next((g for g in optimizer.param_groups if g.get("name") == "base"), optimizer.param_groups[0])
    base_lr = float(base_group["lr"])
    head_min_lrs = {
        "q": base_lr * head_min_lr_mult,
        "y": base_lr * head_min_lr_mult,
        "z": base_lr * head_min_lr_mult,
    }

    param_groups_by_name = {
        group.get("name", f"group_{idx}"): group
        for idx, group in enumerate(optimizer.param_groups)
    }
    head_lr_state = {
        "q": {"best_val": float("inf"), "bad_epochs": 0, "group_name": "head_rot"},
        "y": {"best_val": float("inf"), "bad_epochs": 0, "group_name": "head_y"},
        "z": {"best_val": float("inf"), "bad_epochs": 0, "group_name": "head_z"},
    }

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
                "lr_base",
                "lr_head_rot",
                "lr_head_y",
                "lr_head_z",
                "uncertainty_log_var_rot",
                "uncertainty_log_var_trans",
                "uncertainty_weight_rot",
                "uncertainty_weight_trans",
                "batch_size",
                "seed",
            ],
        )
        csv_writer.writeheader()

    if tensorboard_log_dir is not None:
        if SummaryWriter is None:
            logger.warning("TensorBoard is unavailable because torch.utils.tensorboard could not be imported.")
        else:
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)
            if metadata:
                tb_writer.add_text(
                    "run/metadata",
                    "\n".join(f"{key}: {value}" for key, value in metadata.items()),
                )

    try:
        for epoch in range(last_epoch, num_epochs):
            epoch_start = time.perf_counter()
            model.train()
            train_metrics = {
                "loss": 0.0,
                "loss_q": 0.0,
                "loss_y": 0.0,
                "loss_z": 0.0,
                "abs_y": 0.0,
                "abs_z": 0.0,
                "abs_trans": 0.0,
                "abs_rot_x": 0.0,
            }

            fusion_weights = None
            fusion_weight_sums = None
            fusion_weight_count = 0
            tqdm_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)", leave=False, mininterval=1.0)
            for batch in tqdm_bar:
                optimizer.zero_grad()

                batch = {k: v.to(device) for k, v in batch.items()}
                pose = batch["pose"].to(device)

                pose_pred, fusion_weights = model(batch)    # fusion_weights is only returned for logging/debugging purposes

                base_loss, loss_q, loss_y, loss_z = criterion(pose, pose_pred["pose"])
                loss_trans = loss_y + loss_z
                if uncertainty_module is not None:
                    loss = uncertainty_module(loss_q, loss_trans)
                else:
                    loss = base_loss
                abs_trans_y, abs_trans_z, abs_trans, abs_rot_x = absolute_pose_error_x_rot(pose_pred["pose"].detach(), pose)

                if fusion_weights is not None:
                    fusion_weights_flat = fusion_weights.detach().cpu().flatten().float()
                    if fusion_weight_sums is None:
                        fusion_weight_sums = fusion_weights_flat.clone()
                    else:
                        fusion_weight_sums += fusion_weights_flat
                    fusion_weight_count += 1

                # Update metrics
                current_vals = {
                    "loss": loss.item(),
                    "loss_q": loss_q.item(),
                    "loss_y": loss_y.item(),
                    "loss_z": loss_z.item(),
                    "abs_y": abs_trans_y.item(),
                    "abs_z": abs_trans_z.item(),
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

                if tb_writer is not None:
                    param_norm, grad_norm = _compute_model_norms(model)
                    tb_writer.add_scalar("train_step/loss_total", loss.item(), global_batch_step)
                    tb_writer.add_scalar("train_step/loss_q", loss_q.item(), global_batch_step)
                    tb_writer.add_scalar("train_step/loss_y", loss_y.item(), global_batch_step)
                    tb_writer.add_scalar("train_step/loss_z", loss_z.item(), global_batch_step)
                    tb_writer.add_scalar("train_step/abs_trans", abs_trans.item(), global_batch_step)
                    tb_writer.add_scalar("train_step/abs_rot_x", abs_rot_x.item(), global_batch_step)
                    tb_writer.add_scalar("train_step/grad_norm", grad_norm, global_batch_step)
                    tb_writer.add_scalar("train_step/param_norm", param_norm, global_batch_step)
                    tb_writer.add_scalar("train_step/lr", optimizer.param_groups[0]["lr"], global_batch_step)
                    for group_name, group in param_groups_by_name.items():
                        tb_writer.add_scalar(f"train_step/lr_{group_name}", group["lr"], global_batch_step)

                    if fusion_weights is not None:
                        for idx, weight in enumerate(fusion_weights.detach().cpu().flatten().tolist()):
                            tb_writer.add_scalar(f"fusion/weight_stream_{idx}", float(weight), global_batch_step)

                optimizer.step()
                global_batch_step += 1

                tqdm_bar.set_postfix(loss=f"{loss.item():.4f}")

            num_train = len(train_loader)
            num_val = len(val_loader)
            fw_str = _format_fusion_weights(fusion_weights)

            epoch_avg_loss = train_metrics["loss"] / num_train
            avg_q = train_metrics["loss_q"] / num_train
            avg_y = train_metrics["loss_y"] / num_train
            avg_z = train_metrics["loss_z"] / num_train
            train_abs_y = train_metrics["abs_y"] / num_train
            train_abs_z = train_metrics["abs_z"] / num_train
            train_abs_trans = train_metrics["abs_trans"] / num_train
            train_abs_rot_x = train_metrics["abs_rot_x"] / num_train
            epoch_time = time.perf_counter() - epoch_start

            fusion_weights_avg = None
            if fusion_weight_sums is not None and fusion_weight_count > 0:
                fusion_weights_avg = fusion_weight_sums / float(fusion_weight_count)

            fw_epoch_str = _format_fusion_weights(fusion_weights_avg) if fusion_weights_avg is not None else fw_str

            logger.info(
                f"epoch={epoch+1} train_loss={epoch_avg_loss:.4f} loss_q={avg_q:.4f} "
                f"loss_y={avg_y:.4f} loss_z={avg_z:.4f} train_abs_trans={train_abs_trans:.4f} "
                f"train_abs_rot_x={train_abs_rot_x:.4f} fusion_weights={fw_epoch_str} epoch_time_sec={epoch_time:.2f}"
            )

            # Validation phase
            model.eval()
            val_loss_total = 0.0
            val_loss_q = 0.0
            val_loss_y = 0.0
            val_loss_z = 0.0
            val_abs_y = 0.0
            val_abs_z = 0.0
            val_abs_trans, val_abs_rot_x = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    pose = batch["pose"].to(device)
                    pose_pred, _ = model(batch)
                    val_base_loss, val_q, val_y, val_z = criterion(pose, pose_pred["pose"])
                    val_trans = val_y + val_z
                    if uncertainty_module is not None:
                        val_loss = uncertainty_module(val_q, val_trans)
                    else:
                        val_loss = val_base_loss
                    ay, az, at, arx = absolute_pose_error_x_rot(pose_pred["pose"], pose)
                    val_loss_total += val_loss.item()
                    val_loss_q += val_q.item()
                    val_loss_y += val_y.item()
                    val_loss_z += val_z.item()
                    val_abs_y += ay.item()
                    val_abs_z += az.item()
                    val_abs_trans += at.item()
                    val_abs_rot_x += arx.item()

            val_loss_avg = val_loss_total / num_val
            val_q_avg = val_loss_q / num_val
            val_y_avg = val_loss_y / num_val
            val_z_avg = val_loss_z / num_val
            val_abs_y_avg = val_abs_y / num_val
            val_abs_z_avg = val_abs_z / num_val
            val_abs_trans_avg = val_abs_trans / num_val
            val_abs_rot_x_avg = val_abs_rot_x / num_val
            lr_base = _get_lr_by_group(optimizer, "base")
            lr_head_rot = _get_lr_by_group(optimizer, "head_rot")
            lr_head_y = _get_lr_by_group(optimizer, "head_y")
            lr_head_z = _get_lr_by_group(optimizer, "head_z")
            uncertainty_log_var_rot = ""
            uncertainty_log_var_trans = ""
            uncertainty_weight_rot = ""
            uncertainty_weight_trans = ""
            if uncertainty_module is not None:
                uncertainty_log_var_rot = _to_float(uncertainty_module.log_var_rot)
                uncertainty_log_var_trans = _to_float(uncertainty_module.log_var_trans)
                uncertainty_weight_rot = float(torch.exp(-uncertainty_module.log_var_rot.detach()).item())
                uncertainty_weight_trans = float(torch.exp(-uncertainty_module.log_var_trans.detach()).item())

            if head_lr_decay_enabled:
                per_head_losses = {
                    "q": {"train": avg_q, "val": val_q_avg},
                    "y": {"train": avg_y, "val": val_y_avg},
                    "z": {"train": avg_z, "val": val_z_avg},
                }
                for head_key, head_vals in per_head_losses.items():
                    state = head_lr_state[head_key]
                    best_val = state["best_val"]
                    val_head = head_vals["val"]
                    train_head = head_vals["train"]
                    improved = val_head < best_val
                    if improved:
                        state["best_val"] = val_head
                        state["bad_epochs"] = 0
                        continue

                    overfit_detected = val_head > train_head
                    if overfit_detected:
                        state["bad_epochs"] += 1
                    else:
                        state["bad_epochs"] = 0

                    if state["bad_epochs"] < head_patience:
                        continue

                    group_name = state["group_name"]
                    group = param_groups_by_name.get(group_name)
                    if group is None:
                        continue

                    old_lr = float(group["lr"])
                    new_lr = max(old_lr * head_reduce_factor, head_min_lrs[head_key])
                    if new_lr < old_lr:
                        group["lr"] = new_lr
                        logger.info(
                            f"head_lr_reduce epoch={epoch+1} head={head_key} group={group_name} "
                            f"old_lr={old_lr:.8f} new_lr={new_lr:.8f} train_head={train_head:.6f} "
                            f"val_head={val_head:.6f} bad_epochs={state['bad_epochs']}"
                        )
                    state["bad_epochs"] = 0

            # Save checkpoint when validation improves; keep only the current best.
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                saved_checkpoint_path = save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss_avg,
                    checkpoint_path,
                )

            logger.info(
                f"epoch={epoch+1}/{num_epochs} train_loss={epoch_avg_loss:.4f} val_loss={val_loss_avg:.4f} "
                f"train_abs_trans={train_abs_trans:.4f} "
                f"train_abs_rot_x={train_abs_rot_x:.4f} val_abs_trans={val_abs_trans_avg:.4f} "
                f"val_abs_rot_x={val_abs_rot_x_avg:.4f} lr_base={lr_base:.8f} "
                f"lr_head_rot={lr_head_rot:.8f} lr_head_y={lr_head_y:.8f} "
                f"lr_head_z={lr_head_z:.8f} fusion_weights={fw_str}"
            )
            if uncertainty_module is not None:
                logger.info(
                    f"uncertainty rot_log_var={uncertainty_log_var_rot:.6f} trans_log_var={uncertainty_log_var_trans:.6f} "
                    f"rot_weight={uncertainty_weight_rot:.6f} trans_weight={uncertainty_weight_trans:.6f}"
                )

            if tb_writer is not None:
                global_step = epoch + 1
                tb_writer.add_scalars("loss/total", {"train": epoch_avg_loss, "val": val_loss_avg}, global_step)
                tb_writer.add_scalars("loss/q", {"train": avg_q, "val": val_q_avg}, global_step)
                tb_writer.add_scalars("loss/y", {"train": avg_y, "val": val_y_avg}, global_step)
                tb_writer.add_scalars("loss/z", {"train": avg_z, "val": val_z_avg}, global_step)
                tb_writer.add_scalars("metrics/abs_y", {"train": train_abs_y, "val": val_abs_y_avg}, global_step)
                tb_writer.add_scalars("metrics/abs_z", {"train": train_abs_z, "val": val_abs_z_avg}, global_step)
                tb_writer.add_scalars("metrics/abs_trans", {"train": train_abs_trans, "val": val_abs_trans_avg}, global_step)
                tb_writer.add_scalars("metrics/abs_rot_x", {"train": train_abs_rot_x, "val": val_abs_rot_x_avg}, global_step)
                tb_writer.add_scalar("diagnostics/loss_gap", epoch_avg_loss - val_loss_avg, global_step)
                tb_writer.add_scalar("diagnostics/abs_y_gap", train_abs_y - val_abs_y_avg, global_step)
                tb_writer.add_scalar("diagnostics/abs_z_gap", train_abs_z - val_abs_z_avg, global_step)
                tb_writer.add_scalar("diagnostics/abs_trans_gap", train_abs_trans - val_abs_trans_avg, global_step)
                tb_writer.add_scalar("diagnostics/abs_rot_x_gap", train_abs_rot_x - val_abs_rot_x_avg, global_step)
                tb_writer.add_scalar("training/best_val_loss", best_val_loss, global_step)
                tb_writer.add_scalar("training/epoch_time_sec", epoch_time, global_step)
                tb_writer.add_scalars(
                    "training/lr",
                    {
                        "base": lr_base,
                        "head_rot": lr_head_rot,
                        "head_y": lr_head_y,
                        "head_z": lr_head_z,
                    },
                    global_step,
                )
                if uncertainty_module is not None:
                    tb_writer.add_scalars(
                        "uncertainty/log_var",
                        {
                            "rot": uncertainty_log_var_rot,
                            "trans": uncertainty_log_var_trans,
                        },
                        global_step,
                    )
                    tb_writer.add_scalars(
                        "uncertainty/weight",
                        {
                            "rot": uncertainty_weight_rot,
                            "trans": uncertainty_weight_trans,
                        },
                        global_step,
                    )
                for group_name, group in param_groups_by_name.items():
                    tb_writer.add_scalar(f"training/lr_groups/{group_name}", group["lr"], global_step)
                if fusion_weights is not None:
                    tb_writer.add_text("fusion/weights", fw_str, global_step)
                if fusion_weights_avg is not None:
                    for idx, weight in enumerate(fusion_weights_avg.tolist()):
                        tb_writer.add_scalar(f"fusion/epoch_weight_stream/{idx}", float(weight), global_step)

            if tb_writer is not None and epoch == num_epochs - 1 and metadata:
                hparam_dict = {
                    "model_name": str(metadata.get("model_name", "")),
                    "streams": str(metadata.get("streams", "")),
                    "fusion": str(metadata.get("fusion", "")),
                    "head_type": str(metadata.get("head_type", "")),
                    "freeze_backbone": str(metadata.get("freeze_backbone", "")),
                    "lr": float(metadata.get("lr", optimizer.param_groups[0]["lr"])),
                    "batch_size": int(metadata.get("batch_size", len(train_loader))),
                }
                metric_dict = {
                    "hparams/val_loss": float(val_loss_avg),
                    "hparams/val_abs_trans": float(val_abs_trans_avg),
                    "hparams/val_abs_rot_x": float(val_abs_rot_x_avg),
                }
                tb_writer.add_hparams(hparam_dict, metric_dict)

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
                        "best_loss": f"{best_val_loss:.6f}",
                        "epoch_time_sec": f"{epoch_time:.2f}",
                        "checkpoint_path": saved_checkpoint_path,
                        "model_name": metadata.get("model_name", ""),
                        "streams": metadata.get("streams", ""),
                        "fusion": metadata.get("fusion", ""),
                        "head_type": metadata.get("head_type", ""),
                        "freeze_backbone": metadata.get("freeze_backbone", ""),
                        "lr": metadata.get("lr", ""),
                        "lr_base": f"{lr_base:.8f}",
                        "lr_head_rot": f"{lr_head_rot:.8f}",
                        "lr_head_y": f"{lr_head_y:.8f}",
                        "lr_head_z": f"{lr_head_z:.8f}",
                        "uncertainty_log_var_rot": "" if uncertainty_log_var_rot == "" else f"{uncertainty_log_var_rot:.8f}",
                        "uncertainty_log_var_trans": "" if uncertainty_log_var_trans == "" else f"{uncertainty_log_var_trans:.8f}",
                        "uncertainty_weight_rot": "" if uncertainty_weight_rot == "" else f"{uncertainty_weight_rot:.8f}",
                        "uncertainty_weight_trans": "" if uncertainty_weight_trans == "" else f"{uncertainty_weight_trans:.8f}",
                        "batch_size": metadata.get("batch_size", ""),
                        "seed": metadata.get("seed", ""),
                    }
                )
    finally:
        if csv_file is not None:
            csv_file.close()
        if tb_writer is not None:
            tb_writer.close()

def main(
    config_path,
    checkpointing=False,
    batch_size=None,
    optimizer=None,
    tensorboard_root_dir=None,
    tensorboard_port=6006,
):
  
    cfg = load_config(config_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    metrics_csv_path = log_file.replace(".log", "_metrics.csv")
    tb_root = tensorboard_root_dir or os.path.join("logs", "tensorboard")
    tensorboard_log_dir = os.path.join(tb_root, os.path.splitext(os.path.basename(log_file))[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess data (split and transform) if directory doesn't exist
    data_dir = os.path.join(base_dir, "../Data")
    preprocessed_dir = os.path.join(data_dir, "Data_split/train_preprocessed")
    if not os.path.exists(preprocessed_dir):
        preprocess_data(root_dir=data_dir)
    
    batch_size = batch_size if batch_size is not None else cfg["training"]["batch_size"]
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=os.path.join(data_dir, "Data_split"),
        batch_size=batch_size,
        shuffle=True
    )

    # time and date
    timestamp = time.strftime("%Y%m%d_%H%Mh")
    checkpoint_path = f"checkpoints/checkpoint_{cfg['model']['fusion']}_lr{cfg['training']['lr']}_batch{batch_size}_{timestamp}.pth"

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

    head_lr_cfg = cfg.get("training", {}).get("head_lr_decay", {})
    uncertainty_cfg = cfg.get("training", {}).get("uncertainty_weighting", {})
    use_uncertainty_weighting = bool(uncertainty_cfg.get("enabled", False))
    uncertainty_module = None
    base_lr = float(cfg["training"]["lr"])
    param_groups = _build_named_param_groups(model, base_lr=base_lr)
    resolved_head_lr_cfg = {
        "enabled": bool(head_lr_cfg.get("enabled", True)),
        "patience": int(head_lr_cfg.get("patience", 3)),
        "factor": float(head_lr_cfg.get("factor", 0.3)),
        "min_lr_mult": float(head_lr_cfg.get("min_lr_mult", 0.05)),
    }

    optimizer_name = optimizer if optimizer is not None else cfg["training"]["optimizer"]
    optimizer = get_optimizer(
        param_groups,
        optimizer_name=optimizer_name,
        lr=base_lr,
    )

    if use_uncertainty_weighting:
        uncertainty_module = UncertaintyWeighting(
            init_log_var_rot=float(uncertainty_cfg.get("init_log_var_rot", 0.0)),
            init_log_var_trans=float(uncertainty_cfg.get("init_log_var_trans", 0.0)),
        ).to(device)
        uncertainty_lr_mult = float(uncertainty_cfg.get("lr_mult", 1.0))
        optimizer.add_param_group(
            {
                "name": "uncertainty",
                "params": uncertainty_module.parameters(),
                "lr": base_lr * uncertainty_lr_mult,
            }
        )

    run_metadata = {
        "model_name": cfg["model"]["backbone"],
        "streams": cfg["model"]["streams"],
        "fusion": cfg["model"]["fusion"],
        "head_type": cfg["model"]["head_type"],
        "freeze_backbone": cfg["model"]["freeze_backbone"],
        "lr": cfg["training"]["lr"],
        "batch_size": batch_size,
        "seed": cfg.get("training", {}).get("seed", ""),
        "head_lr_decay": resolved_head_lr_cfg,
        "uncertainty_weighting": uncertainty_cfg if use_uncertainty_weighting else {"enabled": False},
    }

    # TensorBoard and CSV logging setup
    tb_port = tensorboard_port
    tensorboard_process = None

    logger.info(
        f"run_start log_file={log_file} metrics_csv={metrics_csv_path} "
        f"tensorboard_dir={tensorboard_log_dir} "
        f"model={run_metadata['model_name']} fusion={run_metadata['fusion']} "
        f"head_type={run_metadata['head_type']} batch_size={run_metadata['batch_size']} lr={run_metadata['lr']}"
    )
    if use_uncertainty_weighting:
        logger.info(
            "uncertainty_weighting enabled=true "
            f"init_log_var_rot={uncertainty_cfg.get('init_log_var_rot', 0.0)} "
            f"init_log_var_trans={uncertainty_cfg.get('init_log_var_trans', 0.0)} "
            f"lr_mult={uncertainty_cfg.get('lr_mult', 1.0)}"
        )
    if is_port_in_use(tb_port):
        logger.info(f"TensorBoard already running on port {tb_port}. Logging to existing instance.")
    else:
        # Pass the parent log directory so it sees all runs
        tensorboard_process = start_tensorboard(tb_root, port=tb_port)
        
    if tensorboard_process is not None:
        logger.info(f"tensorboard_start status=started run_dir={tensorboard_log_dir} url=http://127.0.0.1:{tb_port}")
    else:
        logger.info(f"tensorboard_start status=attached run_dir={tensorboard_log_dir} url=http://127.0.0.1:{tb_port}")


    # Training
    loss_function = total_loss

    try:
        train_loop(
            model=model,
            num_epochs=cfg["training"]["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=loss_function,
            device=device, 
            uncertainty_module=uncertainty_module,
            checkpointing=checkpointing,
            checkpoint_path=checkpoint_path,
            metrics_csv_path=metrics_csv_path,
            run_metadata=run_metadata,
            tensorboard_log_dir=tensorboard_log_dir,
            head_lr_decay_cfg=resolved_head_lr_cfg,
        )
    finally:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gated.yml", help="Path to config file")
    parser.add_argument("--batch_size", type=int, required=False, help="Override batch size from config")
    parser.add_argument("--optimizer", type=str, required=False, help="Override optimizer from config")
    parser.add_argument(
        "--tensorboard_root",
        type=str,
        default=None,
        help="Root TensorBoard log directory. If omitted, uses logs/tensorboard",
    )
    parser.add_argument(
        "--tensorboard_port",
        type=int,
        default=6006,
        help="TensorBoard port used for attach/start behavior",
    )

    args = parser.parse_args()

    main(
        args.config,
        checkpointing=False,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        tensorboard_root_dir=args.tensorboard_root,
        tensorboard_port=args.tensorboard_port,
    )