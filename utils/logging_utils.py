import csv
import logging
import os
import platform
import socket
import subprocess
import sys

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


METRICS_FIELDNAMES = [
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
]


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_tensorboard(logdir, logger, port=6006):
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
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }

    if platform.system() == "Windows":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        logger.info(f"Attempting to start TensorBoard at http://127.0.0.1:{port}")
        return subprocess.Popen(command, **kwargs)
    except Exception as exc:
        logger.warning(f"Could not start TensorBoard automatically: {exc}")
        return None


def format_fusion_weights(fusion_weights):
    if fusion_weights is None:
        return ""
    return "[" + ",".join(f"{weight:.4f}" for weight in fusion_weights.tolist()) + "]"


def compute_model_norms(model):
    grad_sq_sum = 0.0
    param_sq_sum = 0.0

    for param in model.parameters():
        param_sq_sum += param.detach().pow(2).sum().item()
        if param.grad is not None:
            grad_sq_sum += param.grad.detach().pow(2).sum().item()

    return param_sq_sum ** 0.5, grad_sq_sum ** 0.5


def init_metrics_csv(metrics_csv_path):
    if metrics_csv_path is None:
        return None, None

    csv_file = open(metrics_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=METRICS_FIELDNAMES)
    csv_writer.writeheader()
    return csv_file, csv_writer


def init_tensorboard_writer(tensorboard_log_dir, metadata, logger):
    if tensorboard_log_dir is None:
        return None

    if SummaryWriter is None:
        logger.warning("TensorBoard is unavailable because torch.utils.tensorboard could not be imported.")
        return None

    os.makedirs(tensorboard_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    if metadata:
        tb_writer.add_text(
            "run/metadata",
            "\n".join(f"{key}: {value}" for key, value in metadata.items()),
        )

    return tb_writer


def log_train_step_to_tensorboard(
    tb_writer,
    global_batch_step,
    optimizer,
    model,
    param_groups_by_name,
    loss,
    loss_q,
    loss_y,
    loss_z,
    abs_trans,
    abs_rot_x,
    fusion_weights,
):
    if tb_writer is None:
        return

    param_norm, grad_norm = compute_model_norms(model)
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


def log_epoch_to_tensorboard(
    tb_writer,
    global_step,
    epoch_avg_loss,
    val_loss_avg,
    avg_q,
    val_q_avg,
    avg_y,
    val_y_avg,
    avg_z,
    val_z_avg,
    train_abs_y,
    val_abs_y_avg,
    train_abs_z,
    val_abs_z_avg,
    train_abs_trans,
    val_abs_trans_avg,
    train_abs_rot_x,
    val_abs_rot_x_avg,
    best_val_loss,
    epoch_time,
    lr_base,
    lr_head_rot,
    lr_head_y,
    lr_head_z,
    param_groups_by_name,
    fusion_weights,
    fusion_weights_avg,
    uncertainty_module,
    uncertainty_log_var_rot,
    uncertainty_log_var_trans,
    uncertainty_weight_rot,
    uncertainty_weight_trans,
):
    if tb_writer is None:
        return

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
        tb_writer.add_text("fusion/weights", fusion_weights, global_step)

    if fusion_weights_avg is not None:
        for idx, weight in enumerate(fusion_weights_avg.tolist()):
            tb_writer.add_scalar(f"fusion/epoch_weight_stream/{idx}", float(weight), global_step)


def log_hparams_to_tensorboard(tb_writer, metadata, optimizer, train_loader, val_loss_avg, val_abs_trans_avg, val_abs_rot_x_avg):
    if tb_writer is None or not metadata:
        return

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


def write_epoch_metrics_csv(csv_writer, row):
    if csv_writer is None:
        return
    csv_writer.writerow(row)
