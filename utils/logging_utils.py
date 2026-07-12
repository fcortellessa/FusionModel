from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path
import platform
import socket
import subprocess
import sys
from utils.utils import setup_logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


METRICS_FIELDNAMES = [
    "timestamp", "epoch", "train_loss", "loss_q", "loss_y", "loss_z",
    "val_loss_q", "val_loss_y", "val_loss_z",
    "train_abs_y", "train_abs_z", "train_abs_trans", "train_abs_rot_x",
    "val_abs_y", "val_abs_z", "val_abs_trans", "val_abs_rot_x",
    "val_rot_median_deg", "val_rot_p90_deg", "val_rot_within_5deg",
    "fusion_weights", "train_fusion_gate_mean", "train_fusion_gate_std",
    "train_fusion_gate_fraction_lt_0_1", "train_fusion_gate_fraction_gt_0_9",
    "val_fusion_gate_mean", "val_fusion_gate_std",
    "val_fusion_gate_fraction_lt_0_1", "val_fusion_gate_fraction_gt_0_9",
    "best_loss", "epoch_time_sec", "checkpoint_path",
    "model_name", "streams", "fusion", "head_type", "freeze_backbone",
    "lr", "lr_base", "lr_head_rot", "lr_head_y", "lr_head_z", "lr_schedule_scale",
    "uncertainty_log_var_rot", "uncertainty_log_var_trans",
    "uncertainty_weight_rot", "uncertainty_weight_trans", "batch_size", "seed",
    "num_parameters", "num_trainable_parameters",
]


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_tensorboard(logdir, logger, port=6006):
    if SummaryWriter is None:
        logger.warning("TensorBoard logging is unavailable because torch.utils.tensorboard could not be imported.")
        return None

    command = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(logdir),
        "--port", str(port),
        "--host", "127.0.0.1",
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


def format_fusion_weights(fusion_weights) -> str:
    if fusion_weights is None:
        return ""
    return "[" + ",".join(f"{w:.4f}" for w in fusion_weights.tolist()) + "]"


def compute_model_norms(model) -> tuple[float, float]:
    param_sq_sum = 0.0
    grad_sq_sum = 0.0

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

    Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    if metadata:
        meta_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        tb_writer.add_text("run/metadata", meta_str)

    return tb_writer


def log_train_step_to_tensorboard(
    tb_writer, global_batch_step, optimizer, model, param_groups_by_name,
    loss, loss_q, loss_y, loss_z, abs_trans, abs_rot_x, fusion_weights,
):
    if tb_writer is None:
        return

    param_norm, grad_norm = compute_model_norms(model)
    
    scalars = {
        "loss_total": loss.item(),
        "loss_q": loss_q.item(),
        "loss_y": loss_y.item(),
        "loss_z": loss_z.item(),
        "abs_trans": abs_trans.item(),
        "abs_rot_x": abs_rot_x.item(),
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "lr": optimizer.param_groups[0]["lr"],
    }
    for tag, val in scalars.items():
        tb_writer.add_scalar(f"train_step/{tag}", val, global_batch_step)

    for group_name, group in param_groups_by_name.items():
        tb_writer.add_scalar(f"train_step/lr_{group_name}", group["lr"], global_batch_step)

    if fusion_weights is not None:
        for idx, weight in enumerate(fusion_weights.detach().cpu().flatten().tolist()):
            tb_writer.add_scalar(f"fusion/weight_stream_{idx}", float(weight), global_batch_step)


def log_epoch_to_tensorboard(
    tb_writer, global_step, train_avg, val_avg, lr_stats, lr_schedule_scale,
    param_groups_by_name, fusion_weights, fusion_weights_avg, uncertainty_module,
    uncertainty_stats, best_val_loss, epoch_time,
):
    if tb_writer is None:
        return

    # Bulk log train/val pairings
    metrics_to_log = ["loss", "loss_q", "loss_y", "loss_z", "abs_y", "abs_z", "abs_trans", "abs_rot_x"]
    for metric in metrics_to_log:
        folder = "loss" if "loss" in metric else "metrics"
        tb_writer.add_scalars(f"{folder}/{metric}", {"train": train_avg[metric], "val": val_avg[metric]}, global_step)

    # Bulk log training/validation performance gaps
    gaps = ["loss", "abs_y", "abs_z", "abs_trans", "abs_rot_x"]
    for m in gaps:
        tb_writer.add_scalar(f"diagnostics/{m}_gap", train_avg[m] - val_avg[m], global_step)

    tb_writer.add_scalar("training/best_val_loss", best_val_loss, global_step)
    tb_writer.add_scalar("training/epoch_time_sec", epoch_time, global_step)
    tb_writer.add_scalars("training/lr", {**lr_stats, "schedule_scale": lr_schedule_scale}, global_step)

    if uncertainty_module is not None and uncertainty_stats:
        tb_writer.add_scalars(
            "uncertainty/log_var",
            {"rot": uncertainty_stats["uncertainty_log_var_rot"], "trans": uncertainty_stats["uncertainty_log_var_trans"]},
            global_step,
        )
        tb_writer.add_scalars(
            "uncertainty/weight",
            {"rot": uncertainty_stats["uncertainty_weight_rot"], "trans": uncertainty_stats["uncertainty_weight_trans"]},
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
    if csv_writer is not None:
        csv_writer.writerow(row)


def close_run_writers(csv_file, tb_writer):
    if csv_file is not None:
        csv_file.close()

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


def log_run_start(logger, log_file, artifacts, tensorboard_log_dir, run_metadata, uncertainty_enabled, uncertainty_cfg, lr_schedule_cfg):
    logger.info(
        f"run_start log_file={log_file} metrics_csv={artifacts.get('metrics_csv_path')} "
        f"tensorboard_dir={tensorboard_log_dir} "
        f"model={run_metadata.get('model_name')} fusion={run_metadata.get('fusion')} "
        f"head_type={run_metadata.get('head_type')} batch_size={run_metadata.get('batch_size')} lr={run_metadata.get('lr')}"
    )

    if uncertainty_enabled:
        weighting_type = str(uncertainty_cfg.get('type', 'uncertainty')).lower()
        if weighting_type == 'dwa':
            logger.info(
                "task_weighting enabled=true "
                f"type=dwa dwa_temperature={uncertainty_cfg.get('dwa_temperature', 2.0)}"
            )
        elif weighting_type == 'gradnorm':
            logger.info(
                "task_weighting enabled=true "
                f"type=gradnorm gradnorm_lr={uncertainty_cfg.get('gradnorm_lr', 1e-4)} "
                f"gradnorm_alpha={uncertainty_cfg.get('gradnorm_alpha', 0.0)} "
                f"lr_mult={uncertainty_cfg.get('lr_mult', 1.0)}"
            )
        else:
            logger.info(
                "task_weighting enabled=true "
                f"type=uncertainty init_log_var_rot={uncertainty_cfg.get('init_log_var_rot', 0.0)} "
                f"init_log_var_trans={uncertainty_cfg.get('init_log_var_trans', 0.0)} "
                f"lr_mult={uncertainty_cfg.get('lr_mult', 1.0)}"
            )

    if bool(lr_schedule_cfg.get("enabled", False)):
        logger.info(
            "lr_schedule enabled=true "
            f"type={lr_schedule_cfg.get('type', 'one_cycle')} "
            f"start_lr_mult={lr_schedule_cfg.get('start_lr_mult', 0.2)} "
            f"max_lr_mult={lr_schedule_cfg.get('max_lr_mult', 3.0)} "
            f"final_lr_mult={lr_schedule_cfg.get('final_lr_mult', 0.05)}"
        )


def configure_run_logging(artifacts, tensorboard_root, run_metadata, uncertainty_enabled=False, uncertainty_cfg=None, lr_schedule_cfg=None, logger=None, log_file=None):
    if logger is None or log_file is None:
        logger, log_file = setup_logger(artifacts.get("run_dir", "logs"))

    tb_root = tensorboard_root or artifacts.get("tensorboard_root_dir")
    tensorboard_log_dir = Path(tb_root) / artifacts.get("run_name") if tb_root is not None else None

    log_run_start(
        logger=logger,
        log_file=log_file,
        artifacts=artifacts,
        tensorboard_log_dir=tensorboard_log_dir,
        run_metadata=run_metadata,
        uncertainty_enabled=uncertainty_enabled,
        uncertainty_cfg=uncertainty_cfg or {},
        lr_schedule_cfg=lr_schedule_cfg or {},
    )

    return logger, log_file


def _init_epoch_metrics():
    return {k: 0.0 for k in ["loss", "loss_q", "loss_y", "loss_z", "abs_y", "abs_z", "abs_trans", "abs_rot_x"]}


def log_epoch_summary(
    logger, epoch, num_epochs, train_avg, val_avg, lr_base, lr_head_rot,
    lr_head_y, lr_head_z, fusion_weights_str, fusion_weights_epoch_str,
    epoch_time, uncertainty_stats, best_val_loss,
):
    logger.info(
        f"epoch={epoch+1} train_loss={train_avg['loss']:.4f} loss_q={train_avg['loss_q']:.4f} "
        f"loss_y={train_avg['loss_y']:.4f} loss_z={train_avg['loss_z']:.4f} "
        f"train_abs_trans={train_avg['abs_trans']:.4f} train_abs_rot_x={train_avg['abs_rot_x']:.4f} "
        f"fusion_weights={fusion_weights_epoch_str} epoch_time_sec={epoch_time:.2f}"
    )

    logger.info(
        f"epoch={epoch+1}/{num_epochs} train_loss={train_avg['loss']:.4f} val_loss={val_avg['loss']:.4f} "
        f"train_abs_y={train_avg['abs_y']:.4f} train_abs_z={train_avg['abs_z']:.4f} "
        f"train_abs_trans={train_avg['abs_trans']:.4f} train_abs_rot_x={train_avg['abs_rot_x']:.4f} "
        f"val_abs_y={val_avg['abs_y']:.4f} val_abs_z={val_avg['abs_z']:.4f} "
        f"val_abs_trans={val_avg['abs_trans']:.4f} val_abs_rot_x={val_avg['abs_rot_x']:.4f} "
        f"lr_base={lr_base:.8f} lr_head_rot={lr_head_rot:.8f} lr_head_y={lr_head_y:.8f} "
        f"lr_head_z={lr_head_z:.8f} fusion_weights={fusion_weights_str}"
    )

    if uncertainty_stats:
        try:
            logger.info(
                f"uncertainty rot_log_var={float(uncertainty_stats['uncertainty_log_var_rot']):.6f} "
                f"trans_log_var={float(uncertainty_stats['uncertainty_log_var_trans']):.6f} "
                f"rot_weight={float(uncertainty_stats['uncertainty_weight_rot']):.6f} "
                f"trans_weight={float(uncertainty_stats['uncertainty_weight_trans']):.6f}"
            )
        except (ValueError, TypeError):
            pass
