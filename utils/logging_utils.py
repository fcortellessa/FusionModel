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


def _to_float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_gate_statistics(weights_tensor):
    """
    Compute comprehensive gate statistics from a batch of fusion weights.

    Handles multiple tensor shapes (scalar, channel-wise, spatial):
    - [batch, num_streams]: scalar gating
    - [batch, num_streams, channels]: channel-wise gating
    - [batch, num_streams, height, width]: spatial gating (flattened)

    Args:
        weights_tensor: torch.Tensor or None — fusion gate weights

    Returns:
        dict with keys: mean, std, fraction_lt_0_1, fraction_gt_0_9, p25, p50, p75, p90
        All values are Python floats or None if input is None.
    """
    if weights_tensor is None:
        return {
            "mean": None, "std": None,
            "fraction_lt_0_1": None, "fraction_gt_0_9": None,
            "p25": None, "p50": None, "p75": None, "p90": None,
        }

    try:
        import numpy as np

        # Flatten to 1D for statistics
        w_flat = weights_tensor.flatten().detach().cpu().numpy()

        mean_val = float(np.mean(w_flat))
        std_val = float(np.std(w_flat))
        frac_lt_01 = float(np.mean(w_flat < 0.1))
        frac_gt_09 = float(np.mean(w_flat > 0.9))

        p25_val = float(np.percentile(w_flat, 25))
        p50_val = float(np.percentile(w_flat, 50))
        p75_val = float(np.percentile(w_flat, 75))
        p90_val = float(np.percentile(w_flat, 90))

        return {
            "mean": mean_val,
            "std": std_val,
            "fraction_lt_0_1": frac_lt_01,
            "fraction_gt_0_9": frac_gt_09,
            "p25": p25_val,
            "p50": p50_val,
            "p75": p75_val,
            "p90": p90_val,
        }
    except Exception as e:
        print(f"Warning: Failed to compute gate statistics: {e}")
        return {
            "mean": None, "std": None,
            "fraction_lt_0_1": None, "fraction_gt_0_9": None,
            "p25": None, "p50": None, "p75": None, "p90": None,
        }


METRICS_FIELDNAMES = [
    "timestamp", "epoch", 
    # losses and metrics 
    "train_loss", "val_loss", "loss_q", "loss_y", "loss_z",
    "val_loss_q", "val_loss_y", "val_loss_z",
    "train_abs_y", "train_abs_z", "train_abs_trans", "train_abs_rot_x",
    "train_signed_y", "train_signed_z", "train_signed_rot_x",
    "val_abs_y", "val_abs_z", "val_abs_trans", "val_abs_rot_x",
    "val_signed_y", "val_signed_z", "val_signed_rot_x",
    "val_rot_best_10_percent_deg", "val_rot_median_deg", "val_rot_p90_deg",
    "val_rot_x_best_10_percent_deg", "val_rot_x_median_deg", "val_rot_x_p90_deg",
    "fusion_weights", "train_fusion_gate_mean", "train_fusion_gate_std",
    "train_fusion_gate_fraction_lt_0_1", "train_fusion_gate_fraction_gt_0_9",
    "val_fusion_gate_mean", "val_fusion_gate_std",
    "val_fusion_gate_fraction_lt_0_1", "val_fusion_gate_fraction_gt_0_9",
    # per-task fusion gate parameters (for dual_fusion)
    "train_rotation_fusion_gate_mean", "train_rotation_fusion_gate_std",
    "train_rotation_fusion_gate_fraction_lt_0_1", "train_rotation_fusion_gate_fraction_gt_0_9",
    "val_rotation_fusion_gate_mean", "val_rotation_fusion_gate_std",
    "val_rotation_fusion_gate_fraction_lt_0_1", "val_rotation_fusion_gate_fraction_gt_0_9",
    "train_translation_fusion_gate_mean", "train_translation_fusion_gate_std",
    "train_translation_fusion_gate_fraction_lt_0_1", "train_translation_fusion_gate_fraction_gt_0_9",
    "val_translation_fusion_gate_mean", "val_translation_fusion_gate_std",
    "val_translation_fusion_gate_fraction_lt_0_1", "val_translation_fusion_gate_fraction_gt_0_9",
    "train_fusion_gate_p25", "train_fusion_gate_p50", "train_fusion_gate_p75", "train_fusion_gate_p90",
    "val_fusion_gate_p25", "val_fusion_gate_p50", "val_fusion_gate_p75", "val_fusion_gate_p90",
    # general run metadata
    "best_loss", "epoch_time_sec", "checkpoint_path",
    "model_name", "streams", "fusion", "head_type", "freeze_backbone",
    "lr", "lr_base", "lr_head_rot", "lr_head_y", "lr_head_z", "lr_schedule_scale",
    "uncertainty_log_var_rot", "uncertainty_log_var_trans",
    "uncertainty_weight_rot", "uncertainty_weight_trans", "batch_size", "seed",
    "num_parameters", "num_trainable_parameters", "gflops_per_sample", "gflops_per_batch", "avg_inference_time_ms"
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
        # Channel-wise gates may contain thousands of values per stream.  Log a
        # single stream mean per step instead of creating one TensorBoard event
        # per channel; detailed epoch statistics are saved to metrics.csv.
        weights_to_log = fusion_weights.detach()
        if weights_to_log.ndim > 1:
            weights_to_log = weights_to_log.mean(dim=tuple(range(1, weights_to_log.ndim)))
        for idx, weight in enumerate(weights_to_log.cpu().tolist()):
            tb_writer.add_scalar(f"fusion/weight_stream_{idx}", float(weight), global_batch_step)
            
    if hasattr(model, "rot_adapter_activations") and model.rot_adapter_activations is not None:
        mean_act = model.rot_adapter_activations.detach().mean()
        tb_writer.add_scalar("adapter/rot_mean", mean_act.item(), global_batch_step)
    
    if hasattr(model, "trans_adapter_activations") and model.trans_adapter_activations is not None:
        mean_act = model.trans_adapter_activations.detach().mean()
        tb_writer.add_scalar("adapter/trans_mean", mean_act.item(), global_batch_step)

    # Task-specific feature adaptation ("channel_attention" / "residual_gate",
    # selected via model.task_feature_adapter.type) is only active when the
    # relevant adapters were instantiated; both types share the same
    # rotation_attention/translation_attention tensors, so the same logging
    # code path covers both.
    if hasattr(model, "rotation_attention") and model.rotation_attention is not None:
        mean_attn = model.rotation_attention.detach().mean()
        tb_writer.add_scalar("task_feature_adapter/rotation_mean", mean_attn.item(), global_batch_step)

    if hasattr(model, "translation_attention") and model.translation_attention is not None:
        mean_attn = model.translation_attention.detach().mean()
        tb_writer.add_scalar("task_feature_adapter/translation_mean", mean_attn.item(), global_batch_step)


def log_epoch_to_tensorboard(
    tb_writer, global_step, train_avg, val_avg, lr_stats, lr_schedule_scale,
    param_groups_by_name, fusion_weights, fusion_weights_avg, uncertainty_module,
    uncertainty_stats, best_val_loss, epoch_time,
    train_fusion_weights_raw=None, val_fusion_weights_raw=None,
    train_rot_weights_raw=None, val_rot_weights_raw=None,
    train_trans_weights_raw=None, val_trans_weights_raw=None,
):
    """
    Log epoch metrics to TensorBoard.

    Additional arguments for raw fusion weights (for histogram logging):
    - train_fusion_weights_raw: [batch, num_streams] or [batch, num_streams, ...] raw gate values
    - val_fusion_weights_raw: same structure for validation
    - train_rot_weights_raw: rotation-specific weights (dual_fusion only)
    - val_rot_weights_raw: validation rotation weights
    - train_trans_weights_raw: translation-specific weights (dual_fusion only)
    - val_trans_weights_raw: validation translation weights
    """
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
        log_var_scalars = {
            key: value
            for key, value in {
                "rot": _to_float_or_none(uncertainty_stats["uncertainty_log_var_rot"]),
                "trans": _to_float_or_none(uncertainty_stats["uncertainty_log_var_trans"]),
            }.items()
            if value is not None
        }
        weight_scalars = {
            key: value
            for key, value in {
                "rot": _to_float_or_none(uncertainty_stats["uncertainty_weight_rot"]),
                "trans": _to_float_or_none(uncertainty_stats["uncertainty_weight_trans"]),
            }.items()
            if value is not None
        }
        if log_var_scalars:
            tb_writer.add_scalars("uncertainty/log_var", log_var_scalars, global_step)
        if weight_scalars:
            tb_writer.add_scalars("uncertainty/weight", weight_scalars, global_step)

    for group_name, group in param_groups_by_name.items():
        tb_writer.add_scalar(f"training/lr_groups/{group_name}", group["lr"], global_step)

    if fusion_weights is not None:
        tb_writer.add_text("fusion/weights", fusion_weights, global_step)

    if fusion_weights_avg is not None:
        for idx, weight in enumerate(fusion_weights_avg.tolist()):
            tb_writer.add_scalar(f"fusion/epoch_weight_stream/{idx}", float(weight), global_step)

    # GAP 2: Log full distribution histograms for fusion gate analysis
    # Option B: TensorBoard histograms show the complete distribution per epoch
    if train_fusion_weights_raw is not None:
        tb_writer.add_histogram("fusion/train_weights_distribution", train_fusion_weights_raw.flatten().detach().cpu(), global_step)
    if val_fusion_weights_raw is not None:
        tb_writer.add_histogram("fusion/val_weights_distribution", val_fusion_weights_raw.flatten().detach().cpu(), global_step)

    # GAP 1: Per-task fusion gate distributions (dual_fusion only)
    if train_rot_weights_raw is not None:
        tb_writer.add_histogram("fusion/train_rotation_weights_distribution", train_rot_weights_raw.flatten().detach().cpu(), global_step)
    if val_rot_weights_raw is not None:
        tb_writer.add_histogram("fusion/val_rotation_weights_distribution", val_rot_weights_raw.flatten().detach().cpu(), global_step)

    if train_trans_weights_raw is not None:
        tb_writer.add_histogram("fusion/train_translation_weights_distribution", train_trans_weights_raw.flatten().detach().cpu(), global_step)
    if val_trans_weights_raw is not None:
        tb_writer.add_histogram("fusion/val_translation_weights_distribution", val_trans_weights_raw.flatten().detach().cpu(), global_step)


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
            f"type={lr_schedule_cfg.get('type', 'warmup_cosine')} "
            f"start_lr_mult={lr_schedule_cfg.get('start_lr_mult', 0.2)} "
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
    return {k: 0.0 for k in ["loss", "loss_q", "loss_y", "loss_z", "abs_y", "abs_z", "abs_trans", "abs_rot_x", "signed_y", "signed_z", "signed_rot_x"]}


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
            rot_log_var = _to_float_or_none(uncertainty_stats['uncertainty_log_var_rot'])
            trans_log_var = _to_float_or_none(uncertainty_stats['uncertainty_log_var_trans'])
            rot_weight = _to_float_or_none(uncertainty_stats['uncertainty_weight_rot'])
            trans_weight = _to_float_or_none(uncertainty_stats['uncertainty_weight_trans'])
            if None in (rot_log_var, trans_log_var, rot_weight, trans_weight):
                return
            logger.info(
                f"uncertainty rot_log_var={rot_log_var:.6f} "
                f"trans_log_var={trans_log_var:.6f} "
                f"rot_weight={rot_weight:.6f} "
                f"trans_weight={trans_weight:.6f}"
            )
        except (ValueError, TypeError):
            pass
