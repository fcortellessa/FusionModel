# scripts/train.py

import os
import sys
import time
from pathlib import Path
import argparse
import logging
import random
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from utils.train_utils import (
    _build_epoch_lr_scale_fn,
    _get_lr_by_group,
    DataPipelineManager,
    TrainingConfigBuilder,
    RunSetupManager,
    HeadLRDecayManager,
    EarlyStoppingManager,
    EpochMetricsAggregator,
    build_epoch_metrics_row,
    _apply_metric_accumulation
)
from utils.utils import (
    load_config,
    save_checkpoint,
    get_optimizer,
)
from utils.logging_utils import (
    format_fusion_weights,
    init_metrics_csv,
    init_tensorboard_writer,
    log_train_step_to_tensorboard,
    log_epoch_to_tensorboard,
    log_hparams_to_tensorboard,
    configure_run_logging,
    log_epoch_summary,
    write_epoch_metrics_csv,
    _init_epoch_metrics,
    close_run_writers,
)
from losses.rotation_loss_handlers import get_rotation_handler
from utils.normalization import MEAN_Y, STD_Y, MEAN_Z, STD_Z

logger = logging.getLogger("GripFusionNet")
log_file = None


def set_global_seed(seed, deterministic=True):
    """Seed every RNG used by training before data loaders and models are built."""
    if seed is None or seed == "":
        return None
    if isinstance(seed, (list, tuple)):
        raise TypeError(
            "training.seed must be one integer per training run. "
            "Use a generator spec grid to expand a list of seeds into separate configs."
        )

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuBLAS requires this setting for deterministic CUDA matrix operations.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    return seed


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _compute_uncertainty_stats(uncertainty_module):
    if uncertainty_module is None:
        return {
            "uncertainty_log_var_rot": "",
            "uncertainty_log_var_trans": "",
            "uncertainty_weight_rot": "",
            "uncertainty_weight_trans": "",
        }

    # Prefer standardized get_stats() if available (wrapper added in train_utils)
    if hasattr(uncertainty_module, "get_stats") and callable(getattr(uncertainty_module, "get_stats")):
        try:
            stats = uncertainty_module.get_stats()
            return stats or {
                "uncertainty_log_var_rot": "",
                "uncertainty_log_var_trans": "",
                "uncertainty_weight_rot": "",
                "uncertainty_weight_trans": "",
            }
        except Exception:
            # fall back to attribute access
            pass

    # Backwards-compatible attribute access if module exposes log_var_*
    try:
        if hasattr(uncertainty_module, "get_stats") and callable(getattr(uncertainty_module, "get_stats")):
            stats = uncertainty_module.get_stats()
            if stats:
                return stats

        lv_rot = _to_float(uncertainty_module.log_var_rot)
        lv_trans = _to_float(uncertainty_module.log_var_trans)
        w_rot = float(torch.exp(-uncertainty_module.log_var_rot.detach()).item())
        w_trans = float(torch.exp(-uncertainty_module.log_var_trans.detach()).item())
        return {
            "uncertainty_log_var_rot": lv_rot,
            "uncertainty_log_var_trans": lv_trans,
            "uncertainty_weight_rot": w_rot,
            "uncertainty_weight_trans": w_trans,
        }
    except Exception:
        return {
            "uncertainty_log_var_rot": "",
            "uncertainty_log_var_trans": "",
            "uncertainty_weight_rot": "",
            "uncertainty_weight_trans": "",
        }


def _init_fusion_gate_stats():
    """Create accumulators for compact, per-stream gate diagnostics."""
    return {"sum": None, "sum_sq": None, "count": None, "lt_0_1": None, "gt_0_9": None}


def _accumulate_fusion_gate_stats(stats, gates):
    """Accumulate all gate values while keeping channel-wise gates out of the CSV."""
    if gates is None:
        return

    # Gate shapes are (B, streams) or (B, streams, channels).  Each row below
    # therefore represents every gate value belonging to one stream.
    by_stream = gates.detach().float().cpu().movedim(1, 0).reshape(gates.shape[1], -1)
    gate_sum = by_stream.sum(dim=1)
    gate_sum_sq = by_stream.square().sum(dim=1)
    gate_count = torch.full_like(gate_sum, by_stream.shape[1])
    low_count = (by_stream < 0.1).sum(dim=1)
    high_count = (by_stream > 0.9).sum(dim=1)

    if stats["sum"] is None:
        stats.update({
            "sum": gate_sum,
            "sum_sq": gate_sum_sq,
            "count": gate_count,
            "lt_0_1": low_count,
            "gt_0_9": high_count,
        })
        return

    for key, value in {
        "sum": gate_sum,
        "sum_sq": gate_sum_sq,
        "count": gate_count,
        "lt_0_1": low_count,
        "gt_0_9": high_count,
    }.items():
        stats[key] += value


def _summarize_fusion_gate_stats(stats, prefix):
    """Return CSV-ready, per-stream gate statistics as compact list strings."""
    if stats["sum"] is None:
        return {}, None

    mean = stats["sum"] / stats["count"]
    variance = (stats["sum_sq"] / stats["count"] - mean.square()).clamp_min(0.0)
    summary = {
        f"{prefix}_fusion_gate_mean": format_fusion_weights(mean),
        f"{prefix}_fusion_gate_std": format_fusion_weights(variance.sqrt()),
        f"{prefix}_fusion_gate_fraction_lt_0_1": format_fusion_weights(stats["lt_0_1"] / stats["count"]),
        f"{prefix}_fusion_gate_fraction_gt_0_9": format_fusion_weights(stats["gt_0_9"] / stats["count"]),
    }
    return summary, mean



def _train_single_epoch(
    model,
    train_loader,
    optimizer,
    rotation_handler,
    device,
    tb_writer,
    param_groups_by_name,
    uncertainty_module,
    global_batch_step,
    epoch,
    num_epochs,
    normalize_translation=True,
):
    model.train()
    train_metrics = _init_epoch_metrics()

    fusion_weights = None
    fusion_gate_stats = _init_fusion_gate_stats()

    tqdm_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)", leave=False, mininterval=1.0)

    for batch in tqdm_bar:
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        aux_losses = batch.pop("aux_losses", None)
        pose = batch["pose"].to(device)

        pose_pred, fusion_weights = model(batch, aux_losses=aux_losses)
        fusion_targets = model.last_fusion_targets

        loss, loss_q, loss_y, loss_z, _ = rotation_handler.compute_loss(
            pose_pred, pose, uncertainty_module
        )
        pose_pred_unnorm = {k: v.clone() for k, v in pose_pred.items()}
        if normalize_translation:
            pose_pred_unnorm["trans"][:, 1] = pose_pred_unnorm["trans"][:, 1] * STD_Y + MEAN_Y
            pose_pred_unnorm["trans"][:, 2] = pose_pred_unnorm["trans"][:, 2] * STD_Z + MEAN_Z

        pose_unnorm = pose.clone()
        if normalize_translation:
            pose_unnorm[:, 0] = pose_unnorm[:, 0] * STD_Y + MEAN_Y
            pose_unnorm[:, 1] = pose_unnorm[:, 1] * STD_Z + MEAN_Z

        abs_trans_y, abs_trans_z, abs_trans, abs_rot_x = rotation_handler.compute_metrics(
            pose_pred_unnorm, pose_unnorm
        )

        _accumulate_fusion_gate_stats(fusion_gate_stats, model.last_fusion_weights)

        _apply_metric_accumulation(
            train_metrics,
            {
                "loss": loss.item(),
                "loss_q": loss_q.item(),
                "loss_y": loss_y.item(),
                "loss_z": loss_z.item(),
                "abs_y": abs_trans_y.item(),
                "abs_z": abs_trans_z.item(),
                "abs_trans": abs_trans.item(),
                "abs_rot_x": abs_rot_x.item(),
            },
        )

        avg_loss_so_far = train_metrics["loss"] / (tqdm_bar.n + 1)
        tqdm_bar.set_postfix(
            avg_loss=f"{avg_loss_so_far:.4f}", 
            curr_loss=f"{loss.item():.4f}", 
            q_wt=f"{loss_q.item():.4f}"
        )

        loss.backward()

        log_train_step_to_tensorboard(
            tb_writer=tb_writer,
            global_batch_step=global_batch_step,
            optimizer=optimizer,
            model=model,
            param_groups_by_name=param_groups_by_name,
            loss=loss,
            loss_q=loss_q,
            loss_y=loss_y,
            loss_z=loss_z,
            abs_trans=abs_trans,
            abs_rot_x=abs_rot_x,
            fusion_weights=fusion_weights,
        )

        # Per-step uncertainty logging for debugging (if available)
        if tb_writer is not None and uncertainty_module is not None:
            try:
                stats = uncertainty_module.get_stats() if hasattr(uncertainty_module, "get_stats") else None
                if stats:
                    # log log-variances if present
                    lv_rot = stats.get("uncertainty_log_var_rot", None)
                    lv_trans = stats.get("uncertainty_log_var_trans", None)
                    if lv_rot is not None and lv_rot != "":
                        tb_writer.add_scalar("uncertainty/step_log_var_rot", float(lv_rot), global_batch_step)
                    if lv_trans is not None and lv_trans != "":
                        tb_writer.add_scalar("uncertainty/step_log_var_trans", float(lv_trans), global_batch_step)

                    # log weights if present
                    w_rot = stats.get("uncertainty_weight_rot", None)
                    w_trans = stats.get("uncertainty_weight_trans", None)
                    if w_rot is not None and w_rot != "":
                        tb_writer.add_scalar("uncertainty/step_weight_rot", float(w_rot), global_batch_step)
                    if w_trans is not None and w_trans != "":
                        tb_writer.add_scalar("uncertainty/step_weight_trans", float(w_trans), global_batch_step)
            except Exception:
                pass


        if tb_writer is not None and fusion_targets is not None:
            fusion_targets_mean = fusion_targets.detach().mean(dim=0).cpu().tolist()
            for idx, target in enumerate(fusion_targets_mean):
                tb_writer.add_scalar(f"fusion/target_stream_{idx}", float(target), global_batch_step)

        optimizer.step()
        global_batch_step += 1

    train_avg = EpochMetricsAggregator(train_metrics, len(train_loader)).to_dict()
    fusion_gate_summary, fusion_weights_avg = _summarize_fusion_gate_stats(
        fusion_gate_stats, prefix="train"
    )

    return {
        "train_avg": train_avg,
        "fusion_weights": fusion_weights,
        "fusion_weights_avg": fusion_weights_avg,
        "fusion_gate_summary": fusion_gate_summary,
        "global_batch_step": global_batch_step,
    }


def validation_loop(
    model,
    val_loader,
    rotation_handler,
    device,
    uncertainty_module=None,
    normalize_translation=True,
):
    """
    Run a single validation epoch.
    
    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        rotation_handler: RotationLossHandler instance for computing losses/metrics
        device: Device to run on
        uncertainty_module: Optional uncertainty weighting module
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    val_metrics = _init_epoch_metrics()
    fusion_gate_stats = _init_fusion_gate_stats()
    rotation_errors = []
    
    num_val = len(val_loader)
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            aux_losses = batch.pop("aux_losses", None)
            pose = batch["pose"].to(device)
            pose_pred, _ = model(batch, aux_losses=aux_losses)
            _accumulate_fusion_gate_stats(fusion_gate_stats, model.last_fusion_weights)
            
            # Compute loss and metrics using rotation handler
            loss, loss_q, loss_y, loss_z, loss_trans = rotation_handler.compute_loss(
                pose_pred, pose, uncertainty_module
            )
            
            pose_pred_unnorm = {k: v.clone() for k, v in pose_pred.items()}
            if normalize_translation:
                pose_pred_unnorm["trans"][:, 1] = pose_pred_unnorm["trans"][:, 1] * STD_Y + MEAN_Y
                pose_pred_unnorm["trans"][:, 2] = pose_pred_unnorm["trans"][:, 2] * STD_Z + MEAN_Z

            pose_unnorm = pose.clone()
            if normalize_translation:
                pose_unnorm[:, 0] = pose_unnorm[:, 0] * STD_Y + MEAN_Y
                pose_unnorm[:, 1] = pose_unnorm[:, 1] * STD_Z + MEAN_Z

            abs_trans_y, abs_trans_z, abs_trans, abs_rot_x = rotation_handler.compute_metrics(
                pose_pred_unnorm, pose_unnorm
            )
            rotation_errors.append(
                rotation_handler.compute_rotation_errors_degrees(pose_pred_unnorm, pose_unnorm).detach().cpu()
            )
            
            _apply_metric_accumulation(
                val_metrics,
                {
                    "loss": loss.item(),
                    "loss_q": loss_q.item(),
                    "loss_y": loss_y.item(),
                    "loss_z": loss_z.item(),
                    "abs_y": abs_trans_y.item(),
                    "abs_z": abs_trans_z.item(),
                    "abs_trans": abs_trans.item(),
                    "abs_rot_x": abs_rot_x.item(),
                },
            )
    
    val_gate_summary, _ = _summarize_fusion_gate_stats(fusion_gate_stats, prefix="val")
    all_rotation_errors = torch.cat(rotation_errors)
    rotation_summary = {
        "val_rot_median_deg": torch.median(all_rotation_errors).item(),
        "val_rot_p90_deg": torch.quantile(all_rotation_errors, 0.9).item(),
        "val_rot_within_5deg": (all_rotation_errors <= 5.0).float().mean().item(),
    }
    return EpochMetricsAggregator(val_metrics, num_val).to_dict(), val_gate_summary, rotation_summary


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
    checkpoint_path=None,
    metrics_csv_path=None,
    run_metadata=None,
    tensorboard_log_dir=None,
    head_lr_decay_cfg=None,
    lr_schedule_cfg=None,
    early_stopping_cfg=None,
    rotation_loss_type="geodesic",
    normalize_translation=True,
):
    best_val_loss = float("inf")
    csv_file = None
    csv_writer = None
    tb_writer = None
    metadata = run_metadata or {}
    global_batch_step = 0

    lr_scale_fn = _build_epoch_lr_scale_fn(num_epochs, lr_schedule_cfg)
    base_group = next((g for g in optimizer.param_groups if g.get("name") == "base"), optimizer.param_groups[0])
    base_lr = float(base_group["lr"])
    
    param_groups_by_name = {
        group.get("name", f"group_{idx}"): group
        for idx, group in enumerate(optimizer.param_groups)
    }
    
    # Initialize head LR decay manager
    head_lr_decay_cfg = head_lr_decay_cfg or {}
    head_lr_manager = HeadLRDecayManager(
        head_lr_decay_cfg=head_lr_decay_cfg,
        base_lr=base_lr,
        optimizer=optimizer,
        param_groups_by_name=param_groups_by_name,
    )

    # Initialize early stopping manager (no-op when training.early_stopping.enabled is false)
    early_stopper = EarlyStoppingManager(early_stopping_cfg)

    csv_file, csv_writer = init_metrics_csv(metrics_csv_path)
    tb_writer = init_tensorboard_writer(tensorboard_log_dir, metadata, logger)
    
    # Create rotation loss handler for both training and validation
    rotation_handler = get_rotation_handler(
        model.rotation_parametrisation,
        criterion=criterion,
        rotation_loss_type=rotation_loss_type
    )

    try:
        for epoch in range(last_epoch, num_epochs):
            current_lr_scale = lr_scale_fn(epoch)
            head_lr_manager.apply_lr_scale(current_lr_scale)

            epoch_start = time.perf_counter()
            train_epoch = _train_single_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                rotation_handler=rotation_handler,
                device=device,
                tb_writer=tb_writer,
                param_groups_by_name=param_groups_by_name,
                uncertainty_module=uncertainty_module,
                global_batch_step=global_batch_step,
                epoch=epoch,
                num_epochs=num_epochs,
                normalize_translation=normalize_translation,
            )
            global_batch_step = train_epoch["global_batch_step"]
            train_avg = train_epoch["train_avg"]
            fusion_weights = train_epoch["fusion_weights"]
            fusion_weights_avg = train_epoch["fusion_weights_avg"]
            train_gate_summary = train_epoch["fusion_gate_summary"]
            epoch_time = time.perf_counter() - epoch_start

            fw_str = format_fusion_weights(fusion_weights_avg) if fusion_weights_avg is not None else format_fusion_weights(fusion_weights)
            fw_epoch_str = fw_str

            val_avg, val_gate_summary, val_rotation_summary = validation_loop(
                model=model,
                val_loader=val_loader,
                rotation_handler=rotation_handler,
                device=device,
                uncertainty_module=uncertainty_module,
                normalize_translation=normalize_translation,
            )

            lr_base = _get_lr_by_group(optimizer, "base")
            lr_head_rot = _get_lr_by_group(optimizer, "head_rot")
            lr_head_y = _get_lr_by_group(optimizer, "head_y")
            lr_head_z = _get_lr_by_group(optimizer, "head_z")
            uncertainty_stats = _compute_uncertainty_stats(uncertainty_module)

            head_lr_manager.update(
                train_metrics={
                    "loss_q": train_avg["loss_q"],
                    "loss_y": train_avg["loss_y"],
                    "loss_z": train_avg["loss_z"],
                },
                val_metrics={
                    "loss_q": val_avg["loss_q"],
                    "loss_y": val_avg["loss_y"],
                    "loss_z": val_avg["loss_z"],
                },
                current_lr_scale=current_lr_scale,
                logger=logger,
            )

            if uncertainty_module is not None and hasattr(uncertainty_module, "update_epoch"):
                try:
                    uncertainty_module.update_epoch(
                        train_avg["loss_q"],
                        train_avg["loss_y"] + train_avg["loss_z"],
                    )
                except Exception as exc:
                    logger.debug(f"Task-weighting epoch update skipped: {exc}")

            if val_avg["loss"] < best_val_loss:
                best_val_loss = val_avg["loss"]
                save_checkpoint(model, optimizer, epoch, val_avg["loss"], checkpoint_path)

            should_stop = early_stopper.update(val_avg["loss"])

            log_epoch_summary(
                logger=logger,
                epoch=epoch,
                num_epochs=num_epochs,
                train_avg=train_avg,
                val_avg=val_avg,
                lr_base=lr_base,
                lr_head_rot=lr_head_rot,
                lr_head_y=lr_head_y,
                lr_head_z=lr_head_z,
                fusion_weights_str=fw_str,
                fusion_weights_epoch_str=fw_epoch_str,
                epoch_time=epoch_time,
                uncertainty_stats=uncertainty_stats,
                best_val_loss=best_val_loss,
            )

            global_step = epoch + 1
            log_epoch_to_tensorboard(
                tb_writer=tb_writer,
                global_step=global_step,
                train_avg=train_avg,
                val_avg=val_avg,
                lr_stats={
                    "base": lr_base,
                    "head_rot": lr_head_rot,
                    "head_y": lr_head_y,
                    "head_z": lr_head_z,
                },
                lr_schedule_scale=current_lr_scale,
                param_groups_by_name=param_groups_by_name,
                fusion_weights=fw_str,
                fusion_weights_avg=fusion_weights_avg,
                uncertainty_module=uncertainty_module,
                uncertainty_stats=uncertainty_stats,
                best_val_loss=best_val_loss,
                epoch_time=epoch_time,
            )

            if epoch == num_epochs - 1 or should_stop:
                log_hparams_to_tensorboard(
                    tb_writer=tb_writer,
                    metadata=metadata,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loss_avg=val_avg["loss"],
                    val_abs_trans_avg=val_avg["abs_trans"],
                    val_abs_rot_x_avg=val_avg["abs_rot_x"],
                )

            logger.debug(
                f"Epoch {epoch+1} detailed: q={train_avg['loss_q']:.4f}, y={train_avg['loss_y']:.4f}"
            )

            row = build_epoch_metrics_row(
                metadata={**metadata, **train_gate_summary, **val_gate_summary},
                epoch=epoch,
                train_avg=train_avg,
                val_avg=val_avg,
                fusion_weights_str=fw_str,
                best_val_loss=best_val_loss,
                epoch_time=epoch_time,
                lr_base=lr_base,
                lr_head_rot=lr_head_rot,
                lr_head_y=lr_head_y,
                lr_head_z=lr_head_z,
                current_lr_scale=current_lr_scale,
                uncertainty_stats=uncertainty_stats,
                validation_rotation_summary=val_rotation_summary,
                checkpoint_path=checkpoint_path,
            )
            write_epoch_metrics_csv(csv_writer, row)

            if should_stop:
                logger.info(
                    f"Early stopping at epoch {epoch + 1}: val_loss did not improve by >{early_stopper.min_delta} "
                    f"for {early_stopper.patience} consecutive epochs (best={early_stopper.best_value:.6f})."
                )
                break
    finally:
        close_run_writers(csv_file, tb_writer)


def main(
    config_path,
    batch_size=None,
    optimizer_name=None,
    tensorboard_root_dir=None,
    tensorboard_port=6006,
    force_preprocess=False,
):
    """Main training entry point."""
    global logger, log_file
    
    # Load config
    cfg = load_config(config_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    training_cfg = cfg.get("training", {})
    resolved_seed = set_global_seed(
        training_cfg.get("seed"),
        deterministic=bool(training_cfg.get("deterministic", True)),
    )
    
    # Resolve hyperparameter values
    resolved_batch_size = batch_size if batch_size is not None else cfg["training"]["batch_size"]
    cfg["training"]["batch_size"] = resolved_batch_size
    
    fusion_type = cfg["model"].get("fusion", "unknown_fusion")
    head_type = cfg["model"].get("head_type", "unknown_head")
    optimizer_name = optimizer_name if optimizer_name is not None else cfg["training"].get("optimizer", "adamw")

    # Create the nested run path hierarchy
    # This transforms your flat name into a clean directory tree
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nested_run_name = Path(fusion_type) / head_type / f"{optimizer_name}_{timestamp}_bs{resolved_batch_size}"
    
    # Setup run artifacts, logging, and tensorboard
    run_setup = RunSetupManager(config_path, cfg, resolved_batch_size)
    artifacts = run_setup.setup_artifacts()

    # Overwrite the default tensorboard target directory with nested layout
    tb_root = Path(tensorboard_root_dir or artifacts.get("tensorboard_root_dir", "runs"))
    tensorboard_log_dir = tb_root / nested_run_name

    run_metadata = {
        "model_name": cfg["model"]["backbone"],
        "streams": cfg["model"]["streams"],
        "fusion": fusion_type,
        "head_type": head_type,
        "freeze_backbone": cfg["model"]["freeze_backbone"],
        "lr": cfg["training"]["lr"],
        "batch_size": resolved_batch_size,
        "seed": cfg.get("training", {}).get("seed", ""),
    }

    logger, log_file = configure_run_logging(artifacts, str(tensorboard_log_dir.parent), run_metadata, logger=None, log_file=None)
    run_setup.logger = logger  # Update logger after creation
    if resolved_seed is not None:
        logger.info(
            "Global RNG seed set to %s (deterministic=%s)",
            resolved_seed,
            bool(training_cfg.get("deterministic", True)),
        )
    
    tensorboard_log_dir, _ = run_setup.setup_tensorboard(tensorboard_port, tensorboard_root_dir)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    data_pipeline = DataPipelineManager(cfg, base_dir, force_preprocess=force_preprocess, logger=logger)
    train_loader, val_loader, _ = data_pipeline.get_loaders(resolved_batch_size)
    
    # Build training components from config
    config_builder = TrainingConfigBuilder(cfg, device, logger=logger)
    model, optimizer, uncertainty_module, loss_function, param_groups = config_builder.build()
    
    # Override optimizer if specified via CLI
    if optimizer_name is not None:
        base_lr = float(cfg["training"]["lr"])
        optimizer = get_optimizer(param_groups, optimizer_name=optimizer_name, lr=base_lr)
        if uncertainty_module is not None:
            task_weighting_cfg = cfg.get("training", {}).get("task_weighting", cfg.get("training", {}).get("uncertainty_weighting", {}))
            uncertainty_lr_mult = float(task_weighting_cfg.get("lr_mult", 1.0))
            optimizer.add_param_group({
                "name": "uncertainty",
                "params": uncertainty_module.parameters(),
                "lr": base_lr * uncertainty_lr_mult,
            })
    
    # Prepare training metadata
    head_lr_cfg = cfg.get("training", {}).get("head_lr_decay", {})
    lr_schedule_cfg = cfg.get("training", {}).get("lr_schedule", {})
    early_stopping_cfg = cfg.get("training", {}).get("early_stopping", {})
    task_weighting_cfg = cfg.get("training", {}).get("task_weighting", cfg.get("training", {}).get("uncertainty_weighting", {}))
    
    resolved_head_lr_cfg = {
        "enabled": bool(head_lr_cfg.get("enabled", True)),
        "patience": int(head_lr_cfg.get("patience", 3)),
        "factor": float(head_lr_cfg.get("factor", 0.3)),
        "min_lr_mult": float(head_lr_cfg.get("min_lr_mult", 0.05)),
    }
    
    run_metadata = {
        "model_name": cfg["model"]["backbone"],
        "streams": cfg["model"]["streams"],
        "fusion": cfg["model"]["fusion"],
        "head_type": cfg["model"]["head_type"],
        "freeze_backbone": cfg["model"]["freeze_backbone"],
        "lr": cfg["training"]["lr"],
        "batch_size": resolved_batch_size,
        "seed": cfg.get("training", {}).get("seed", ""),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "head_lr_decay": resolved_head_lr_cfg,
        "lr_schedule": lr_schedule_cfg if lr_schedule_cfg else {"enabled": False},
        "task_weighting": task_weighting_cfg if uncertainty_module is not None else {"enabled": False},
        "early_stopping": early_stopping_cfg if early_stopping_cfg else {"enabled": False},
    }
    
    # Save manifest
    run_setup.save_manifest(run_metadata)
    
    # Get rotation loss type for 2D branch
    rotation_loss_type = cfg.get("model", {}).get("rotation_loss_type", "geodesic")

    # Train
    train_loop(
        model=model,
        num_epochs=cfg["training"]["epochs"],
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=loss_function,
        device=device,
        uncertainty_module=uncertainty_module,
        checkpoint_path=artifacts["checkpoint_path"],
        metrics_csv_path=artifacts["metrics_csv_path"],
        run_metadata=run_metadata,
        tensorboard_log_dir=tensorboard_log_dir,
        head_lr_decay_cfg=resolved_head_lr_cfg,
        lr_schedule_cfg=lr_schedule_cfg,
        early_stopping_cfg=early_stopping_cfg,
        rotation_loss_type=rotation_loss_type,
        normalize_translation=cfg.get("data", {}).get("normalize_translation", True),
    )


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
    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Force rebuild of Data_split and *_preprocessed folders before training",
    )

    args = parser.parse_args()

    main(
        args.config,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer,
        tensorboard_root_dir=args.tensorboard_root,
        tensorboard_port=args.tensorboard_port,
        force_preprocess=args.force_preprocess,
    )
