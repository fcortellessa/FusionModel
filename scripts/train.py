# scripts/train.py

import os
import sys
import time
import argparse
import logging
import torch
import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from base_model.pose_estimation_model import PoseEstimationNet
from data_processing.preprocess_data import preprocess_data
from losses.se2_losses import geodesic_loss_2d_unit_vector, normalized_mse_loss_2d_unit_vector, wrapped_angle_loss, axis_angle_to_quaternion, quaternion_to_2d_unit_vector
from losses.se3_losses import se3_direct_loss, rotation_vector_to_quaternion, quaternion_to_rotation_vector
from losses.absolute_pose_error import absolute_pose_error_x_rot
from losses.original_loss import total_loss
from losses.uncertainty_weighting import UncertaintyWeighting
from utils.train_utils import (
    _build_epoch_lr_scale_fn, _build_named_param_groups, _get_lr_by_group,
    DataPipelineManager, TrainingConfigBuilder, RunSetupManager, HeadLRDecayManager, EpochMetricsAggregator 
)
from utils.utils import (
    setup_logger,
    load_config,
    save_checkpoint,
    load_checkpoint,
    get_optimizer,
    build_run_artifacts,
    save_yaml,
    save_json,
)
from utils.logging_utils import (
    is_port_in_use,
    start_tensorboard,
    format_fusion_weights,
    init_metrics_csv,
    init_tensorboard_writer,
    log_train_step_to_tensorboard,
    log_epoch_to_tensorboard,
    log_hparams_to_tensorboard,
    write_epoch_metrics_csv,
)

from data_processing.dataloader_preprocessed import get_dataloaders
from losses.rotation_loss_handlers import get_rotation_handler

logger = logging.getLogger("GripFusionNet")
log_file = None


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def validation_loop(
    model,
    val_loader,
    rotation_handler,
    device,
    uncertainty_module=None,
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
    val_metrics = {
        "loss": 0.0,
        "loss_q": 0.0,
        "loss_y": 0.0,
        "loss_z": 0.0,
        "abs_y": 0.0,
        "abs_z": 0.0,
        "abs_trans": 0.0,
        "abs_rot_x": 0.0,
    }
    
    num_val = len(val_loader)
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pose = batch["pose"].to(device)
            pose_pred, _ = model(batch)
            
            # Compute loss and metrics using rotation handler
            loss, loss_q, loss_y, loss_z, loss_trans = rotation_handler.compute_loss(
                pose_pred, pose, uncertainty_module
            )
            abs_trans_y, abs_trans_z, abs_trans, abs_rot_x = rotation_handler.compute_metrics(
                pose_pred, pose
            )
            
            # Accumulate metrics
            val_metrics["loss"] += loss.item()
            val_metrics["loss_q"] += loss_q.item()
            val_metrics["loss_y"] += loss_y.item()
            val_metrics["loss_z"] += loss_z.item()
            val_metrics["abs_y"] += abs_trans_y.item()
            val_metrics["abs_z"] += abs_trans_z.item()
            val_metrics["abs_trans"] += abs_trans.item()
            val_metrics["abs_rot_x"] += abs_rot_x.item()
    
    # Average metrics
    val_metrics_avg = {key: val / num_val for key, val in val_metrics.items()}
    return val_metrics_avg


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
    lr_schedule_cfg=None,
    rotation_loss_type="geodesic",
):
    best_val_loss = float("inf")
    csv_file = None
    csv_writer = None
    tb_writer = None
    metadata = run_metadata or {}
    global_batch_step = 0
    saved_checkpoint_path = ""

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

    csv_file, csv_writer = init_metrics_csv(metrics_csv_path)
    tb_writer = init_tensorboard_writer(tensorboard_log_dir, metadata, logger)
    
    # Create rotation loss handler for both training and validation
    rotation_handler = get_rotation_handler(
        getattr(model, "rotation_parametrisation", "quaternion"),
        criterion=criterion,
        rotation_loss_type=rotation_loss_type
    )

    try:
        for epoch in range(last_epoch, num_epochs):
            current_lr_scale = lr_scale_fn(epoch)
            head_lr_manager.apply_lr_scale(current_lr_scale)

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

                # Compute loss and metrics using rotation handler
                loss, loss_q, loss_y, loss_z, loss_trans = rotation_handler.compute_loss(
                    pose_pred, pose, uncertainty_module
                )
                abs_trans_y, abs_trans_z, abs_trans, abs_rot_x = rotation_handler.compute_metrics(
                    pose_pred, pose
                )

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

                optimizer.step()
                global_batch_step += 1

                tqdm_bar.set_postfix(loss=f"{loss.item():.4f}")

            num_train = len(train_loader)
            num_val = len(val_loader)
            fw_str = format_fusion_weights(fusion_weights)

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

            fw_epoch_str = format_fusion_weights(fusion_weights_avg) if fusion_weights_avg is not None else fw_str

            logger.info(
                f"epoch={epoch+1} train_loss={epoch_avg_loss:.4f} loss_q={avg_q:.4f} "
                f"loss_y={avg_y:.4f} loss_z={avg_z:.4f} train_abs_trans={train_abs_trans:.4f} "
                f"train_abs_rot_x={train_abs_rot_x:.4f} fusion_weights={fw_epoch_str} epoch_time_sec={epoch_time:.2f}"
            )

            # Validation phase - uses separate function to avoid duplication
            val_metrics_avg = validation_loop(
                model=model,
                val_loader=val_loader,
                rotation_handler=rotation_handler,
                device=device,
                uncertainty_module=uncertainty_module,
            )
            
            val_loss_avg = val_metrics_avg["loss"]
            val_q_avg = val_metrics_avg["loss_q"]
            val_y_avg = val_metrics_avg["loss_y"]
            val_z_avg = val_metrics_avg["loss_z"]
            val_abs_y_avg = val_metrics_avg["abs_y"]
            val_abs_z_avg = val_metrics_avg["abs_z"]
            val_abs_trans_avg = val_metrics_avg["abs_trans"]
            val_abs_rot_x_avg = val_metrics_avg["abs_rot_x"]

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

            # Update head learning rates based on validation performance
            head_lr_manager.update(
                train_metrics={"loss_q": avg_q, "loss_y": avg_y, "loss_z": avg_z},
                val_metrics={"loss_q": val_q_avg, "loss_y": val_y_avg, "loss_z": val_z_avg},
                current_lr_scale=current_lr_scale,
                logger=logger,
            )

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
                f"train_abs_y={train_abs_y:.4f} train_abs_z={train_abs_z:.4f} "
                f"train_abs_trans={train_abs_trans:.4f} train_abs_rot_x={train_abs_rot_x:.4f} "
                f"val_abs_y={val_abs_y_avg:.4f} val_abs_z={val_abs_z_avg:.4f} "
                f"val_abs_trans={val_abs_trans_avg:.4f} val_abs_rot_x={val_abs_rot_x_avg:.4f} lr_base={lr_base:.8f} "
                f"lr_head_rot={lr_head_rot:.8f} lr_head_y={lr_head_y:.8f} "
                f"lr_head_z={lr_head_z:.8f} fusion_weights={fw_str}"
            )
            if uncertainty_module is not None:
                logger.info(
                    f"uncertainty rot_log_var={uncertainty_log_var_rot:.6f} trans_log_var={uncertainty_log_var_trans:.6f} "
                    f"rot_weight={uncertainty_weight_rot:.6f} trans_weight={uncertainty_weight_trans:.6f}"
                )

            global_step = epoch + 1
            log_epoch_to_tensorboard(
                tb_writer=tb_writer,
                global_step=global_step,
                epoch_avg_loss=epoch_avg_loss,
                val_loss_avg=val_loss_avg,
                avg_q=avg_q,
                val_q_avg=val_q_avg,
                avg_y=avg_y,
                val_y_avg=val_y_avg,
                avg_z=avg_z,
                val_z_avg=val_z_avg,
                train_abs_y=train_abs_y,
                val_abs_y_avg=val_abs_y_avg,
                train_abs_z=train_abs_z,
                val_abs_z_avg=val_abs_z_avg,
                train_abs_trans=train_abs_trans,
                val_abs_trans_avg=val_abs_trans_avg,
                train_abs_rot_x=train_abs_rot_x,
                val_abs_rot_x_avg=val_abs_rot_x_avg,
                best_val_loss=best_val_loss,
                epoch_time=epoch_time,
                lr_base=lr_base,
                lr_head_rot=lr_head_rot,
                lr_head_y=lr_head_y,
                lr_head_z=lr_head_z,
                lr_schedule_scale=current_lr_scale,
                param_groups_by_name=param_groups_by_name,
                fusion_weights=fw_str,
                fusion_weights_avg=fusion_weights_avg,
                uncertainty_module=uncertainty_module,
                uncertainty_log_var_rot=uncertainty_log_var_rot,
                uncertainty_log_var_trans=uncertainty_log_var_trans,
                uncertainty_weight_rot=uncertainty_weight_rot,
                uncertainty_weight_trans=uncertainty_weight_trans,
            )

            if epoch == num_epochs - 1:
                log_hparams_to_tensorboard(
                    tb_writer=tb_writer,
                    metadata=metadata,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loss_avg=val_loss_avg,
                    val_abs_trans_avg=val_abs_trans_avg,
                    val_abs_rot_x_avg=val_abs_rot_x_avg,
                )

            logger.debug(f"Epoch {epoch+1} detailed: q={avg_q:.4f}, y={avg_y:.4f}")

            write_epoch_metrics_csv(
                csv_writer,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch + 1,
                    "train_loss": f"{epoch_avg_loss:.6f}",
                    "loss_q": f"{avg_q:.6f}",
                    "loss_y": f"{avg_y:.6f}",
                    "loss_z": f"{avg_z:.6f}",
                    "train_abs_y": f"{train_abs_y:.6f}",
                    "train_abs_z": f"{train_abs_z:.6f}",
                    "train_abs_trans": f"{train_abs_trans:.6f}",
                    "train_abs_rot_x": f"{train_abs_rot_x:.6f}",
                    "val_abs_y": f"{val_abs_y_avg:.6f}",
                    "val_abs_z": f"{val_abs_z_avg:.6f}",
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
                    "lr_schedule_scale": f"{current_lr_scale:.8f}",
                    "uncertainty_log_var_rot": "" if uncertainty_log_var_rot == "" else f"{uncertainty_log_var_rot:.8f}",
                    "uncertainty_log_var_trans": "" if uncertainty_log_var_trans == "" else f"{uncertainty_log_var_trans:.8f}",
                    "uncertainty_weight_rot": "" if uncertainty_weight_rot == "" else f"{uncertainty_weight_rot:.8f}",
                    "uncertainty_weight_trans": "" if uncertainty_weight_trans == "" else f"{uncertainty_weight_trans:.8f}",
                    "batch_size": metadata.get("batch_size", ""),
                    "seed": metadata.get("seed", ""),
                },
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
    
    # Resolve batch size
    batch_size = batch_size if batch_size is not None else cfg["training"]["batch_size"]
    cfg["training"]["batch_size"] = batch_size
    
    # Setup run artifacts, logging, and tensorboard
    run_setup = RunSetupManager(config_path, cfg, batch_size)
    artifacts = run_setup.setup_artifacts()
    logger, log_file = setup_logger(artifacts["run_dir"])
    run_setup.logger = logger  # Update logger after creation
    
    tensorboard_log_dir, _ = run_setup.setup_tensorboard(tensorboard_port, tensorboard_root_dir)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    data_pipeline = DataPipelineManager(cfg, base_dir, force_preprocess=force_preprocess, logger=logger)
    train_loader, val_loader, test_loader = data_pipeline.get_loaders(batch_size)
    
    # Build training components from config
    config_builder = TrainingConfigBuilder(cfg, device, logger=logger)
    model, optimizer, uncertainty_module, loss_function, param_groups = config_builder.build()
    
    # Override optimizer if specified via CLI
    if optimizer_name is not None:
        from utils.utils import get_optimizer
        base_lr = float(cfg["training"]["lr"])
        optimizer = get_optimizer(param_groups, optimizer_name=optimizer_name, lr=base_lr)
        if uncertainty_module is not None:
            uncertainty_lr_mult = float(cfg.get("training", {}).get("uncertainty_weighting", {}).get("lr_mult", 1.0))
            optimizer.add_param_group({
                "name": "uncertainty",
                "params": uncertainty_module.parameters(),
                "lr": base_lr * uncertainty_lr_mult,
            })
    
    # Prepare training metadata
    head_lr_cfg = cfg.get("training", {}).get("head_lr_decay", {})
    lr_schedule_cfg = cfg.get("training", {}).get("lr_schedule", {})
    uncertainty_cfg = cfg.get("training", {}).get("uncertainty_weighting", {})
    
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
        "batch_size": batch_size,
        "seed": cfg.get("training", {}).get("seed", ""),
        "head_lr_decay": resolved_head_lr_cfg,
        "lr_schedule": lr_schedule_cfg if lr_schedule_cfg else {"enabled": False},
        "uncertainty_weighting": uncertainty_cfg if uncertainty_module is not None else {"enabled": False},
    }
    
    # Log configuration
    logger.info(
        f"run_start log_file={log_file} metrics_csv={artifacts['metrics_csv_path']} "
        f"tensorboard_dir={tensorboard_log_dir} "
        f"model={run_metadata['model_name']} fusion={run_metadata['fusion']} "
        f"head_type={run_metadata['head_type']} batch_size={run_metadata['batch_size']} lr={run_metadata['lr']}"
    )
    if uncertainty_module is not None:
        logger.info(
            "uncertainty_weighting enabled=true "
            f"init_log_var_rot={uncertainty_cfg.get('init_log_var_rot', 0.0)} "
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
    
    # Save manifest
    run_setup.save_manifest(run_metadata)
    
    # Get rotation loss type for 2D branch
    rotation_loss_type = cfg.get("model", {}).get("rotation_loss_type", "geodesic")

    # Train
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
            checkpoint_path=artifacts["checkpoint_path"],
            metrics_csv_path=artifacts["metrics_csv_path"],
            run_metadata=run_metadata,
            tensorboard_log_dir=tensorboard_log_dir,
            head_lr_decay_cfg=resolved_head_lr_cfg,
            lr_schedule_cfg=lr_schedule_cfg,
            rotation_loss_type=rotation_loss_type,
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
    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Force rebuild of Data_split and *_preprocessed folders before training",
    )

    args = parser.parse_args()

    main(
        args.config,
        checkpointing=False,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer,
        tensorboard_root_dir=args.tensorboard_root,
        tensorboard_port=args.tensorboard_port,
        force_preprocess=args.force_preprocess,
    )