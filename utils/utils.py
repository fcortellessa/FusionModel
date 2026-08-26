import json
import os
import random
import re
import sys
from datetime import datetime
import logging

import numpy as np
import torch
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_data_root(cfg, repo_root):
    """Resolve the dataset root directory from `data.root`, defaulting to <repo_root>/Data.

    Shared by every entry point that needs to locate a run's data (training and
    testing) so they all agree on the same folder for the same config.
    """
    configured_root = cfg.get("data", {}).get("root", "")
    if configured_root:
        return os.path.abspath(configured_root)
    return os.path.abspath(os.path.join(repo_root, "Data"))


def resolve_split_seed(cfg):
    """Resolve the seed used for the train/val/test split.

    `data.split_seed` when set, otherwise `training.seed` (matching the split's
    historical implicit behavior). See data_processing/GripFusionNet_split.py.
    """
    return cfg.get("data", {}).get("split_seed", cfg.get("training", {}).get("seed"))


def resolve_group_aware_split(cfg):
    """Resolve whether the grasp-event-group-aware split method is used.

    `data.group_aware_split`, defaulting to True (the leak-free method; see
    data_processing/GripFusionNet_split.py). Set to False only to reproduce
    the legacy per-frame shuffle for comparison. Must be resolved the same
    way at train time and at eval time -- otherwise `ensure_test_split_current`
    would silently regenerate a run's split with the *other* method before
    evaluating it.
    """
    return bool(cfg.get("data", {}).get("group_aware_split", True))


def sanitize_filename(value):
    value = str(value)
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip(" ._")
    return value or "run"


def build_run_artifacts(config_path, cfg, batch_size, output_root="runs", existing_run_dir=None):
    """Build (or, when resuming, reuse) a run's artifact paths.

    existing_run_dir: when resuming training (see --resume_from in train.py),
    pass the original run's directory here instead of generating a fresh
    timestamped one -- so the resumed segment continues writing
    checkpoint_best.pth/metrics.csv/config.yml into the same run folder
    rather than starting a second, disconnected run.
    """
    if existing_run_dir is not None:
        run_dir = existing_run_dir
        run_root_dir = os.path.dirname(os.path.normpath(run_dir)) or output_root
        run_name = os.path.basename(os.path.normpath(run_dir))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fusion_name = sanitize_filename(cfg["model"]["fusion"])
        run_name = f"{timestamp}_bs{batch_size}_{fusion_name}"
        run_root_dir = output_root
        run_dir = os.path.join(run_root_dir, run_name)
    results_dir = os.path.join(run_dir, "results")
    tensorboard_root_dir = os.path.join(run_root_dir, "tensorboard")
    tensorboard_run_dir = os.path.join(tensorboard_root_dir, run_name)

    for directory in [run_root_dir, run_dir, results_dir, tensorboard_root_dir, tensorboard_run_dir]:
        os.makedirs(directory, exist_ok=True)

    return {
        "run_root_dir": run_root_dir,
        "run_dir": run_dir,
        "run_name": run_name,
        "results_dir": results_dir,
        "tensorboard_root_dir": tensorboard_root_dir,
        "tensorboard_run_dir": tensorboard_run_dir,
        "checkpoint_path": os.path.join(run_dir, "checkpoint_best.pth"),
        "metrics_csv_path": os.path.join(run_dir, "metrics.csv"),
        "config_snapshot_path": os.path.join(run_dir, "config.yml"),
        "manifest_path": os.path.join(run_dir, "manifest.json"),
    }


def save_yaml(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("GripFusionNet")
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if main() is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler (STDOUT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, log_file


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path="checkpoints", interval=1,
                     uncertainty_module=None, early_stopper=None, head_lr_manager=None,
                     global_plateau_manager=None, best_val_loss=None, save_rng_state=True):
    """Save checkpoint_best.pth.

    Beyond the model weights, this also captures everything needed to resume
    training from this exact point rather than restarting at epoch 0 --
    optimizer state (Adam/AdamW momentum), the learned uncertainty-weighting
    log-variances (if task_weighting is enabled), each stateful LR/early-stop
    manager's own state_dict, and the RNG states. All of the extra fields are
    optional and additive: a checkpoint saved without them (or by older code)
    still loads fine for pure inference via load_checkpoint(..., load_optimizer=False).

    RNG-state restoration is best-effort, not a bit-exact-reproducibility
    guarantee -- torch DataLoader workers (num_workers>0) derive their own
    per-worker seeds at DataLoader construction time, which this doesn't
    replay, so a resumed run's exact batch order/augmentation draws can still
    diverge from an uninterrupted run even with RNG state restored. It does
    still meaningfully help (model forward-pass randomness, e.g. dropout,
    and single-process shuffling).
    """
    if epoch % interval == 0:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        if best_val_loss is not None:
            payload['best_val_loss'] = best_val_loss
        if uncertainty_module is not None:
            payload['uncertainty_state_dict'] = uncertainty_module.state_dict()
        if early_stopper is not None and hasattr(early_stopper, "state_dict"):
            payload['early_stopper_state'] = early_stopper.state_dict()
        if head_lr_manager is not None and hasattr(head_lr_manager, "state_dict"):
            payload['head_lr_manager_state'] = head_lr_manager.state_dict()
        if global_plateau_manager is not None and hasattr(global_plateau_manager, "state_dict"):
            payload['global_plateau_manager_state'] = global_plateau_manager.state_dict()
        if save_rng_state:
            payload['rng_state'] = {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            }
        torch.save(payload, checkpoint_path)
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path, device, strict=False, load_optimizer=True):
    """
    Load checkpoint with model and optimizer state dicts.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        strict: Whether to strictly enforce matching keys (default False for robustness)
        load_optimizer: Whether to load optimizer state dict (default True). Set to False for inference/testing
                       when optimizer structure may differ from training time.
    
    Returns:
        checkpoint dict with 'epoch', 'loss', 'model_state_dict', 'optimizer_state_dict'
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            # Handle parameter group mismatch (e.g., when uncertainty module was added during training)
            if "different number of parameter groups" in str(e):
                print(f"[WARNING] Could not load optimizer state dict due to parameter group mismatch: {e}")
                print("[WARNING] This typically happens when uncertainty weighting or other modules were added during training.")
                print("[WARNING] Proceeding with model weights only.")
            else:
                raise
    
    return checkpoint


def load_checkpoint_for_resume(checkpoint_path, model, optimizer, device, uncertainty_module=None, restore_rng=True):
    """Load a checkpoint saved by save_checkpoint() to resume *training* (not just
    inference) from the exact epoch it was saved at.

    Unlike load_checkpoint() (used by test.py/eval scripts with
    load_optimizer=False for pure inference), this requires the checkpoint to
    actually contain 'optimizer_state_dict' -- checkpoints saved before
    checkpoint-resume support was added only have 'model_state_dict', and
    should go through load_checkpoint() instead.

    Restores model + optimizer + uncertainty_module (if given) weights and,
    if present, the RNG states captured at save time (best-effort -- see the
    save_rng_state note on save_checkpoint(); DataLoader worker shuffling
    doesn't replay exactly). Returns the raw checkpoint dict so the caller can
    additionally restore its LR/early-stopping manager states (those are
    constructed later, inside train_loop) and read 'epoch'/'best_val_loss' to
    resume the epoch loop at the right point.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    if 'optimizer_state_dict' not in checkpoint:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} has no 'optimizer_state_dict' -- it was "
            "saved before checkpoint-resume support was added and can only be used for "
            "inference (see load_checkpoint), not training resume."
        )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if uncertainty_module is not None and 'uncertainty_state_dict' in checkpoint:
        uncertainty_module.load_state_dict(checkpoint['uncertainty_state_dict'])

    if restore_rng and checkpoint.get('rng_state'):
        rng = checkpoint['rng_state']
        if rng.get('torch') is not None:
            # torch.load(..., map_location=device) above moves every tensor in the
            # checkpoint onto `device`, including this one -- but set_rng_state()
            # requires a plain CPU ByteTensor (CUDA RNG state is restored separately
            # below via set_rng_state_all), so move it back explicitly.
            torch.set_rng_state(rng['torch'].to("cpu", torch.uint8))
        if rng.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.to("cpu", torch.uint8) for state in rng['cuda']])
        if rng.get('numpy') is not None:
            np.random.set_state(rng['numpy'])
        if rng.get('python') is not None:
            random.setstate(rng['python'])

    return checkpoint


def strip_optimizer_state(checkpoint_path):
    """Remove 'optimizer_state_dict' from a saved checkpoint in place.

    Call this once training has genuinely concluded (early stopping fired, or the
    configured epoch count was reached) and there is nothing left to resume --
    never on a crash/kill, since that checkpoint may still need --resume_from later.
    The optimizer state is by far the largest piece of checkpoint-resume state
    (AdamW keeps two extra tensors per parameter, roughly doubling the checkpoint's
    size on top of the model weights) and serves no purpose once only inference is
    left to do with the checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "optimizer_state_dict" not in checkpoint:
        return
    del checkpoint["optimizer_state_dict"]
    torch.save(checkpoint, checkpoint_path)


def get_optimizer(model_or_params, optimizer_name="Adam", lr=0.001, weight_decay=0.0):
    if hasattr(model_or_params, "parameters"):
        params = model_or_params.parameters()
    else:
        params = model_or_params

    if optimizer_name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "MuSGD":
        from ultralytics.optim import MuSGD
        return MuSGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Available options: Adam, AdamW, RMSprop, SGD, MuSGD.")
