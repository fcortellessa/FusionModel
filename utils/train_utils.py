import torch
import math



def _get_lr_by_group(optimizer, group_name):
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    return float(optimizer.param_groups[0]["lr"])

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


def _build_epoch_lr_scale_fn(num_epochs, schedule_cfg):
    schedule_cfg = schedule_cfg or {}
    if not bool(schedule_cfg.get("enabled", False)):
        return lambda epoch_idx: 1.0

    schedule_type = str(schedule_cfg.get("type", "one_cycle")).lower()
    if schedule_type not in {"one_cycle", "warmup_cosine"}:
        raise ValueError(
            f"Unsupported lr_schedule type: {schedule_type}. Supported options: one_cycle, warmup_cosine."
        )

    start_lr_mult = float(schedule_cfg.get("start_lr_mult", 0.2))
    peak_lr_mult = float(schedule_cfg.get("max_lr_mult", 3.0))
    final_lr_mult = float(schedule_cfg.get("final_lr_mult", 0.05))
    if "warmup_epochs" in schedule_cfg:
        warmup_epochs = int(schedule_cfg.get("warmup_epochs", 1))
    else:
        pct_start = float(schedule_cfg.get("pct_start", 0.2))
        warmup_epochs = int(round(num_epochs * pct_start))
    warmup_epochs = max(1, min(warmup_epochs, max(1, num_epochs - 1)))
    warmup_end_epoch = warmup_epochs - 1
    decay_span = max(1, num_epochs - warmup_epochs)

    def lr_scale(epoch_idx):
        if num_epochs <= 1:
            return peak_lr_mult if schedule_type == "one_cycle" else 1.0

        if epoch_idx <= warmup_end_epoch:
            if warmup_end_epoch == 0:
                return peak_lr_mult if schedule_type == "one_cycle" else 1.0
            progress = epoch_idx / float(warmup_end_epoch)
            target_peak = peak_lr_mult if schedule_type == "one_cycle" else 1.0
            return start_lr_mult + (target_peak - start_lr_mult) * progress

        progress = min(max((epoch_idx - warmup_end_epoch) / float(decay_span), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        target_peak = peak_lr_mult if schedule_type == "one_cycle" else 1.0
        return final_lr_mult + (target_peak - final_lr_mult) * cosine

    return lr_scale


# ============================================================================
# Manager Classes for Training Organization
# ============================================================================

class HeadLRDecayManager:
    """Manages per-head learning rate decay based on validation performance."""
    
    def __init__(self, head_lr_decay_cfg, base_lr, optimizer, param_groups_by_name):
        """
        Args:
            head_lr_decay_cfg: Config dict with 'enabled', 'patience', 'factor', 'min_lr_mult'
            base_lr: Base learning rate
            optimizer: PyTorch optimizer
            param_groups_by_name: Dict mapping group names to param groups
        """
        self.enabled = bool(head_lr_decay_cfg.get("enabled", True))
        self.patience = int(head_lr_decay_cfg.get("patience", 3))
        self.reduce_factor = float(head_lr_decay_cfg.get("factor", 0.3))
        self.min_lr_mult = float(head_lr_decay_cfg.get("min_lr_mult", 0.05))
        
        self.optimizer = optimizer
        self.param_groups_by_name = param_groups_by_name
        self.base_lr = base_lr
        
        # Track base LRs for each group
        self.base_group_lrs = {
            group_name: float(group["lr"])
            for group_name, group in param_groups_by_name.items()
        }
        
        # Track multipliers for each group
        self.group_lr_multipliers = {group_name: 1.0 for group_name in self.base_group_lrs}
        
        # Per-head tracking state
        self.head_state = {
            "q": {"best_val": float("inf"), "bad_epochs": 0, "group_name": "head_rot"},
            "y": {"best_val": float("inf"), "bad_epochs": 0, "group_name": "head_y"},
            "z": {"best_val": float("inf"), "bad_epochs": 0, "group_name": "head_z"},
        }
        
        self.min_lrs = {
            "q": base_lr * self.min_lr_mult,
            "y": base_lr * self.min_lr_mult,
            "z": base_lr * self.min_lr_mult,
        }
    
    def update(self, train_metrics, val_metrics, current_lr_scale, logger=None):
        """
        Update head LR multipliers based on training and validation metrics.
        
        Args:
            train_metrics: Dict with 'loss_q', 'loss_y', 'loss_z'
            val_metrics: Dict with 'loss_q', 'loss_y', 'loss_z'
            current_lr_scale: Current LR schedule scale factor
            logger: Optional logger for decay notifications
        """
        if not self.enabled:
            return
        
        per_head_losses = {
            "q": {"train": train_metrics["loss_q"], "val": val_metrics["loss_q"]},
            "y": {"train": train_metrics["loss_y"], "val": val_metrics["loss_y"]},
            "z": {"train": train_metrics["loss_z"], "val": val_metrics["loss_z"]},
        }
        
        for head_key, head_vals in per_head_losses.items():
            state = self.head_state[head_key]
            best_val = state["best_val"]
            val_head = head_vals["val"]
            train_head = head_vals["train"]
            
            # Check if validation improved
            if val_head < best_val:
                state["best_val"] = val_head
                state["bad_epochs"] = 0
                continue
            
            # Check for overfitting
            overfit_detected = val_head > train_head
            if overfit_detected:
                state["bad_epochs"] += 1
            else:
                state["bad_epochs"] = 0
            
            # Apply decay if patience exceeded
            if state["bad_epochs"] < self.patience:
                continue
            
            group_name = state["group_name"]
            group = self.param_groups_by_name.get(group_name)
            if group is None:
                continue
            
            old_lr = float(group["lr"])
            new_multiplier = max(
                self.group_lr_multipliers.get(group_name, 1.0) * self.reduce_factor,
                self.min_lr_mult
            )
            self.group_lr_multipliers[group_name] = new_multiplier
            new_lr = max(
                self.base_group_lrs.get(group_name, old_lr) * current_lr_scale * new_multiplier,
                self.min_lrs[head_key]
            )
            
            if new_lr < old_lr:
                group["lr"] = new_lr
                if logger:
                    logger.info(
                        f"head_lr_reduce head={head_key} group={group_name} "
                        f"old_lr={old_lr:.8f} new_lr={new_lr:.8f} train_head={train_head:.6f} "
                        f"val_head={val_head:.6f} bad_epochs={state['bad_epochs']}"
                    )
            
            state["bad_epochs"] = 0
    
    def apply_lr_scale(self, current_lr_scale):
        """Apply global LR schedule scale to all groups."""
        for group_idx, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("name", f"group_{group_idx}")
            base_group_lr = self.base_group_lrs.get(group_name, float(group["lr"]))
            group_multiplier = self.group_lr_multipliers.get(group_name, 1.0)
            group["lr"] = base_group_lr * current_lr_scale * group_multiplier


class EpochMetricsAggregator:
    """Aggregates and provides easy access to epoch metrics."""
    
    def __init__(self, metrics_dict, num_batches):
        """
        Args:
            metrics_dict: Dict with accumulated metrics
            num_batches: Number of batches in the epoch
        """
        self.metrics_dict = metrics_dict
        self.num_batches = num_batches
    
    @property
    def avg_loss(self):
        return self.metrics_dict["loss"] / self.num_batches
    
    @property
    def avg_loss_q(self):
        return self.metrics_dict["loss_q"] / self.num_batches
    
    @property
    def avg_loss_y(self):
        return self.metrics_dict["loss_y"] / self.num_batches
    
    @property
    def avg_loss_z(self):
        return self.metrics_dict["loss_z"] / self.num_batches
    
    @property
    def avg_abs_y(self):
        return self.metrics_dict["abs_y"] / self.num_batches
    
    @property
    def avg_abs_z(self):
        return self.metrics_dict["abs_z"] / self.num_batches
    
    @property
    def avg_abs_trans(self):
        return self.metrics_dict["abs_trans"] / self.num_batches
    
    @property
    def avg_abs_rot_x(self):
        return self.metrics_dict["abs_rot_x"] / self.num_batches
    
    def to_dict(self):
        """Return all averaged metrics as a dict."""
        return {
            "loss": self.avg_loss,
            "loss_q": self.avg_loss_q,
            "loss_y": self.avg_loss_y,
            "loss_z": self.avg_loss_z,
            "abs_y": self.avg_abs_y,
            "abs_z": self.avg_abs_z,
            "abs_trans": self.avg_abs_trans,
            "abs_rot_x": self.avg_abs_rot_x,
        }


class TrainingConfigBuilder:
    """Builds training components (model, optimizer, loss, uncertainty) from config."""
    
    def __init__(self, cfg, device, logger=None):
        """
        Args:
            cfg: Config dict
            device: torch device
            logger: Optional logger
        """
        self.cfg = cfg
        self.device = device
        self.logger = logger
    
    def build_model(self):
        """Build model from config."""
        from base_model.pose_estimation_model import PoseEstimationNet
        
        model = PoseEstimationNet(
            model_name=self.cfg["model"]["backbone"],
            streams=self.cfg["model"]["streams"],
            fusion=self.cfg["model"]["fusion"],
            freeze_backbone=self.cfg["model"]["freeze_backbone"],
            rotation_parametrisation=self.cfg["model"]["head_type"],
            activation=self.cfg["model"]["activation"],
            rotation_layers=[
                self.cfg["model"]["mlp_quat_1"],
                self.cfg["model"]["mlp_quat_2"],
                self.cfg["model"]["mlp_quat_3"]
            ],
            translation_layers=[
                self.cfg["model"]["mlp_trans_1"],
                self.cfg["model"]["mlp_trans_2"],
                self.cfg["model"]["mlp_trans_3"]
            ],
            dropout_rot=self.cfg["training"]["dropout_rot"],
            dropout_trans=self.cfg["training"]["dropout_trans"],
            use_batchnorm=self.cfg["training"]["use_batchnorm"]
        )
        return model.to(self.device)
    
    def build_loss_function(self):
        """Build loss function from config."""
        from losses.original_loss import total_loss
        from losses.se2_losses import total_loss_2D_vector
        
        head_type = self.cfg.get("model", {}).get("head_type", "quaternion")
        if head_type == "2D_unit_vector":
            return total_loss_2D_vector
        else:
            return total_loss
    
    def build_optimizer(self, model, param_groups=None):
        """Build optimizer from config."""
        from utils.utils import get_optimizer
        
        if param_groups is None:
            param_groups = _build_named_param_groups(model, base_lr=float(self.cfg["training"]["lr"]))
        
        optimizer_name = self.cfg["training"]["optimizer"]
        base_lr = float(self.cfg["training"]["lr"])
        
        return get_optimizer(param_groups, optimizer_name=optimizer_name, lr=base_lr)
    
    def build_uncertainty_module(self, optimizer, base_lr):
        """Build uncertainty weighting module if enabled."""
        from torch import nn
        
        uncertainty_cfg = self.cfg.get("training", {}).get("uncertainty_weighting", {})
        use_uncertainty_weighting = bool(uncertainty_cfg.get("enabled", False))
        
        if not use_uncertainty_weighting:
            return None
        
        class UncertaintyWeighting(nn.Module):
            def __init__(self, init_log_var_rot=0.0, init_log_var_trans=0.0):
                super().__init__()
                self.log_var_rot = nn.Parameter(torch.tensor(float(init_log_var_rot), dtype=torch.float32))
                self.log_var_trans = nn.Parameter(torch.tensor(float(init_log_var_trans), dtype=torch.float32))
            
            def forward(self, loss_rot, loss_trans):
                weighted_rot = torch.exp(-self.log_var_rot) * loss_rot + self.log_var_rot
                weighted_trans = torch.exp(-self.log_var_trans) * loss_trans + self.log_var_trans
                return weighted_rot + weighted_trans
        
        uncertainty_module = UncertaintyWeighting(
            init_log_var_rot=float(uncertainty_cfg.get("init_log_var_rot", 0.0)),
            init_log_var_trans=float(uncertainty_cfg.get("init_log_var_trans", 0.0)),
        ).to(self.device)
        
        uncertainty_lr_mult = float(uncertainty_cfg.get("lr_mult", 1.0))
        optimizer.add_param_group({
            "name": "uncertainty",
            "params": uncertainty_module.parameters(),
            "lr": base_lr * uncertainty_lr_mult,
        })
        
        return uncertainty_module
    
    def build(self):
        """Build all training components.
        
        Returns:
            Tuple of (model, optimizer, uncertainty_module, loss_function, param_groups)
        """
        model = self.build_model()
        param_groups = _build_named_param_groups(model, base_lr=float(self.cfg["training"]["lr"]))
        optimizer = self.build_optimizer(model, param_groups=param_groups)
        base_lr = float(self.cfg["training"]["lr"])
        uncertainty_module = self.build_uncertainty_module(optimizer, base_lr)
        loss_function = self.build_loss_function()
        
        if self.logger:
            self.logger.info(f"Built model: {self.cfg['model']['backbone']}")
            self.logger.info(f"Built optimizer: {self.cfg['training']['optimizer']}")
            if uncertainty_module is not None:
                self.logger.info("Built uncertainty weighting module")
        
        return model, optimizer, uncertainty_module, loss_function, param_groups


class DataPipelineManager:
    """Manages data loading and preprocessing pipeline."""
    
    def __init__(self, cfg, base_dir, force_preprocess=False, logger=None):
        """
        Args:
            cfg: Config dict
            base_dir: Base directory for relative paths
            force_preprocess: Force reprocessing of data
            logger: Optional logger
        """
        self.cfg = cfg
        self.base_dir = base_dir
        self.force_preprocess = force_preprocess
        self.logger = logger
    
    def _get_data_dir(self):
        """Resolve data directory from config."""
        import os
        
        configured_data_root = self.cfg.get("data", {}).get("root", "")
        if configured_data_root:
            return os.path.abspath(configured_data_root)
        else:
            return os.path.abspath(os.path.join(self.base_dir, "../Data"))
    
    def preprocess_if_needed(self):
        """Preprocess data if needed."""
        import os
        from data_processing.preprocess_data import preprocess_data
        
        data_dir = self._get_data_dir()
        
        aug_cfg = self.cfg.get("data", {}).get("augmentation", {})
        aug_output_folder = str(aug_cfg.get("output_folder", "train_augmentation_default"))
        use_photo_aug = bool(aug_cfg.get("enabled", False))
        use_preaugmented = bool(self.cfg.get("data", {}).get("use_preaugmented_train_dir", False))
        train_folder = aug_output_folder if use_preaugmented else "train"
        preprocessed_dir = os.path.join(data_dir, "Data_split", train_folder, "train_preprocessing")
        should_preprocess = self.force_preprocess or (not os.path.exists(preprocessed_dir))
        
        if should_preprocess:
            if self.logger:
                self.logger.info(f"Preprocessing data from {data_dir}")
                if use_preaugmented:
                    self.logger.info(f"Using pre-augmented training data from {aug_output_folder}")
            
            preprocess_data(
                root_dir=data_dir,
                use_photometric_augmentation=use_photo_aug,
                aug_seed=int(aug_cfg.get("seed", 42)),
                aug_photo_count=int(aug_cfg.get("photo_count", 1)),
                aug_overwrite=bool(aug_cfg.get("overwrite", False)),
                aug_output_folder=aug_output_folder,
                use_preaugmented_train_dir=use_preaugmented,
            )
        elif use_photo_aug:
            if self.logger:
                self.logger.info(
                    "Photometric augmentation enabled but using existing preprocessed data. "
                    "Use --force_preprocess to regenerate with augmentation."
                )
    
    def get_loaders(self, batch_size):
        """Get train, val, test dataloaders.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        import os
        from data_processing.dataloader_preprocessed import get_dataloaders
        
        self.preprocess_if_needed()
        
        data_dir = self._get_data_dir()
        aug_cfg = self.cfg.get("data", {}).get("augmentation", {})
        use_preaugmented = bool(self.cfg.get("data", {}).get("use_preaugmented_train_dir", False))
        train_folder = str(aug_cfg.get("output_folder", "train_augmentation_default")) if use_preaugmented else "train"
        train_loader, val_loader, test_loader = get_dataloaders(
            root_dir=os.path.join(data_dir, "Data_split"),
            batch_size=batch_size,
            shuffle=True,
            train_folder=train_folder,
        )
        
        if self.logger:
            self.logger.info(f"Loaded dataloaders with batch_size={batch_size}")
        
        return train_loader, val_loader, test_loader


class RunSetupManager:
    """Manages run setup: logging, artifacts, tensorboard, manifests."""
    
    def __init__(self, config_path, cfg, batch_size, logger=None):
        """
        Args:
            config_path: Path to config file
            cfg: Config dict
            batch_size: Batch size
            logger: Optional logger
        """
        self.config_path = config_path
        self.cfg = cfg
        self.batch_size = batch_size
        self.logger = logger
        self.artifacts = None
    
    def setup_artifacts(self):
        """Build run artifacts directory structure."""
        from utils.utils import build_run_artifacts, save_yaml
        
        self.artifacts = build_run_artifacts(
            self.config_path,
            self.cfg,
            self.batch_size,
            output_root="runs"
        )
        save_yaml(self.cfg, self.artifacts["config_snapshot_path"])
        
        if self.logger:
            self.logger.info(f"Created run directory: {self.artifacts['run_dir']}")
        
        return self.artifacts
    
    def setup_tensorboard(self, tb_port, tb_root=None):
        """Setup tensorboard logging.
        
        Args:
            tb_port: TensorBoard port
            tb_root: Optional override for TB root directory
            
        Returns:
            Tuple of (tensorboard_log_dir, tensorboard_process)
        """
        import os
        from utils.logging_utils import is_port_in_use, start_tensorboard
        
        if self.artifacts is None:
            raise ValueError("Must call setup_artifacts() first")
        
        tb_root = tb_root or self.artifacts["tensorboard_root_dir"]
        tensorboard_log_dir = os.path.join(tb_root, self.artifacts["run_name"])
        
        tensorboard_process = None
        if is_port_in_use(tb_port):
            if self.logger:
                self.logger.info(f"TensorBoard already running on port {tb_port}")
        else:
            tensorboard_process = start_tensorboard(tb_root, logger=self.logger, port=tb_port)
            if self.logger:
                self.logger.info(f"Started TensorBoard on port {tb_port}")
        
        if self.logger:
            self.logger.info(f"TensorBoard log dir: {tensorboard_log_dir}")
        
        return tensorboard_log_dir, tensorboard_process
    
    def save_manifest(self, run_metadata):
        """Save run manifest file."""
        import os
        from utils.utils import save_json
        
        if self.artifacts is None:
            raise ValueError("Must call setup_artifacts() first")
        
        run_manifest = {
            "config_path": os.path.abspath(self.config_path),
            "config_snapshot_path": os.path.abspath(self.artifacts["config_snapshot_path"]),
            "run_root_dir": os.path.abspath(self.artifacts["run_root_dir"]),
            "run_dir": os.path.abspath(self.artifacts["run_dir"]),
            "run_name": self.artifacts["run_name"],
            "checkpoint_path": os.path.abspath(self.artifacts["checkpoint_path"]),
            "metrics_csv_path": os.path.abspath(self.artifacts["metrics_csv_path"]),
            "tensorboard_run_dir": os.path.abspath(self.artifacts["tensorboard_run_dir"]),
            **run_metadata,
        }
        
        save_json(run_manifest, self.artifacts["manifest_path"])
        
        if self.logger:
            self.logger.info(f"Saved run manifest to {self.artifacts['manifest_path']}")
        
        return run_manifest
    
    def get_artifacts(self):
        """Get artifacts dict (setup_artifacts must be called first)."""
        if self.artifacts is None:
            self.setup_artifacts()
        return self.artifacts

