# train_utils.py
import math
from datetime import datetime

import os
import torch
from torch import nn

from utils.utils import build_run_artifacts, save_yaml, get_optimizer, save_json


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
        "head_trans": [],
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
        elif name.startswith("trans_head."):
            grouped_params["head_trans"].append(param)
        else:
            grouped_params["base"].append(param)

    param_groups = []
    for group_name in ["base", "head_rot", "head_y", "head_z", "head_trans"]:
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
    """Warmup + cosine decay schedule, expressed as a per-epoch multiplier on base_lr.

    Ramps linearly from `start_lr_mult` to 1.0 over the warmup epochs, then cosine-decays
    from 1.0 down to `final_lr_mult` over the remaining epochs.

    Only handles `lr_schedule.type: warmup_cosine`. `reduce_on_plateau` is a stateful,
    val-loss-driven schedule and is handled separately by `GlobalPlateauLRManager` in
    the train loop instead of this stateless per-epoch function.
    """
    schedule_cfg = schedule_cfg or {}
    if not bool(schedule_cfg.get("enabled", False)):
        return lambda epoch_idx: 1.0

    schedule_type = str(schedule_cfg.get("type", "warmup_cosine")).lower()
    if schedule_type == "reduce_on_plateau":
        return lambda epoch_idx: 1.0

    # one_cycle/multi_cycle are retired; old configs using those names now just get
    # warmup_cosine behavior (their max_lr_mult key, if present, is ignored).
    if schedule_type not in {"warmup_cosine", "one_cycle", "multi_cycle"}:
        raise ValueError(
            f"Unsupported lr_schedule type: {schedule_type}. "
            f"Supported options: warmup_cosine, reduce_on_plateau."
        )

    start_lr_mult = float(schedule_cfg.get("start_lr_mult", 0.2))
    final_lr_mult = float(schedule_cfg.get("final_lr_mult", 0.05))
    if "warmup_epochs" in schedule_cfg:
        warmup_epochs = int(schedule_cfg.get("warmup_epochs", 1))
    else:
        pct_start = float(schedule_cfg.get("pct_start", 0.2))   # pct = percentage of total epochs for warmup
        warmup_epochs = int(round(num_epochs * pct_start))
    warmup_epochs = max(1, min(warmup_epochs, max(1, num_epochs - 1)))
    warmup_end_epoch = warmup_epochs - 1
    decay_span = max(1, num_epochs - warmup_epochs)

    def lr_scale(epoch_idx):
        if num_epochs <= 1:
            return 1.0

        if epoch_idx <= warmup_end_epoch:
            if warmup_end_epoch == 0:
                return 1.0
            progress = epoch_idx / float(warmup_end_epoch)
            return start_lr_mult + (1.0 - start_lr_mult) * progress

        progress = min(max((epoch_idx - warmup_end_epoch) / float(decay_span), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_mult + (1.0 - final_lr_mult) * cosine

    return lr_scale

def _apply_metric_accumulation(metric_sums, batch_values):
    for key, value in batch_values.items():
        metric_sums[key] += value

def build_epoch_metrics_row(
    metadata,
    epoch,
    train_avg,
    val_avg,
    fusion_weights_str,
    best_val_loss,
    epoch_time,
    lr_base,
    lr_head_rot,
    lr_head_y,
    lr_head_z,
    current_lr_scale,
    uncertainty_stats,
    validation_rotation_summary=None,
    checkpoint_path="",
):
    metadata = metadata or {}
    uncertainty_stats = uncertainty_stats or {}
    validation_rotation_summary = validation_rotation_summary or {}

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "epoch": epoch + 1,
        "train_loss": train_avg["loss"],
        "val_loss": val_avg["loss"],
        "loss_q": train_avg["loss_q"],
        "loss_y": train_avg["loss_y"],
        "loss_z": train_avg["loss_z"],
        "val_loss_q": val_avg["loss_q"],
        "val_loss_y": val_avg["loss_y"],
        "val_loss_z": val_avg["loss_z"],
        "train_abs_y": train_avg["abs_y"],
        "train_abs_z": train_avg["abs_z"],
        "train_abs_trans": train_avg["abs_trans"],
        "train_abs_rot_x": train_avg["abs_rot_x"],
        "val_abs_y": val_avg["abs_y"],
        "val_abs_z": val_avg["abs_z"],
        "val_abs_trans": val_avg["abs_trans"],
        "val_abs_rot_x": val_avg["abs_rot_x"],
        "val_rot_median_deg": validation_rotation_summary.get("val_rot_median_deg", ""),
        "val_rot_p90_deg": validation_rotation_summary.get("val_rot_p90_deg", ""),
        "val_rot_within_5deg": validation_rotation_summary.get("val_rot_within_5deg", ""),
        "fusion_weights": fusion_weights_str,
        "train_fusion_gate_mean": metadata.get("train_fusion_gate_mean", ""),
        "train_fusion_gate_std": metadata.get("train_fusion_gate_std", ""),
        "train_fusion_gate_fraction_lt_0_1": metadata.get("train_fusion_gate_fraction_lt_0_1", ""),
        "train_fusion_gate_fraction_gt_0_9": metadata.get("train_fusion_gate_fraction_gt_0_9", ""),
        "val_fusion_gate_mean": metadata.get("val_fusion_gate_mean", ""),
        "val_fusion_gate_std": metadata.get("val_fusion_gate_std", ""),
        "val_fusion_gate_fraction_lt_0_1": metadata.get("val_fusion_gate_fraction_lt_0_1", ""),
        "val_fusion_gate_fraction_gt_0_9": metadata.get("val_fusion_gate_fraction_gt_0_9", ""),
        "best_loss": best_val_loss,
        "epoch_time_sec": epoch_time,
        "checkpoint_path": os.path.abspath(checkpoint_path) if checkpoint_path else "",
        "model_name": metadata.get("model_name", ""),
        "streams": metadata.get("streams", ""),
        "fusion": metadata.get("fusion", ""),
        "head_type": metadata.get("head_type", ""),
        "freeze_backbone": metadata.get("freeze_backbone", ""),
        "lr": lr_base,
        "lr_base": lr_base,
        "lr_head_rot": lr_head_rot,
        "lr_head_y": lr_head_y,
        "lr_head_z": lr_head_z,
        "lr_schedule_scale": current_lr_scale,
        "uncertainty_log_var_rot": uncertainty_stats.get("uncertainty_log_var_rot", ""),
        "uncertainty_log_var_trans": uncertainty_stats.get("uncertainty_log_var_trans", ""),
        "uncertainty_weight_rot": uncertainty_stats.get("uncertainty_weight_rot", ""),
        "uncertainty_weight_trans": uncertainty_stats.get("uncertainty_weight_trans", ""),
        "batch_size": metadata.get("batch_size", ""),
        "seed": metadata.get("seed", ""),
        "num_parameters": metadata.get("num_parameters", ""),
        "num_trainable_parameters": metadata.get("num_trainable_parameters", ""),
    }

# ============================================================================
# Manager Classes for Training Organization
# ============================================================================

class _MultiplierOptimizer(torch.optim.Optimizer):
    """A single-param-group no-op Optimizer whose `lr` is a pure multiplier in [min_lr_mult, 1.0].

    ReduceLROnPlateau requires a real torch.optim.Optimizer and only ever reads/writes
    param_groups[0]["lr"]. Running it over this shadow multiplier (instead of the real,
    cosine-scaled group LR) lets the plateau reduction persist independently of the
    epoch's cosine scale, which is re-applied separately every epoch in
    `HeadLRDecayManager.apply_lr_scale`.
    """

    def __init__(self):
        dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        super().__init__([dummy_param], {"lr": 1.0})

    def step(self, closure=None):
        pass


class HeadLRDecayManager:
    """Per-head ReduceLROnPlateau: decays each head's LR off that head's own best-ever val loss."""

    def __init__(self, head_lr_decay_cfg, optimizer, param_groups_by_name):
        """
        Args:
            head_lr_decay_cfg: Config dict with 'enabled', 'patience', 'factor', 'min_delta'
                (minimum absolute val-loss improvement to reset a head's patience counter --
                same idea as EarlyStoppingManager's min_delta, so tiny fluctuations don't
                reset the count; default 0.0 matches PyTorch's own tiny default tolerance
                behavior of requiring *some* improvement, just without a relative threshold),
                and either 'min_lr' (absolute LR floor per head, aligned with
                GlobalPlateauLRManager's 'min_lr' -- both express the floor as an absolute LR)
                or the legacy 'min_lr_mult' (floor as a fraction of that head's base_lr, kept
                for backward compatibility with existing configs; 'min_lr' takes precedence
                if both are set)
            optimizer: PyTorch optimizer
            param_groups_by_name: Dict mapping group names to param groups
        """
        self.enabled = bool(head_lr_decay_cfg.get("enabled", True))
        patience = int(head_lr_decay_cfg.get("patience", 3))
        factor = float(head_lr_decay_cfg.get("factor", 0.3))
        min_delta = float(head_lr_decay_cfg.get("min_delta", 0.0))
        min_lr = head_lr_decay_cfg.get("min_lr", None)
        min_lr_mult_legacy = float(head_lr_decay_cfg.get("min_lr_mult", 0.05))

        self.optimizer = optimizer
        self.param_groups_by_name = param_groups_by_name

        # Track base LRs for each group so the global cosine schedule can be re-applied
        # on top of the plateau-reduced multiplier every epoch.
        self.base_group_lrs = {
            group_name: float(group["lr"])
            for group_name, group in param_groups_by_name.items()
        }

        # "trans" is used instead of separate "y"/"z" keys when the model has a single
        # joint translation head (both live in the same "head_trans" param group, so they
        # can only be decayed together, off the combined y+z val loss).
        if "head_trans" in param_groups_by_name:
            head_to_group = {"q": "head_rot", "trans": "head_trans"}
        else:
            head_to_group = {"q": "head_rot", "y": "head_y", "z": "head_z"}
        self.head_to_group = head_to_group
        self.group_to_head = {v: k for k, v in head_to_group.items()}
        self.plateau_schedulers = {}
        self.multiplier_views = {}
        for head_key, group_name in head_to_group.items():
            if group_name not in param_groups_by_name:
                continue
            view = _MultiplierOptimizer()
            self.multiplier_views[head_key] = view
            # The shadow view's "lr" is a pure multiplier starting at 1.0. When 'min_lr' is
            # given (absolute LR, matching GlobalPlateauLRManager), convert it to the
            # equivalent multiplier floor for this head's own base LR; otherwise fall back
            # to the legacy 'min_lr_mult' fraction directly.
            if min_lr is not None:
                group_base_lr = self.base_group_lrs.get(group_name, 0.0)
                min_lr_mult = (float(min_lr) / group_base_lr) if group_base_lr > 0 else 0.0
            else:
                min_lr_mult = min_lr_mult_legacy
            self.plateau_schedulers[head_key] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                view,
                mode="min",
                factor=factor,
                patience=patience,
                threshold=min_delta,
                threshold_mode="abs",
                min_lr=min_lr_mult,
            )

    def update(self, train_metrics, val_metrics, current_lr_scale, logger=None):
        """
        Step each head's ReduceLROnPlateau scheduler on that head's own val loss.

        Args:
            train_metrics: Dict with 'loss_q', 'loss_y', 'loss_z' (unused; kept for call-site compatibility)
            val_metrics: Dict with 'loss_q', 'loss_y', 'loss_z'
            current_lr_scale: Current LR schedule scale factor (unused here; the plateau
                multiplier is combined with it in apply_lr_scale)
            logger: Optional logger for decay notifications
        """
        if not self.enabled:
            return

        val_by_head = {
            "q": val_metrics["loss_q"],
            "y": val_metrics["loss_y"],
            "z": val_metrics["loss_z"],
            "trans": val_metrics["loss_y"] + val_metrics["loss_z"],
        }

        for head_key, scheduler in self.plateau_schedulers.items():
            view = self.multiplier_views[head_key]
            old_mult = float(view.param_groups[0]["lr"])
            scheduler.step(val_by_head[head_key])
            new_mult = float(view.param_groups[0]["lr"])
            if new_mult < old_mult and logger:
                group_name = self.head_to_group[head_key]
                logger.info(
                    f"head_lr_reduce head={head_key} group={group_name} "
                    f"old_mult={old_mult:.6f} new_mult={new_mult:.6f} val_head={val_by_head[head_key]:.6f} "
                    f"num_bad_epochs={scheduler.num_bad_epochs}"
                )

    def apply_lr_scale(self, current_lr_scale):
        """Apply the global LR schedule scale to all groups, layered on top of each head's
        persistent plateau multiplier."""
        self.apply_reference_lrs({
            group.get("name", f"group_{idx}"): self.base_group_lrs.get(
                group.get("name", f"group_{idx}"), float(group["lr"])
            ) * current_lr_scale
            for idx, group in enumerate(self.optimizer.param_groups)
        })

    def apply_reference_lrs(self, reference_lr_by_group):
        """Multiply each head's persistent plateau factor onto an externally-computed
        reference LR per group.

        `reference_lr_by_group` is whatever the base schedule (warmup_cosine's per-epoch
        scale, or GlobalPlateauLRManager's own live LR) has decided each group's LR should
        be *before* per-head decay is layered on -- this lets per-head decay compose with
        either base schedule, since it never assumes who produced the reference value.
        """
        for group_idx, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("name", f"group_{group_idx}")
            reference_lr = reference_lr_by_group.get(group_name, float(group["lr"]))
            group["lr"] = reference_lr * self._plateau_multiplier(group_name)

    def _plateau_multiplier(self, group_name):
        head_key = self.group_to_head.get(group_name)
        if head_key is None:
            return 1.0
        view = self.multiplier_views.get(head_key)
        if view is None:
            return 1.0
        return float(view.param_groups[0]["lr"])


class GlobalPlateauLRManager:
    """Global ReduceLROnPlateau: decays every param group's LR together off overall val_loss.

    Runs the stock `torch.optim.lr_scheduler.ReduceLROnPlateau` over a shadow
    `_MultiplierOptimizer` (the same trick `HeadLRDecayManager` uses) rather than the real
    optimizer directly, so its reduction is a pure, persistent multiplier independent of
    whatever else (e.g. per-head decay) also writes to `optimizer.param_groups[*]["lr"]`.
    That keeps this manager composable with `HeadLRDecayManager`: this produces the
    per-group *reference* LR for the epoch, which `HeadLRDecayManager.apply_reference_lrs`
    then multiplies its own per-head factor onto -- so "per-head decay compatible with any
    base schedule" also covers reduce_on_plateau as the base, not just warmup_cosine.
    """

    def __init__(self, lr_schedule_cfg, optimizer, param_groups_by_name):
        lr_schedule_cfg = lr_schedule_cfg or {}
        self.enabled = (
            bool(lr_schedule_cfg.get("enabled", False))
            and str(lr_schedule_cfg.get("type", "warmup_cosine")).lower() == "reduce_on_plateau"
        )
        self.optimizer = optimizer
        self.base_group_lrs = {
            group_name: float(group["lr"])
            for group_name, group in param_groups_by_name.items()
        }
        self.multiplier_view = None
        self.scheduler = None
        if self.enabled:
            self.multiplier_view = _MultiplierOptimizer()
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.multiplier_view,
                mode="min",
                factor=float(lr_schedule_cfg.get("factor", 0.2)),
                patience=int(lr_schedule_cfg.get("patience", 5)),
                threshold=float(lr_schedule_cfg.get("min_delta", 0.0)),
                threshold_mode="abs",
                min_lr=float(lr_schedule_cfg.get("min_lr", 0.0)),
            )

    def update(self, val_loss, logger=None):
        """Call once per epoch with val_avg['loss']."""
        if not self.enabled:
            return
        old_mult = float(self.multiplier_view.param_groups[0]["lr"])
        self.scheduler.step(val_loss)
        new_mult = float(self.multiplier_view.param_groups[0]["lr"])
        if logger and new_mult < old_mult:
            logger.info(
                f"global_lr_reduce val_loss={val_loss:.6f} "
                f"old_mult={old_mult:.6f} new_mult={new_mult:.6f} "
                f"num_bad_epochs={self.scheduler.num_bad_epochs}"
            )

    def reference_lrs(self):
        """Return this epoch's per-group reference LR (base_lr * persistent plateau mult),
        for `HeadLRDecayManager.apply_reference_lrs` to layer per-head decay on top of."""
        mult = float(self.multiplier_view.param_groups[0]["lr"]) if self.enabled else 1.0
        return {group_name: base_lr * mult for group_name, base_lr in self.base_group_lrs.items()}


class EarlyStoppingManager:
    """Stops training when val_loss stops improving by more than min_delta."""

    def __init__(self, early_stopping_cfg):
        early_stopping_cfg = early_stopping_cfg or {}
        self.enabled = bool(early_stopping_cfg.get("enabled", False))
        self.patience = int(early_stopping_cfg.get("patience", 10))
        self.min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
        self.best_value = float("inf")
        self.bad_epochs = 0

    def update(self, val_loss):
        """Call once per epoch with val_avg['loss']. Returns True if training should stop now."""
        if not self.enabled:
            return False
        if val_loss < self.best_value - self.min_delta:
            self.best_value = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


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
            use_batchnorm=self.cfg["training"]["use_batchnorm"],
            task_adapter=self.cfg["model"].get("task_adapter", None),
            task_feature_adapter=self._is_task_feature_adapter_enabled(),
            task_feature_adapter_type=self._get_task_feature_adapter_type(),
            dual_fusion=bool(self.cfg["model"].get("dual_fusion", False)),
            translation_head_mode=self.cfg["model"].get("translation_head_mode", "separate")
        )
        return model.to(self.device)

    def _is_task_feature_adapter_enabled(self):
        """Read `model.task_feature_adapter.enabled` from config, defaulting to False.

        Supports both the documented mapping form (`{"enabled": true}`) and a plain
        boolean value for convenience/backward compatibility.
        """
        task_feature_adapter_cfg = self.cfg["model"].get("task_feature_adapter", False)
        if isinstance(task_feature_adapter_cfg, dict):
            return bool(task_feature_adapter_cfg.get("enabled", False))
        return bool(task_feature_adapter_cfg)

    def _get_task_feature_adapter_type(self):
        """Read `model.task_feature_adapter.type` from config, defaulting to "channel_attention"."""
        task_feature_adapter_cfg = self.cfg["model"].get("task_feature_adapter", False)
        if isinstance(task_feature_adapter_cfg, dict):
            return task_feature_adapter_cfg.get("type", "channel_attention")
        return "channel_attention"
    
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
        
        if param_groups is None:
            param_groups = _build_named_param_groups(model, base_lr=float(self.cfg["training"]["lr"]))
        
        optimizer_name = self.cfg["training"]["optimizer"]
        base_lr = float(self.cfg["training"]["lr"])
        
        return get_optimizer(param_groups, optimizer_name=optimizer_name, lr=base_lr)
    
    def build_uncertainty_module(self, optimizer, base_lr):
        """Build task weighting module if enabled."""
        training_cfg = self.cfg.get("training", {})
        task_weighting_cfg = training_cfg.get("task_weighting", training_cfg.get("uncertainty_weighting", {}))
        use_task_weighting = bool(task_weighting_cfg.get("enabled", False))

        if not use_task_weighting:
            return None

        # type can be 'uncertainty' (learned log-variance), 'gradnorm', or 'dwa'
        utype = str(task_weighting_cfg.get("type", "uncertainty")).lower()

        class UncertaintyWrapper(nn.Module):
            """Standardised wrapper that exposes `forward` and `get_stats()`.

            get_stats() returns dict keys used by build_epoch_metrics_row:
            - uncertainty_log_var_rot
            - uncertainty_log_var_trans
            - uncertainty_weight_rot
            - uncertainty_weight_trans
            """

            def __init__(self, module, kind: str):
                super().__init__()
                self.module = module
                self.kind = kind

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

            def get_stats(self):
                if self.kind == "uncertainty":
                    try:
                        lv_rot = float(self.module.log_var_rot.detach().cpu().item())
                        lv_trans = float(self.module.log_var_trans.detach().cpu().item())
                        w_rot = float(torch.exp(-self.module.log_var_rot).detach().cpu().item())
                        w_trans = float(torch.exp(-self.module.log_var_trans).detach().cpu().item())
                        return {
                            "uncertainty_log_var_rot": lv_rot,
                            "uncertainty_log_var_trans": lv_trans,
                            "uncertainty_weight_rot": w_rot,
                            "uncertainty_weight_trans": w_trans,
                        }
                    except Exception:
                        return {}

                if self.kind == "gradnorm":
                    try:
                        weights = self.module.get_weights()
                        if weights is None:
                            return {}
                        w = weights.detach().cpu()
                        return {
                            "uncertainty_log_var_rot": "",
                            "uncertainty_log_var_trans": "",
                            "uncertainty_weight_rot": float(w[0].item()),
                            "uncertainty_weight_trans": float(w[1].item()),
                        }
                    except Exception:
                        return {}

                return {}

        if utype == "uncertainty":
            # use external implementation from losses/uncertainty_weighting.py
            from losses.uncertainty_weighting import UncertaintyWeighting

            base_module = UncertaintyWeighting(
                init_log_var_rot=float(task_weighting_cfg.get("init_log_var_rot", 0.0)),
                init_log_var_trans=float(task_weighting_cfg.get("init_log_var_trans", 0.0)),
            ).to(self.device)

            wrapper = UncertaintyWrapper(base_module, kind="uncertainty")

            params = list(wrapper.parameters())
            if params:
                uncertainty_lr_mult = float(task_weighting_cfg.get("lr_mult", 1.0))
                optimizer.add_param_group({
                    "name": "uncertainty",
                    "params": params,
                    "lr": base_lr * uncertainty_lr_mult,
                })

            return wrapper

        elif utype == "gradnorm":
            # adapt external GradNorm implementation to a simple two-loss interface
            from losses.gradnorm_pytorch import GradNormLossWeighter

            class GradNormAdapter(nn.Module):
                def __init__(self, cfg):
                    super().__init__()
                    lr = float(cfg.get("gradnorm_lr", 1e-4))
                    alpha = float(cfg.get("gradnorm_alpha", 0.0))
                    # GradNormLossWeighter doesn't expose trainable nn.Parameters by default
                    self.gn = GradNormLossWeighter(
                        num_losses=2,
                        learning_rate=lr,
                        restoring_force_alpha=alpha,
                    )

                def forward(self, loss_rot, loss_trans, activations=None, **kwargs):
                    # GradNormLossWeighter expects a 1D tensor of losses
                    losses = torch.stack([loss_rot, loss_trans])
                    # pass activations if available (optional)
                    return self.gn(losses, activations=activations, **kwargs)

                def get_weights(self):
                    # expose current renormalized loss weights if available
                    try:
                        return getattr(self.gn, "loss_weights")
                    except Exception:
                        return None

            adapter = GradNormAdapter(task_weighting_cfg).to(self.device)
            wrapper = UncertaintyWrapper(adapter, kind="gradnorm")

            params = list(wrapper.parameters())
            if params:
                uncertainty_lr_mult = float(task_weighting_cfg.get("lr_mult", 1.0))
                optimizer.add_param_group({
                    "name": "uncertainty",
                    "params": params,
                    "lr": base_lr * uncertainty_lr_mult,
                })

            return wrapper

        elif utype == "dwa":
            from losses.task_weighting import DynamicWeightAverage

            dwa_module = DynamicWeightAverage(
                temperature=float(task_weighting_cfg.get("dwa_temperature", 2.0)),
            ).to(self.device)

            wrapper = UncertaintyWrapper(dwa_module, kind="dwa")
            return wrapper

        else:
            raise ValueError(f"Unsupported uncertainty weighting type: {utype}")
    
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

        configured_data_root = self.cfg.get("data", {}).get("root", "")
        if configured_data_root:
            return os.path.abspath(configured_data_root)
        else:
            return os.path.abspath(os.path.join(self.base_dir, "../Data"))
    
    def preprocess_if_needed(self):
        """Preprocess data if needed."""
        from data_processing.preprocess_data import preprocess_data
        from data_processing.GripFusionNet_split import split_manifest_matches

        data_dir = self._get_data_dir()

        aug_cfg = self.cfg.get("data", {}).get("augmentation", {})
        use_photo_aug = bool(aug_cfg.get("enabled", False))

        # split_seed is independent of training.seed so a training.seed sweep can
        # reuse one fixed split; falls back to training.seed when unset so existing
        # configs keep their current (implicit) behavior.
        split_seed = self.cfg.get("data", {}).get(
            "split_seed", self.cfg.get("training", {}).get("seed")
        )

        split_root = os.path.join(data_dir, "Data_split")
        preprocessed_dir = os.path.join(split_root, "train", "train_preprocessing")
        cache_exists = os.path.exists(preprocessed_dir)

        # Cheap check (listdir + manifest read) so a split_seed change is picked
        # up automatically even when the preprocessing cache already exists.
        split_is_current = split_manifest_matches(data_dir, split_root, seed=split_seed)

        should_preprocess = self.force_preprocess or not cache_exists or not split_is_current

        if should_preprocess:
            if self.force_preprocess:
                reason = "--force_preprocess"
            elif not split_is_current:
                reason = "data.split_seed (or ratios/source data) changed"
            else:
                reason = "preprocessing cache missing"

            if self.logger:
                self.logger.info(f"Preprocessing data from {data_dir} ({reason})")

            preprocess_data(
                root_dir=data_dir,
                use_photometric_augmentation=use_photo_aug,
                aug_seed=int(aug_cfg.get("seed", 42)),
                aug_photo_count=int(aug_cfg.get("photo_count", 1)),
                aug_overwrite=bool(aug_cfg.get("overwrite", False)),
                aug_output_folder=str(aug_cfg.get("output_folder", "train_augmentation_default")),
                use_preaugmented_train_dir=False,
                split_seed=split_seed,
                force_split=self.force_preprocess,
            )
        elif use_photo_aug:
            if self.logger:
                self.logger.info(
                    "Photometric augmentation is applied online by the training dataloader."
                )
    
    def get_loaders(self, batch_size):
        """Get train, val, test dataloaders.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (train_loader, val_loader, test_loader, trans_norm_stats). The
            last element is the {mean_y, std_y, mean_z, std_z} dict actually used to
            normalize translation targets, computed from this run's own train split
            (see compute_translation_norm_stats) -- callers should snapshot it into
            the run's config so predictions can be un-normalized consistently later.
        """
        from data_processing.dataloader_preprocessed import get_dataloaders

        self.preprocess_if_needed()

        data_dir = self._get_data_dir()
        aug_cfg = self.cfg.get("data", {}).get("augmentation", {})
        train_loader, val_loader, test_loader, trans_norm_stats = get_dataloaders(
            root_dir=os.path.join(data_dir, "Data_split"),
            batch_size=batch_size,
            shuffle=True,
            train_folder="train",
            use_online_augmentation=bool(aug_cfg.get("enabled", False)),
            augmentation_cfg=aug_cfg,
            image_size=int(self.cfg.get("data", {}).get("image_size", 640)),
            normalize_translation=bool(self.cfg.get("data", {}).get("normalize_translation", True)),
        )

        if self.logger:
            self.logger.info(f"Loaded dataloaders with batch_size={batch_size}")

        return train_loader, val_loader, test_loader, trans_norm_stats


class RunSetupManager:
    """Manages run setup: logging, artifacts, tensorboard, manifests."""
    
    def __init__(self, config_path, cfg, batch_size, logger=None, run_directory=None):
        """
        Args:
            config_path: Path to config file
            cfg: Config dict
            batch_size: Batch size
            logger: Optional logger
            run_directory: Optional override for the run output root (default "runs")
        """
        self.config_path = config_path
        self.cfg = cfg
        self.batch_size = batch_size
        self.logger = logger
        self.artifacts = None
        self.run_directory = run_directory or "runs"

    def setup_artifacts(self):
        """Build the run artifacts directory structure.

        Does NOT write config.yml yet -- call save_config_snapshot() once cfg is
        fully resolved (e.g. after data preprocessing has filled in
        data.translation_norm_stats), so the snapshot reflects what the run
        actually trained with.
        """

        self.artifacts = build_run_artifacts(
            self.config_path,
            self.cfg,
            self.batch_size,
            output_root=self.run_directory
        )

        if self.logger:
            self.logger.info(f"Created run directory: {self.artifacts['run_dir']}")

        return self.artifacts

    def save_config_snapshot(self):
        """Write config.yml. Call after cfg has been finalized (see setup_artifacts)."""
        if self.artifacts is None:
            raise ValueError("Must call setup_artifacts() first")
        save_yaml(self.cfg, self.artifacts["config_snapshot_path"])

    def setup_tensorboard(self, tb_port, tb_root=None):
        """Setup tensorboard logging.
        
        Args:
            tb_port: TensorBoard port
            tb_root: Optional override for TB root directory
            
        Returns:
            Tuple of (tensorboard_log_dir, tensorboard_process)
        """
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

