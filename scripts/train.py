# scripts/train.py

import json
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
    GlobalPlateauLRManager,
    PerHeadEarlyStoppingManager,
    PerHeadCheckpointSelector,
    EpochMetricsAggregator,
    build_epoch_metrics_row,
    _apply_metric_accumulation
)
from utils.utils import (
    load_config,
    save_checkpoint,
    load_checkpoint_for_resume,
    strip_optimizer_state,
    get_optimizer,
)
from utils.normalization import canonicalize_translation_norm_stats
from utils.logging_utils import (
    format_fusion_weights,
    init_metrics_csv,
    init_tensorboard_writer,
    log_train_step_to_tensorboard,
    compute_model_norms,
    log_epoch_to_tensorboard,
    log_hparams_to_tensorboard,
    configure_run_logging,
    log_epoch_summary,
    write_epoch_metrics_csv,
    _init_epoch_metrics,
    close_run_writers,
)
from losses.rotation_loss_handlers import get_rotation_handler
from utils.flops_utils import compute_inference_flops

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


def _reduce_dual_fusion_branches(value):
    """Collapse a dual_fusion {"rotation": T, "translation": T} dict into one tensor.

    Averages the two task-specific branches so the existing single-fusion gate-stat/
    TensorBoard pipeline (built around one fusion module) keeps working unchanged for
    dual_fusion runs. Plain tensors/None (the non-dual_fusion case) pass through as-is.
    """
    if not isinstance(value, dict):
        return value
    branches = [v for v in value.values() if v is not None]
    if not branches:
        return None
    return torch.stack(branches, dim=0).mean(dim=0)


def _compute_signed_errors(pose_pred, pose, rotation_handler):
    """Compute signed (non-absolute) errors for translation and rotation.

    Returns: (signed_y, signed_z, signed_rot_x) as mean errors per batch.
    Positive means prediction is larger than ground truth.
    """
    trans_pred = pose_pred["trans"][:, 1:3]
    trans_gt = pose[:, 1:3]

    signed_y = (trans_pred[:, 0] - trans_gt[:, 0]).mean()
    signed_z = (trans_pred[:, 1] - trans_gt[:, 1]).mean()

    # Compute signed rotation error using the rotation handler
    # which handles all parametrization types
    if hasattr(rotation_handler, 'compute_rotation_errors_degrees_signed'):
        signed_rot_errors = rotation_handler.compute_rotation_errors_degrees_signed(pose_pred, pose)
        signed_rot_x = signed_rot_errors.mean()
    else:
        # Fallback: use absolute error if signed method not available
        signed_rot_x = torch.tensor(0.0, device=pose.device)

    return signed_y, signed_z, signed_rot_x


def _init_fusion_gate_stats():
    """Create accumulators for compact, per-stream gate diagnostics.

    Also tracks raw weights for:
    - GAP 1: Per-task fusion gates (dual_fusion)
    - GAP 2: Full distributions for TensorBoard histograms
    """
    return {
        "sum": None, "sum_sq": None, "count": None, "lt_0_1": None, "gt_0_9": None, "mid_0_4_0_6": None,
        # GAP 1: Per-task accumulators (dual_fusion only)
        "rotation_sum": None, "rotation_sum_sq": None, "rotation_count": None,
        "rotation_lt_0_1": None, "rotation_gt_0_9": None, "rotation_mid_0_4_0_6": None,
        "translation_sum": None, "translation_sum_sq": None, "translation_count": None,
        "translation_lt_0_1": None, "translation_gt_0_9": None, "translation_mid_0_4_0_6": None,
        # GAP 2: Raw weights for distributions (TensorBoard histograms)
        "raw_weights": [],  # List of tensors to concatenate later
        "raw_rotation_weights": [],
        "raw_translation_weights": [],
        # Per-(stream, channel) sums, kept separate from the pooled "sum"/"sum_sq" above
        # so the pooled std (which mixes across-sample and across-channel variation
        # into one number, see _compute_between_gate_stds) can be split into the two.
        # "n" is the number of SAMPLES accumulated (not samples*channels like "count").
        "csum": None, "csumsq": None, "n": None,
        "rotation_csum": None, "rotation_csumsq": None, "rotation_n": None,
        "translation_csum": None, "translation_csumsq": None, "translation_n": None,
        # Fusion module's own parameter/gradient norm (train-only: accumulated once per
        # training step right after backward(), see _accumulate_fusion_grad_norm).
        "grad_norm_sum": 0.0, "param_norm_sum": 0.0, "grad_norm_count": 0,
        "rotation_grad_norm_sum": 0.0, "rotation_param_norm_sum": 0.0, "rotation_grad_norm_count": 0,
        "translation_grad_norm_sum": 0.0, "translation_param_norm_sum": 0.0, "translation_grad_norm_count": 0,
        # Per-scale (P3/P4/P5) mean gate value per stream, keyed by (task_name, scale).
        # Only populated when scale_dims is passed to _accumulate_fusion_gate_stats
        # (channel-wise, 1D-pipeline gates only -- see PoseEstimationNet.stream_scale_dims).
        "scale_sum": {},
        "scale_count": {},
        # Per-scale specialization (fraction of that scale's channels with gate <0.1 /
        # >0.9), keyed the same way -- lets a per-scale heatmap show *decisiveness*,
        # not just average trust, per pyramid scale.
        "scale_lt_0_1": {},
        "scale_gt_0_9": {},
        "scale_elem_count": {},
    }


def _accumulate_fusion_gate_stats(stats, gates, task_name=None, scale_dims=None):
    """
    Accumulate gate statistics for per-stream analysis.

    Args:
        stats: Accumulator dict (from _init_fusion_gate_stats)
        gates: Fusion gate weights, shape (B, streams) or (B, streams, channels) or (B, streams, H, W)
               Can also be None (skipped) or a dict with 'rotation'/'translation' keys for dual_fusion
        task_name: Optional task identifier ('rotation' or 'translation') for dual_fusion.
                   If None, accumulates to the main stats (non-dual case or averaged case).
        scale_dims: Optional dict {"p3": size, "p4": size, "p5": size} giving the flattened
                    per-scale channel counts within `gates`' last dim (see
                    PoseEstimationNet.stream_scale_dims). When the gate shape matches
                    (channel-wise, 1D pipeline), also accumulates a per-scale, per-stream
                    mean so gate trust can be broken down by pyramid scale.
    """
    if gates is None:
        return

    # GAP 2: Store raw weights for TensorBoard histogram
    raw_key = f"raw_{task_name}_weights" if task_name else "raw_weights"
    if raw_key in stats and gates is not None:
        gates_cpu = gates.detach().float().cpu()
        stats[raw_key].append(gates_cpu)

    # Per-scale breakdown: only meaningful for channel-wise gates from the 1D pipeline,
    # i.e. shape (B, streams, channels) with channels == p3 + p4 + p5.
    if scale_dims and gates.dim() == 3 and gates.shape[-1] == sum(scale_dims.values()):
        gates_cpu = gates.detach().float().cpu()
        offset = 0
        for scale, size in scale_dims.items():
            chunk = gates_cpu[:, :, offset:offset + size]  # (B, streams, size)
            offset += size
            chunk_mean = chunk.mean(dim=(0, 2))  # (streams,) -- mean over batch and this scale's channels
            chunk_lt = (chunk < 0.1).sum(dim=(0, 2)).float()  # (streams,)
            chunk_gt = (chunk > 0.9).sum(dim=(0, 2)).float()  # (streams,)
            chunk_elems = chunk.shape[0] * chunk.shape[2]
            key = (task_name, scale)
            if key not in stats["scale_sum"]:
                stats["scale_sum"][key] = chunk_mean.clone()
                stats["scale_count"][key] = 1
                stats["scale_lt_0_1"][key] = chunk_lt.clone()
                stats["scale_gt_0_9"][key] = chunk_gt.clone()
                stats["scale_elem_count"][key] = chunk_elems
            else:
                stats["scale_sum"][key] += chunk_mean
                stats["scale_count"][key] += 1
                stats["scale_lt_0_1"][key] += chunk_lt
                stats["scale_gt_0_9"][key] += chunk_gt
                stats["scale_elem_count"][key] += chunk_elems

    # Gate shapes are (B, streams) or (B, streams, channels) or (B, streams, H, W).
    # Reshape to (num_streams, -1) so each row is all gate values for one stream.
    gates_bsd = gates.detach().float().cpu().reshape(gates.shape[0], gates.shape[1], -1)  # (B, streams, D)
    by_stream = gates_bsd.movedim(1, 0).reshape(gates.shape[1], -1)
    gate_sum = by_stream.sum(dim=1)
    gate_sum_sq = by_stream.square().sum(dim=1)
    gate_count = torch.full_like(gate_sum, by_stream.shape[1])
    low_count = (by_stream < 0.1).sum(dim=1)
    high_count = (by_stream > 0.9).sum(dim=1)
    # Threshold-free complement to lt_0_1/gt_0_9: how much of the gate sits near-undecided
    # (~0.5) rather than committed to either extreme -- distinguishes a gate that's
    # genuinely uncertain from one that's bimodally split, which lt_0_1+gt_0_9 alone
    # can't tell apart.
    mid_count = ((by_stream >= 0.4) & (by_stream <= 0.6)).sum(dim=1)

    # Per-(stream, channel) sums over the SAMPLE dimension only (not pooled over
    # channels like gate_sum above) -- lets _summarize_fusion_gate_stats later split
    # the pooled std into across-sample vs. across-channel variation.
    csum_batch = gates_bsd.sum(dim=0)              # (streams, D)
    csumsq_batch = gates_bsd.square().sum(dim=0)   # (streams, D)
    n_batch = gates_bsd.shape[0]                   # number of samples in this batch

    # Determine which stats dict keys to update
    if task_name:
        # GAP 1: Per-task accumulation (dual_fusion)
        sum_key = f"{task_name}_sum"
        sum_sq_key = f"{task_name}_sum_sq"
        count_key = f"{task_name}_count"
        lt_key = f"{task_name}_lt_0_1"
        gt_key = f"{task_name}_gt_0_9"
        mid_key = f"{task_name}_mid_0_4_0_6"
        csum_key, csumsq_key, n_key = f"{task_name}_csum", f"{task_name}_csumsq", f"{task_name}_n"
    else:
        # Standard accumulation (non-dual or averaged)
        sum_key, sum_sq_key = "sum", "sum_sq"
        count_key = "count"
        lt_key, gt_key = "lt_0_1", "gt_0_9"
        mid_key = "mid_0_4_0_6"
        csum_key, csumsq_key, n_key = "csum", "csumsq", "n"

    if stats[sum_key] is None:
        stats.update({
            sum_key: gate_sum,
            sum_sq_key: gate_sum_sq,
            count_key: gate_count,
            lt_key: low_count,
            gt_key: high_count,
            mid_key: mid_count,
            csum_key: csum_batch,
            csumsq_key: csumsq_batch,
            n_key: n_batch,
        })
        return

    stats[csum_key] += csum_batch
    stats[csumsq_key] += csumsq_batch
    stats[n_key] += n_batch

    for key, value in {
        sum_key: gate_sum,
        sum_sq_key: gate_sum_sq,
        count_key: gate_count,
        lt_key: low_count,
        gt_key: high_count,
        mid_key: mid_count,
    }.items():
        stats[key] += value


def _compute_between_gate_stds(csum, csumsq, n):
    """Split a pooled gate std into variation ACROSS SAMPLES vs. ACROSS CHANNELS.

    The existing pooled `*_fusion_gate_std` (see _accumulate_fusion_gate_stats) flattens
    every (sample, channel) gate value together, so it can't tell a gate that varies
    per-sample (real per-sample trust) apart from a static per-channel pattern (channel 5
    always ~0.3, channel 6 always ~0.7, identical for every sample) -- both inflate that
    one pooled number identically. This computes, from the per-(stream, channel) sums
    accumulated over the sample dimension:
      between_sample_std: for each channel, the std of its gate value across samples,
        averaged over channels -- 0 only if every channel is a per-sample constant.
      between_channel_std: the std, across channels, of each channel's own across-sample
        mean -- 0 only if every channel behaves identically on average (a scalar gate,
        D=1, is always exactly 0 here by construction).
    Both are (streams,) tensors, matching the pooled std's shape.
    """
    cmean = csum / n                                          # (streams, D)
    cvar = (csumsq / n - cmean.square()).clamp_min(0.0)        # (streams, D)
    between_sample_std = cvar.sqrt().mean(dim=1)
    between_channel_std = cmean.std(dim=1, unbiased=False)
    return between_sample_std, between_channel_std


def _accumulate_fusion_grad_norm(stats, model):
    """Accumulate the fusion module's OWN parameter/gradient norm for one training step.

    scripts/train.py has no per-module gradient visibility: log_train_step_to_tensorboard's
    compute_model_norms(model) sums over the WHOLE model, which can't show whether the
    fusion module specifically is receiving a healthy gradient or a vanishing one (the
    hypothesized cause of gated_concat's near-constant gate: a cold-start, +/-1e-3
    initialized last layer that starts with little gradient reaching its hidden units).

    Call once per training step, after loss.backward() (+ grad clipping) so .grad is
    populated, and never during validation (there is no backward pass to measure).
    """
    if model.dual_fusion:
        for task_name, module in (("rotation", model.fusion_rotation), ("translation", model.fusion_translation)):
            param_norm, grad_norm = compute_model_norms(module)
            stats[f"{task_name}_grad_norm_sum"] += grad_norm
            stats[f"{task_name}_param_norm_sum"] += param_norm
            stats[f"{task_name}_grad_norm_count"] += 1
    else:
        param_norm, grad_norm = compute_model_norms(model.fusion)
        stats["grad_norm_sum"] += grad_norm
        stats["param_norm_sum"] += param_norm
        stats["grad_norm_count"] += 1


def _summarize_fusion_gate_stats(stats, prefix):
    """
    Return CSV-ready gate statistics and raw weights for TensorBoard.

    Computes:
    - Mean, std, and sparsity fractions (existing per-stream scalars)
    - Percentiles p25, p50, p75, p90 (GAP 2: Option A for CSV)
    - Raw weights (GAP 2: Option B for TensorBoard histograms)
    - Per-task stats if dual_fusion (GAP 1)

    Args:
        stats: Accumulator dict from _init_fusion_gate_stats with sums, counts, raw weights
        prefix: "train" or "val" for CSV column names

    Returns:
        (summary_dict, mean_weights, raw_weights, raw_rotation_weights, raw_translation_weights)
        where each raw_* is either a concatenated tensor or None
    """
    # `sum` (the non-dual/main accumulator) is only ever populated on the non-dual_fusion
    # path (see call sites below); dual_fusion runs only fill rotation_sum/translation_sum.
    # Previously this function early-returned an empty summary whenever `sum` was None,
    # which silently discarded the rotation/translation stats too for every dual_fusion
    # run (confirmed empirically: dual_fusion runs' metrics.csv had every
    # *_fusion_gate_* column blank, main AND per-task). Guard the main block on `sum`
    # instead, so per-task and per-scale stats still populate when only they are present.
    if stats["sum"] is None and stats.get("rotation_sum") is None and stats.get("translation_sum") is None:
        return {}, None, None, None, None

    summary = {}
    mean = None

    # Main (averaged or non-dual) stats
    if stats["sum"] is not None:
        mean = stats["sum"] / stats["count"]
        variance = (stats["sum_sq"] / stats["count"] - mean.square()).clamp_min(0.0)
        summary.update({
            f"{prefix}_fusion_gate_mean": format_fusion_weights(mean),
            f"{prefix}_fusion_gate_std": format_fusion_weights(variance.sqrt()),
            f"{prefix}_fusion_gate_fraction_lt_0_1": format_fusion_weights(stats["lt_0_1"] / stats["count"]),
            f"{prefix}_fusion_gate_fraction_gt_0_9": format_fusion_weights(stats["gt_0_9"] / stats["count"]),
            f"{prefix}_fusion_gate_fraction_mid_0_4_0_6": format_fusion_weights(stats["mid_0_4_0_6"] / stats["count"]),
        })
        if stats.get("csum") is not None:
            between_sample_std, between_channel_std = _compute_between_gate_stds(
                stats["csum"], stats["csumsq"], stats["n"]
            )
            summary[f"{prefix}_fusion_gate_between_sample_std"] = format_fusion_weights(between_sample_std)
            summary[f"{prefix}_fusion_gate_between_channel_std"] = format_fusion_weights(between_channel_std)

    # Fusion module's own gradient/parameter norm (train-only, see
    # _accumulate_fusion_grad_norm -- val never calls it, so grad_norm_count stays 0).
    if stats.get("grad_norm_count"):
        summary[f"{prefix}_fusion_grad_norm"] = stats["grad_norm_sum"] / stats["grad_norm_count"]
        summary[f"{prefix}_fusion_param_norm"] = stats["param_norm_sum"] / stats["grad_norm_count"]
    for task_name in ("rotation", "translation"):
        count = stats.get(f"{task_name}_grad_norm_count")
        if count:
            summary[f"{prefix}_{task_name}_fusion_grad_norm"] = stats[f"{task_name}_grad_norm_sum"] / count
            summary[f"{prefix}_{task_name}_fusion_param_norm"] = stats[f"{task_name}_param_norm_sum"] / count

    # GAP 2: Option A - Compute and add percentiles to CSV
    from utils.logging_utils import compute_gate_statistics
    raw_weights = torch.cat(stats["raw_weights"]) if stats["raw_weights"] else None
    if raw_weights is not None:
        gate_stats = compute_gate_statistics(raw_weights)
        summary.update({
            f"{prefix}_fusion_gate_p25": gate_stats["p25"],
            f"{prefix}_fusion_gate_p50": gate_stats["p50"],
            f"{prefix}_fusion_gate_p75": gate_stats["p75"],
            f"{prefix}_fusion_gate_p90": gate_stats["p90"],
        })

    # GAP 1: Per-task stats (dual_fusion only)
    raw_rotation_weights = torch.cat(stats["raw_rotation_weights"]) if stats["raw_rotation_weights"] else None
    raw_translation_weights = torch.cat(stats["raw_translation_weights"]) if stats["raw_translation_weights"] else None

    if raw_rotation_weights is not None:
        rotation_stats = compute_gate_statistics(raw_rotation_weights)
        summary.update({
            f"{prefix}_rotation_fusion_gate_mean": rotation_stats["mean"],
            f"{prefix}_rotation_fusion_gate_std": rotation_stats["std"],
            f"{prefix}_rotation_fusion_gate_fraction_lt_0_1": rotation_stats["fraction_lt_0_1"],
            f"{prefix}_rotation_fusion_gate_fraction_gt_0_9": rotation_stats["fraction_gt_0_9"],
            f"{prefix}_rotation_fusion_gate_fraction_mid_0_4_0_6": rotation_stats["fraction_mid_0_4_0_6"],
        })
        if stats.get("rotation_csum") is not None:
            between_sample_std, between_channel_std = _compute_between_gate_stds(
                stats["rotation_csum"], stats["rotation_csumsq"], stats["rotation_n"]
            )
            summary[f"{prefix}_rotation_fusion_gate_between_sample_std"] = format_fusion_weights(between_sample_std)
            summary[f"{prefix}_rotation_fusion_gate_between_channel_std"] = format_fusion_weights(between_channel_std)

    if raw_translation_weights is not None:
        translation_stats = compute_gate_statistics(raw_translation_weights)
        summary.update({
            f"{prefix}_translation_fusion_gate_mean": translation_stats["mean"],
            f"{prefix}_translation_fusion_gate_std": translation_stats["std"],
            f"{prefix}_translation_fusion_gate_fraction_lt_0_1": translation_stats["fraction_lt_0_1"],
            f"{prefix}_translation_fusion_gate_fraction_gt_0_9": translation_stats["fraction_gt_0_9"],
            f"{prefix}_translation_fusion_gate_fraction_mid_0_4_0_6": translation_stats["fraction_mid_0_4_0_6"],
        })
        if stats.get("translation_csum") is not None:
            between_sample_std, between_channel_std = _compute_between_gate_stds(
                stats["translation_csum"], stats["translation_csumsq"], stats["translation_n"]
            )
            summary[f"{prefix}_translation_fusion_gate_between_sample_std"] = format_fusion_weights(between_sample_std)
            summary[f"{prefix}_translation_fusion_gate_between_channel_std"] = format_fusion_weights(between_channel_std)

    # Per-scale (P3/P4/P5) mean + specialization gate value per stream -- only present
    # for channel-wise gates from the 1D pipeline (see PoseEstimationNet.stream_scale_dims
    # and the scale_dims argument to _accumulate_fusion_gate_stats).
    for (task_name, scale), total in stats["scale_sum"].items():
        count = stats["scale_count"][(task_name, scale)]
        scale_mean = total / count
        col_prefix = f"{prefix}_{task_name}_" if task_name else f"{prefix}_"
        summary[f"{col_prefix}fusion_gate_{scale}_mean"] = format_fusion_weights(scale_mean)

        elem_count = stats["scale_elem_count"][(task_name, scale)]
        scale_lt = stats["scale_lt_0_1"][(task_name, scale)] / elem_count
        scale_gt = stats["scale_gt_0_9"][(task_name, scale)] / elem_count
        summary[f"{col_prefix}fusion_gate_{scale}_specialization"] = format_fusion_weights(scale_lt + scale_gt)

    return summary, mean, raw_weights, raw_rotation_weights, raw_translation_weights


def _save_raw_gate_weights(checkpoint_path, epoch, raw_weights, raw_rot_weights, raw_trans_weights):
    """
    Persist the raw (unaggregated, per-channel) validation gate-weight tensors
    for the best-so-far epoch, alongside checkpoint_best.pth.

    metrics.csv only ever stores per-stream aggregates (mean/std/fraction/
    percentile), and TensorBoard's fusion/*_weights_distribution histograms
    (see utils/logging_utils.py) can end up under a different root than
    runs/<run>/tensorboard/ depending on how training was launched (e.g.
    train_multiple_runs.py redirects --tensorboard_root to
    logs/tensorboard/<session_tag>/). Saving the raw tensor directly next to
    the checkpoint gives a source of truth for exact per-channel gate
    analysis (e.g. true violin plots) that doesn't depend on locating or
    parsing TensorBoard event files.
    """
    if raw_weights is None and raw_rot_weights is None and raw_trans_weights is None:
        return

    payload = {"epoch": epoch}
    if raw_weights is not None:
        payload["val_gate_weights"] = raw_weights
    if raw_rot_weights is not None:
        payload["val_rotation_gate_weights"] = raw_rot_weights
    if raw_trans_weights is not None:
        payload["val_translation_gate_weights"] = raw_trans_weights

    save_path = os.path.join(os.path.dirname(checkpoint_path), "gate_weights_best_epoch.pt")
    torch.save(payload, save_path)


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
    trans_norm_stats=None,
    grad_clip_max_norm=None,
    grad_clip_per_parameter_group=False,
):
    model.train()
    stats = canonicalize_translation_norm_stats(trans_norm_stats)
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
        fusion_weights = _reduce_dual_fusion_branches(fusion_weights)
        fusion_targets = _reduce_dual_fusion_branches(model.last_fusion_targets)

        loss, loss_q, loss_y, loss_z, _ = rotation_handler.compute_loss(
            pose_pred, pose, uncertainty_module
        )
        # Only clone tensor values -- pose_pred may carry non-tensor extras
        # (e.g. "task_feature_adapter_attention" is a dict of attention
        # tensors, not a tensor itself, when model.task_feature_adapter is
        # enabled; see PoseEstimationNet.forward).
        pose_pred_unnorm = {k: v.clone() for k, v in pose_pred.items() if torch.is_tensor(v)}
        if normalize_translation:
            pose_pred_unnorm["trans"][:, 1] = pose_pred_unnorm["trans"][:, 1] * stats["std_y"] + stats["mean_y"]
            pose_pred_unnorm["trans"][:, 2] = pose_pred_unnorm["trans"][:, 2] * stats["std_z"] + stats["mean_z"]
            # "pose" ([x,y,z,qx,qy,qz,qw]) is a separate tensor from "trans" (see
            # PoseEstimationNet.forward's torch.cat) -- QuaternionHandler.compute_metrics
            # reads "pose" directly, so it must be un-normalized too or metrics compare
            # a normalized prediction against unnormalized ground truth.
            pose_pred_unnorm["pose"][:, 1] = pose_pred_unnorm["pose"][:, 1] * stats["std_y"] + stats["mean_y"]
            pose_pred_unnorm["pose"][:, 2] = pose_pred_unnorm["pose"][:, 2] * stats["std_z"] + stats["mean_z"]

        pose_unnorm = pose.clone()
        if normalize_translation:
            pose_unnorm[:, 1] = pose_unnorm[:, 1] * stats["std_y"] + stats["mean_y"]
            pose_unnorm[:, 2] = pose_unnorm[:, 2] * stats["std_z"] + stats["mean_z"]

        abs_trans_y, abs_trans_z, abs_trans, abs_rot_x = rotation_handler.compute_metrics(
            pose_pred_unnorm, pose_unnorm
        )

        # Compute signed errors for this batch
        signed_y, signed_z, signed_rot_x = _compute_signed_errors(pose_pred_unnorm, pose_unnorm, rotation_handler)

        # GAP 1: Accumulate per-task fusion gates if dual_fusion, else averaged
        scale_dims = getattr(model, "stream_scale_dims", None)
        if isinstance(model.last_fusion_weights, dict) and model.dual_fusion:
            # Dual fusion: accumulate rotation and translation gates separately
            _accumulate_fusion_gate_stats(fusion_gate_stats, model.last_fusion_weights.get("rotation"), task_name="rotation", scale_dims=scale_dims)
            _accumulate_fusion_gate_stats(fusion_gate_stats, model.last_fusion_weights.get("translation"), task_name="translation", scale_dims=scale_dims)
        else:
            # Non-dual or averaged: use existing behavior
            _accumulate_fusion_gate_stats(fusion_gate_stats, _reduce_dual_fusion_branches(model.last_fusion_weights), scale_dims=scale_dims)

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
                "signed_y": signed_y.item(),
                "signed_z": signed_z.item(),
                "signed_rot_x": signed_rot_x.item(),
            },
        )

        avg_loss_so_far = train_metrics["loss"] / (tqdm_bar.n + 1)
        tqdm_bar.set_postfix(
            avg_loss=f"{avg_loss_so_far:.4f}", 
            curr_loss=f"{loss.item():.4f}", 
            q_wt=f"{loss_q.item():.4f}"
        )

        if not (
            uncertainty_module is not None
            and getattr(uncertainty_module, "kind", None) == "gradnorm"
        ):
            loss.backward()

        if grad_clip_max_norm is not None:
            if grad_clip_per_parameter_group:
                # Clip each optimizer param group's own gradient norm independently to
                # max_norm, instead of one norm over every parameter together. Motivation:
                # the head's first Linear layer alone holds ~90% of this model's parameters
                # (see CLAUDE.md's "Gate granularity / naming" note), so under a single
                # global clip it can dominate the total norm and throttle every other
                # group's (backbone/gate/other head) update by the same factor whenever its
                # own gradient is large -- see the gate/feature-shrinkage discussion this
                # targets. Note: unlike the global clip below (torch.nn.utils.clip_grad_norm_
                # over model.parameters()), this also clips the "uncertainty" param group
                # when task_weighting.type=uncertainty (its log_var params live on the
                # separate uncertainty_module, added via optimizer.add_param_group, and were
                # never part of model.parameters()) -- a deliberate side effect, not an
                # oversight: those two scalars should arguably be clipped too.
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group["params"], max_norm=grad_clip_max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

        _accumulate_fusion_grad_norm(fusion_gate_stats, model)

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
                uncertainty_stats = (
                    uncertainty_module.get_stats() if hasattr(uncertainty_module, "get_stats") else None
                )
                if uncertainty_stats:
                    # log log-variances if present
                    lv_rot = uncertainty_stats.get("uncertainty_log_var_rot", None)
                    lv_trans = uncertainty_stats.get("uncertainty_log_var_trans", None)
                    if lv_rot is not None and lv_rot != "":
                        tb_writer.add_scalar("uncertainty/step_log_var_rot", float(lv_rot), global_batch_step)
                    if lv_trans is not None and lv_trans != "":
                        tb_writer.add_scalar("uncertainty/step_log_var_trans", float(lv_trans), global_batch_step)

                    # log weights if present
                    w_rot = uncertainty_stats.get("uncertainty_weight_rot", None)
                    w_trans = uncertainty_stats.get("uncertainty_weight_trans", None)
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
    fusion_gate_summary, fusion_weights_avg, train_raw_weights, train_raw_rot_weights, train_raw_trans_weights = _summarize_fusion_gate_stats(
        fusion_gate_stats, prefix="train"
    )

    return {
        "train_avg": train_avg,
        "fusion_weights": fusion_weights,
        "fusion_weights_avg": fusion_weights_avg,
        "fusion_gate_summary": fusion_gate_summary,
        "global_batch_step": global_batch_step,
        # GAP 2: Raw weights for TensorBoard histograms
        "train_raw_weights": train_raw_weights,
        "train_raw_rot_weights": train_raw_rot_weights,
        "train_raw_trans_weights": train_raw_trans_weights,
    }


def validation_loop(
    model,
    val_loader,
    rotation_handler,
    device,
    uncertainty_module=None,
    normalize_translation=True,
    trans_norm_stats=None,
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
    stats = canonicalize_translation_norm_stats(trans_norm_stats)
    val_metrics = _init_epoch_metrics()
    fusion_gate_stats = _init_fusion_gate_stats()
    rotation_errors = []
    uncertainty_for_loss = (
        None
        if uncertainty_module is not None and getattr(uncertainty_module, "kind", None) == "gradnorm"
        else uncertainty_module
    )
    
    num_val = len(val_loader)
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            aux_losses = batch.pop("aux_losses", None)
            pose = batch["pose"].to(device)
            pose_pred, _ = model(batch, aux_losses=aux_losses)

            # GAP 1: Accumulate per-task fusion gates if dual_fusion, else averaged
            scale_dims = getattr(model, "stream_scale_dims", None)
            if isinstance(model.last_fusion_weights, dict) and model.dual_fusion:
                # Dual fusion: accumulate rotation and translation gates separately
                _accumulate_fusion_gate_stats(fusion_gate_stats, model.last_fusion_weights.get("rotation"), task_name="rotation", scale_dims=scale_dims)
                _accumulate_fusion_gate_stats(fusion_gate_stats, model.last_fusion_weights.get("translation"), task_name="translation", scale_dims=scale_dims)
            else:
                # Non-dual or averaged: use existing behavior
                _accumulate_fusion_gate_stats(fusion_gate_stats, _reduce_dual_fusion_branches(model.last_fusion_weights), scale_dims=scale_dims)
            
            # Compute loss and metrics using rotation handler
            loss, loss_q, loss_y, loss_z, loss_trans = rotation_handler.compute_loss(
                pose_pred, pose, uncertainty_for_loss
            )

            # Only clone tensor values -- see matching comment in the train loop above.
            pose_pred_unnorm = {k: v.clone() for k, v in pose_pred.items() if torch.is_tensor(v)}
            if normalize_translation:
                pose_pred_unnorm["trans"][:, 1] = pose_pred_unnorm["trans"][:, 1] * stats["std_y"] + stats["mean_y"]
                pose_pred_unnorm["trans"][:, 2] = pose_pred_unnorm["trans"][:, 2] * stats["std_z"] + stats["mean_z"]
                # "pose" is a separate tensor from "trans" (see forward()'s torch.cat);
                # QuaternionHandler.compute_metrics reads "pose" directly, so it must be
                # un-normalized too -- see matching comment in _train_single_epoch.
                pose_pred_unnorm["pose"][:, 1] = pose_pred_unnorm["pose"][:, 1] * stats["std_y"] + stats["mean_y"]
                pose_pred_unnorm["pose"][:, 2] = pose_pred_unnorm["pose"][:, 2] * stats["std_z"] + stats["mean_z"]

            pose_unnorm = pose.clone()
            if normalize_translation:
                pose_unnorm[:, 1] = pose_unnorm[:, 1] * stats["std_y"] + stats["mean_y"]
                pose_unnorm[:, 2] = pose_unnorm[:, 2] * stats["std_z"] + stats["mean_z"]

            abs_trans_y, abs_trans_z, abs_trans, abs_rot_x = rotation_handler.compute_metrics(
                pose_pred_unnorm, pose_unnorm
            )
            rotation_errors.append(
                rotation_handler.compute_rotation_errors_degrees(pose_pred_unnorm, pose_unnorm).detach().cpu()
            )

            # Compute signed errors for this batch
            signed_y, signed_z, signed_rot_x = _compute_signed_errors(pose_pred_unnorm, pose_unnorm, rotation_handler)

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
                    "signed_y": signed_y.item(),
                    "signed_z": signed_z.item(),
                    "signed_rot_x": signed_rot_x.item(),
                },
            )
    
    val_gate_summary, _, val_raw_weights, val_raw_rot_weights, val_raw_trans_weights = _summarize_fusion_gate_stats(fusion_gate_stats, prefix="val")
    all_rotation_errors = torch.cat(rotation_errors)
    rotation_summary = {
        "val_rot_best_10_percent_deg": torch.quantile(all_rotation_errors, 0.1).item(),
        "val_rot_median_deg": torch.median(all_rotation_errors).item(),
        "val_rot_p90_deg": torch.quantile(all_rotation_errors, 0.9).item(),
        "val_rot_x_best_10_percent_deg": torch.quantile(all_rotation_errors, 0.1).item(),  # Alias for X-rotation only (planar case)
        "val_rot_x_median_deg": torch.median(all_rotation_errors).item(),
        "val_rot_x_p90_deg": torch.quantile(all_rotation_errors, 0.9).item(),
    }
    return (
        EpochMetricsAggregator(val_metrics, num_val).to_dict(),
        val_gate_summary,
        rotation_summary,
        # GAP 2: Raw weights for TensorBoard histograms
        val_raw_weights,
        val_raw_rot_weights,
        val_raw_trans_weights,
    )


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
    checkpoint_selection_cfg=None,
    rotation_loss_type="geodesic",
    normalize_translation=True,
    trans_norm_stats=None,
    grad_clip_max_norm=None,
    grad_clip_per_parameter_group=False,
    resume_state=None,
    rot_scaling_factor=None,
    checkpoint_diagnostic=False,
):
    # resume_state is the raw checkpoint dict from load_checkpoint_for_resume()
    # (model/optimizer/uncertainty weights are already restored by that point --
    # this only carries the pieces that don't exist until the managers below are
    # constructed). None means a normal from-scratch run.
    best_val_loss = float(resume_state["best_val_loss"]) if resume_state and "best_val_loss" in resume_state else float("inf")
    if resume_state is not None:
        last_epoch = int(resume_state["epoch"]) + 1
    csv_file = None
    csv_writer = None
    tb_writer = None
    metadata = run_metadata or {}
    global_batch_step = 0

    # Diagnostic only, opt-in (training.checkpoint_diagnostic) -- does NOT change which
    # checkpoint is kept (that's still decided by val_fixed_loss a few lines below,
    # unchanged). Tracks, in parallel, which epoch would have been "best" by the raw,
    # unweighted per-head metrics alone (val_abs_rot_x / val_abs_trans), so a run's
    # manifest.json records whether checkpoint_best.pth's selection (by val_fixed_loss)
    # agrees with what the interpretable accuracy metrics alone would have picked --
    # see the "Fixed-scale, non-learned composite ... used for checkpoint selection"
    # comment below for why val_fixed_loss is used instead of these metrics directly
    # (not to replace that choice, just to make any divergence from it visible/traceable
    # per run instead of requiring a manual replay against metrics.csv after the fact).
    checkpoint_diagnostic = bool(checkpoint_diagnostic)
    diag_best = {
        "rot": (None, float("inf")), "trans": (None, float("inf")), "checkpoint": (None, None, None),
    } if checkpoint_diagnostic else None

    lr_scale_fn = _build_epoch_lr_scale_fn(num_epochs, lr_schedule_cfg)

    param_groups_by_name = {
        group.get("name", f"group_{idx}"): group
        for idx, group in enumerate(optimizer.param_groups)
    }
    
    # Initialize head LR decay manager
    head_lr_decay_cfg = head_lr_decay_cfg or {}
    head_lr_manager = HeadLRDecayManager(
        head_lr_decay_cfg=head_lr_decay_cfg,
        optimizer=optimizer,
        param_groups_by_name=param_groups_by_name,
    )

    # Initialize global plateau LR manager (no-op unless lr_schedule.type == reduce_on_plateau)
    global_plateau_manager = GlobalPlateauLRManager(
        lr_schedule_cfg=lr_schedule_cfg,
        optimizer=optimizer,
        param_groups_by_name=param_groups_by_name,
    )

    # Initialize early stopping manager (no-op when training.early_stopping.enabled is false).
    # Per-head (val_abs_rot_x / val_abs_trans, AND-gated) instead of the composite val_loss --
    # see PerHeadEarlyStoppingManager docstring for why.
    early_stopper = PerHeadEarlyStoppingManager(early_stopping_cfg)

    # Checkpoint-save criterion. Default ("combined_loss", the only behavior that
    # ever existed before) keeps comparing val_fixed_loss below, unchanged. Opt-in
    # ("per_head_pareto") switches checkpoint_best.pth itself to PerHeadCheckpointSelector
    # -- see its docstring. "both" keeps checkpoint_best.pth on the original combined_loss
    # criterion (so resume/test.py behavior is untouched) and additionally writes a
    # second file, checkpoint_best_per_head.pth, whenever the pareto criterion fires --
    # for A/B comparing the two selection strategies on one run before committing to
    # switching the primary checkpoint over.
    checkpoint_selection_cfg = checkpoint_selection_cfg or {}
    checkpoint_selection_type = checkpoint_selection_cfg.get("type", "combined_loss")
    use_pareto_for_primary = checkpoint_selection_type == "per_head_pareto"
    save_dual_checkpoints = checkpoint_selection_type == "both"
    checkpoint_selector = (
        PerHeadCheckpointSelector(checkpoint_selection_cfg)
        if use_pareto_for_primary or save_dual_checkpoints
        else None
    )
    secondary_checkpoint_path = (
        os.path.join(os.path.dirname(checkpoint_path), "checkpoint_best_per_head.pth")
        if save_dual_checkpoints and checkpoint_path is not None
        else None
    )

    if resume_state is not None:
        head_lr_manager.load_state_dict(resume_state.get("head_lr_manager_state"))
        global_plateau_manager.load_state_dict(resume_state.get("global_plateau_manager_state"))
        early_stopper.load_state_dict(resume_state.get("early_stopper_state"))
        if checkpoint_selector is not None:
            checkpoint_selector.load_state_dict(resume_state.get("checkpoint_selector_state"))
        logger.info(
            f"Resumed from checkpoint at epoch={resume_state['epoch']} "
            f"(continuing from epoch={last_epoch}), best_val_loss={best_val_loss:.6f}"
        )

    csv_file, csv_writer = init_metrics_csv(metrics_csv_path, append=(resume_state is not None))
    tb_writer = init_tensorboard_writer(tensorboard_log_dir, metadata, logger)
    
    # Create rotation loss handler for both training and validation
    rotation_handler = get_rotation_handler(
        model.rotation_parametrisation,
        criterion=criterion,
        rotation_loss_type=rotation_loss_type,
        rot_scaling_factor=rot_scaling_factor,
    )

    try:
        for epoch in range(last_epoch, num_epochs):
            # Base schedule (warmup_cosine or reduce_on_plateau) produces this epoch's
            # per-group reference LR; per-head decay then multiplies its own persistent
            # factor on top, so it composes with either base schedule.
            if global_plateau_manager.enabled:
                reference_lrs = global_plateau_manager.reference_lrs()
                head_lr_manager.apply_reference_lrs(reference_lrs)
                current_lr_scale = float(global_plateau_manager.multiplier_view.param_groups[0]["lr"])
            else:
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
                trans_norm_stats=trans_norm_stats,
                grad_clip_max_norm=grad_clip_max_norm,
                grad_clip_per_parameter_group=grad_clip_per_parameter_group,
            )
            global_batch_step = train_epoch["global_batch_step"]
            train_avg = train_epoch["train_avg"]
            fusion_weights = train_epoch["fusion_weights"]
            fusion_weights_avg = train_epoch["fusion_weights_avg"]
            train_gate_summary = train_epoch["fusion_gate_summary"]
            epoch_time = time.perf_counter() - epoch_start

            fw_str = format_fusion_weights(fusion_weights_avg) if fusion_weights_avg is not None else format_fusion_weights(fusion_weights)
            fw_epoch_str = fw_str

            val_avg, val_gate_summary, val_rotation_summary, val_raw_weights, val_raw_rot_weights, val_raw_trans_weights = validation_loop(
                model=model,
                val_loader=val_loader,
                rotation_handler=rotation_handler,
                device=device,
                uncertainty_module=uncertainty_module,
                normalize_translation=normalize_translation,
                trans_norm_stats=trans_norm_stats,
            )

            lr_base = _get_lr_by_group(optimizer, "base")
            lr_head_rot = _get_lr_by_group(optimizer, "head_rot")
            lr_head_y = _get_lr_by_group(optimizer, "head_y")
            lr_head_z = _get_lr_by_group(optimizer, "head_z")
            lr_fusion = _get_lr_by_group(optimizer, "fusion") if "fusion" in param_groups_by_name else None
            uncertainty_stats = _compute_uncertainty_stats(uncertainty_module)

            head_lr_manager.update(
                train_metrics={
                    "loss_q": train_avg["loss_q"],
                    "loss_y": train_avg["loss_y"],
                    "loss_z": train_avg["loss_z"],
                },
                val_metrics={
                    "abs_rot_x": val_avg["abs_rot_x"],
                    "abs_y": val_avg["abs_y"],
                    "abs_z": val_avg["abs_z"],
                    "abs_trans": val_avg["abs_trans"],
                },
                current_lr_scale=current_lr_scale,
                logger=logger,
            )

            # Fixed-scale, non-learned composite (loss_q already carries the fixed
            # rot_scaling_factor -- see "Unit balancing" in CLAUDE.md) used for
            # checkpoint selection and the global plateau schedule instead of
            # val_avg["loss"]. val_avg["loss"] runs through uncertainty_module when
            # task_weighting is enabled, whose learned log_var terms can drift and
            # inflate it independent of actual prediction quality (confirmed on the
            # q3_global_vs_perHead_LR sweep), which would otherwise pick the wrong
            # "best" checkpoint and mistime the global LR decay.
            val_fixed_loss = val_avg["loss_q"] + val_avg["loss_y"] + val_avg["loss_z"]

            if diag_best is not None:
                if val_avg["abs_rot_x"] < diag_best["rot"][1]:
                    diag_best["rot"] = (epoch, val_avg["abs_rot_x"])
                if val_avg["abs_trans"] < diag_best["trans"][1]:
                    diag_best["trans"] = (epoch, val_avg["abs_trans"])

            global_plateau_manager.update(val_fixed_loss, logger=logger)

            if uncertainty_module is not None and hasattr(uncertainty_module, "update_epoch"):
                try:
                    uncertainty_module.update_epoch(
                        train_avg["loss_q"],
                        train_avg["loss_y"] + train_avg["loss_z"],
                    )
                except Exception as exc:
                    logger.debug(f"Task-weighting epoch update skipped: {exc}")

            # checkpoint_selector.update() is side-effecting (advances its EMA + best state)
            pareto_is_best = (
                checkpoint_selector.update(val_avg["abs_rot_x"], val_avg["abs_trans"])
                if checkpoint_selector is not None
                else False
            )
            is_new_best = pareto_is_best if use_pareto_for_primary else val_fixed_loss < best_val_loss

            if is_new_best:
                best_val_loss = val_fixed_loss
                if diag_best is not None:
                    diag_best["checkpoint"] = (epoch, val_avg["abs_rot_x"], val_avg["abs_trans"])
                save_checkpoint(
                    model, optimizer, epoch, val_fixed_loss, checkpoint_path,
                    uncertainty_module=uncertainty_module,
                    early_stopper=early_stopper,
                    head_lr_manager=head_lr_manager,
                    global_plateau_manager=global_plateau_manager,
                    best_val_loss=best_val_loss,
                    checkpoint_selector=checkpoint_selector,
                )
                _save_raw_gate_weights(
                    checkpoint_path, epoch, val_raw_weights, val_raw_rot_weights, val_raw_trans_weights
                )

            # "both" mode only: write the second, pareto-selected checkpoint alongside
            # the primary combined_loss-selected one, without disturbing the primary's
            # resume/test.py behavior.
            if save_dual_checkpoints and pareto_is_best:
                save_checkpoint(
                    model, optimizer, epoch, val_fixed_loss, secondary_checkpoint_path,
                    uncertainty_module=uncertainty_module,
                    early_stopper=early_stopper,
                    head_lr_manager=head_lr_manager,
                    global_plateau_manager=global_plateau_manager,
                    best_val_loss=val_fixed_loss,
                    checkpoint_selector=checkpoint_selector,
                )
                logger.info(
                    f"checkpoint_best_per_head.pth updated at epoch {epoch} "
                    f"(val_abs_rot_x={val_avg['abs_rot_x']:.6f} deg, val_abs_trans={val_avg['abs_trans']:.6f} m)"
                )

            should_stop = early_stopper.update(val_avg["abs_rot_x"], val_avg["abs_trans"])

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
                lr_fusion=lr_fusion,
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
                    **({"fusion": lr_fusion} if lr_fusion is not None else {}),
                },
                lr_schedule_scale=current_lr_scale,
                param_groups_by_name=param_groups_by_name,
                fusion_weights=fw_str,
                fusion_weights_avg=fusion_weights_avg,
                uncertainty_module=uncertainty_module,
                uncertainty_stats=uncertainty_stats,
                best_val_loss=best_val_loss,
                epoch_time=epoch_time,
                # GAP 2: Raw weights for TensorBoard histograms (Option B)
                train_fusion_weights_raw=train_epoch.get("train_raw_weights"),
                val_fusion_weights_raw=val_raw_weights,
                # GAP 1: Per-task weights for dual_fusion analysis
                train_rot_weights_raw=train_epoch.get("train_raw_rot_weights"),
                val_rot_weights_raw=val_raw_rot_weights,
                train_trans_weights_raw=train_epoch.get("train_raw_trans_weights"),
                val_trans_weights_raw=val_raw_trans_weights,
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
                lr_fusion=lr_fusion,
                current_lr_scale=current_lr_scale,
                uncertainty_stats=uncertainty_stats,
                validation_rotation_summary=val_rotation_summary,
                checkpoint_path=checkpoint_path,
            )
            write_epoch_metrics_csv(csv_writer, row)

            if should_stop:
                logger.info(
                    f"Early stopping at epoch {epoch + 1}: neither val_abs_rot_x nor val_abs_trans "
                    f"improved by >{early_stopper.min_delta * 100:.2f}% for {early_stopper.patience} "
                    f"consecutive epochs (best_rot={early_stopper.best['rot']:.6f} deg, "
                    f"best_trans={early_stopper.best['trans']:.6f} m)."
                )
                break

        if checkpoint_path is not None:
            strip_optimizer_state(checkpoint_path)
            logger.info("Training complete -- stripped optimizer state from checkpoint_best.pth")
        if secondary_checkpoint_path is not None and os.path.exists(secondary_checkpoint_path):
            strip_optimizer_state(secondary_checkpoint_path)
            logger.info("Training complete -- stripped optimizer state from checkpoint_best_per_head.pth")

        if diag_best is not None and checkpoint_path is not None:
            ckpt_epoch, ckpt_rot, ckpt_trans = diag_best["checkpoint"]
            rot_epoch, rot_val = diag_best["rot"]
            trans_epoch, trans_val = diag_best["trans"]
            diagnostic = {
                "checkpoint_selection_metric": "val_fixed_loss (val_loss_q + val_loss_y + val_loss_z)",
                "checkpoint_epoch": ckpt_epoch,
                "checkpoint_val_abs_rot_x": ckpt_rot,
                "checkpoint_val_abs_trans": ckpt_trans,
                "best_val_abs_rot_x_epoch": rot_epoch,
                "best_val_abs_rot_x": rot_val,
                "best_val_abs_trans_epoch": trans_epoch,
                "best_val_abs_trans": trans_val,
                "rot_x_gap_epochs": None if (ckpt_epoch is None or rot_epoch is None) else ckpt_epoch - rot_epoch,
                "trans_gap_epochs": None if (ckpt_epoch is None or trans_epoch is None) else ckpt_epoch - trans_epoch,
            }
            diag_path = os.path.join(os.path.dirname(checkpoint_path), "checkpoint_diagnostic.json")
            with open(diag_path, "w") as f:
                json.dump(diagnostic, f, indent=2)
            logger.info(
                f"checkpoint_diagnostic.json: checkpoint kept from epoch {ckpt_epoch} "
                f"(rot_x={ckpt_rot:.4f}, trans={ckpt_trans:.5f}); best raw val_abs_rot_x was "
                f"epoch {rot_epoch} ({rot_val:.4f}), best raw val_abs_trans was epoch {trans_epoch} "
                f"({trans_val:.5f})"
            )
    finally:
        close_run_writers(csv_file, tb_writer)


def main(
    config_path,
    batch_size=None,
    optimizer_name=None,
    tensorboard_root_dir=None,
    tensorboard_port=6006,
    force_preprocess=False,
    run_directory=None,
    resume_from=None,
    no_tensorboard_server=False,
):
    """Main training entry point.

    resume_from: path to an existing run's output directory (the folder
    containing its checkpoint_best.pth/config.yml/metrics.csv, e.g.
    runs/<run_dir>/<run_name>) to continue training from its last saved
    checkpoint instead of starting fresh at epoch 0. Config is loaded from
    that run's own config.yml snapshot (not config_path) so a resume can't
    silently drift from what the original run actually trained with;
    config_path is ignored when resume_from is given.
    """
    global logger, log_file

    resume_checkpoint_path = None
    if resume_from is not None:
        resume_checkpoint_path = os.path.join(resume_from, "checkpoint_best.pth")
        if not os.path.exists(resume_checkpoint_path):
            raise FileNotFoundError(
                f"--resume_from {resume_from} has no checkpoint_best.pth to resume from"
            )
        config_path = os.path.join(resume_from, "config.yml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"--resume_from {resume_from} has no config.yml snapshot to resume with"
            )

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
    run_setup = RunSetupManager(config_path, cfg, resolved_batch_size, run_directory=run_directory,
                                no_tensorboard_server=no_tensorboard_server)
    artifacts = run_setup.setup_artifacts(existing_run_dir=resume_from)

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

    # Load data (this preprocesses the train split if needed, which is where
    # translation normalization stats are computed from)
    data_pipeline = DataPipelineManager(cfg, base_dir, force_preprocess=force_preprocess, logger=logger)
    train_loader, val_loader, _, trans_norm_stats = data_pipeline.get_loaders(resolved_batch_size)

    # Snapshot the resolved translation normalization stats into cfg before writing
    # config.yml, so every run self-documents the mean/std it actually trained
    # under (computed from this run's own train split -- see
    # compute_translation_norm_stats).
    cfg.setdefault("data", {})["translation_norm_stats"] = dict(
        canonicalize_translation_norm_stats(trans_norm_stats)
    )
    run_setup.save_config_snapshot()

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

    # Compute model FLOPs and parameters. Must run before the resume-checkpoint load
    # below: its profiling hook registers extra total_ops/total_params buffers on
    # every submodule as a side effect, which are then part of model.state_dict() --
    # the checkpoint being resumed was saved (during a *previous* run) after this same
    # call had already run on its model instance, so a freshly-built model needs those
    # buffer slots registered too before a strict state_dict load will match cleanly.
    flops_stats = compute_inference_flops(
        model,
        streams=cfg["model"]["streams"],
        batch_size=resolved_batch_size,
        device=device
    )

    logger.info(
        f"Model: {flops_stats['params_millions']:.2f}M params, "
        f"{flops_stats['gflops_per_sample']:.3f} GFLOPs/sample, "
        f"{flops_stats['gflops_per_batch']:.3f} GFLOPs/batch"
    )

    # Load the resume checkpoint (if any) now, after any --optimizer override and the
    # FLOPs profiling above, so both optimizer_state_dict and model_state_dict line up
    # with whichever optimizer/model instance is final.
    resume_checkpoint = None
    if resume_checkpoint_path is not None:
        resume_checkpoint = load_checkpoint_for_resume(
            resume_checkpoint_path, model, optimizer, device,
            uncertainty_module=uncertainty_module,
        )

    # Prepare training metadata
    head_lr_cfg = cfg.get("training", {}).get("head_lr_decay", {})
    lr_schedule_cfg = cfg.get("training", {}).get("lr_schedule", {})
    early_stopping_cfg = cfg.get("training", {}).get("early_stopping", {})
    checkpoint_selection_cfg = cfg.get("training", {}).get("checkpoint_selection", {})
    task_weighting_cfg = cfg.get("training", {}).get("task_weighting", cfg.get("training", {}).get("uncertainty_weighting", {}))
    grad_clip_cfg = cfg.get("training", {}).get("grad_clip", {})
    grad_clip_max_norm = float(grad_clip_cfg["max_norm"]) if grad_clip_cfg.get("enabled", False) else None
    # per_parameter_group: clip each optimizer param group's own norm to max_norm independently,
    # instead of one norm over the whole model (see _train_single_epoch's clipping site for why).
    # Default False -- identical to every existing run's behavior.
    grad_clip_per_parameter_group = bool(grad_clip_cfg.get("per_parameter_group", False))
    # Diagnostic-only (see train_loop's diag_best comment) -- writes
    # <run_dir>/checkpoint_diagnostic.json comparing the epoch checkpoint_best.pth was
    # actually saved from (by val_fixed_loss) against the epoch that would have been
    # picked by the raw val_abs_rot_x/val_abs_trans metrics alone. Does NOT change which
    # checkpoint is kept. Default False -- identical to every existing run's behavior.
    checkpoint_diagnostic = bool(cfg.get("training", {}).get("checkpoint_diagnostic", False))

    resolved_head_lr_cfg = {
        "enabled": bool(head_lr_cfg.get("enabled", True)),
        "patience": int(head_lr_cfg.get("patience", 3)),
        "factor": float(head_lr_cfg.get("factor", 0.3)),
        "min_delta": float(head_lr_cfg.get("min_delta", 0.0)),
    }
    # 'min_lr' (absolute, aligned with lr_schedule.min_lr) takes precedence; falls back to
    # the legacy 'min_lr_mult' (fraction of each head's base_lr) for existing configs.
    if "min_lr" in head_lr_cfg:
        resolved_head_lr_cfg["min_lr"] = float(head_lr_cfg["min_lr"])
    else:
        resolved_head_lr_cfg["min_lr_mult"] = float(head_lr_cfg.get("min_lr_mult", 0.05))
    
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
        "gflops_per_sample": flops_stats["gflops_per_sample"],
        "gflops_per_batch": flops_stats["gflops_per_batch"],
        "head_lr_decay": resolved_head_lr_cfg,
        "lr_schedule": lr_schedule_cfg if lr_schedule_cfg else {"enabled": False},
        "task_weighting": task_weighting_cfg if uncertainty_module is not None else {"enabled": False},
        "early_stopping": early_stopping_cfg if early_stopping_cfg else {"enabled": False},
        "checkpoint_selection": checkpoint_selection_cfg if checkpoint_selection_cfg else {"type": "combined_loss"},
        "grad_clip": {
            "enabled": grad_clip_max_norm is not None,
            "max_norm": grad_clip_max_norm,
            "per_parameter_group": grad_clip_per_parameter_group,
        },
        "checkpoint_diagnostic": checkpoint_diagnostic,
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
        checkpoint_selection_cfg=checkpoint_selection_cfg,
        rotation_loss_type=rotation_loss_type,
        normalize_translation=cfg.get("data", {}).get("normalize_translation", True),
        trans_norm_stats=trans_norm_stats,
        grad_clip_max_norm=grad_clip_max_norm,
        grad_clip_per_parameter_group=grad_clip_per_parameter_group,
        resume_state=resume_checkpoint,
        rot_scaling_factor=cfg.get("model", {}).get("rot_scaling_factor"),
        checkpoint_diagnostic=checkpoint_diagnostic,
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
        help=(
            "Force rebuild of Data_split and *_preprocessed folders before training "
            "(also recomputes RGB/digit/translation normalization stats from this "
            "run's train split -- see --force_renormalize for an alias of this same "
            "flag named for that specific effect)."
        ),
    )
    parser.add_argument(
        "--force_renormalize",
        action="store_true",
        help=(
            "Alias for --force_preprocess. Use this when your goal is specifically to "
            "recompute normalization stats (e.g. data.translation_norm_stats) from the "
            "train split -- for example after upgrading to real translation "
            "normalization, when the existing Data_split cache still has the old "
            "un-normalized/no-op stats baked in."
        ),
    )
    parser.add_argument(
        "--run_directory",
        type=str,
        default=None,
        help="Root directory for run outputs (default: runs)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Path to an existing run's output directory (containing "
            "checkpoint_best.pth/config.yml/metrics.csv) to resume training from its "
            "last saved checkpoint, instead of starting a fresh run at epoch 0. "
            "Config is loaded from that run's own config.yml snapshot -- --config is "
            "ignored when this is set."
        ),
    )

    parser.add_argument(
        "--no_tensorboard_server",
        action="store_true",
        help="Do not start a background TensorBoard server (event files are still written). "
             "Implied automatically inside SLURM jobs.",
    )

    args = parser.parse_args()

    main(
        args.config,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer,
        tensorboard_root_dir=args.tensorboard_root,
        tensorboard_port=args.tensorboard_port,
        force_preprocess=args.force_preprocess or args.force_renormalize,
        run_directory=args.run_directory,
        resume_from=args.resume_from,
        no_tensorboard_server=args.no_tensorboard_server,
    )
