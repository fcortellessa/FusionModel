import torch
import math


def quaternion_to_2d_unit_vector(quat):
    """
    Convert quaternion (x,y,z,w) to a 2D unit vector (sin, cos) representing rotation around the X-axis (roll).
    Input: quat tensor shape (B,4)
    Output: tensor shape (B,2) with [sin(roll), cos(roll)]
    """
    # Ensure shape
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)

    q = torch.nn.functional.normalize(quat, p=2, dim=1, eps=1e-7)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    return torch.stack([torch.sin(roll), torch.cos(roll)], dim=1)


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

