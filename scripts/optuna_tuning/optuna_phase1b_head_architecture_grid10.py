#!/usr/bin/env python3
"""
Phase 1B: prediction-head architecture search at spatial_grid_size=10 (current baseline).

Searches mlp_quat_1/2/3 and mlp_trans_1/2/3 widths with spatial_grid_size FIXED at 10.
This produces a ~44,800-dim fused vector, the current baseline bottleneck (44× to 1024
for rotation, 175× to 256 for translation).

Runs alongside Phase 1A (grid=6) so you can directly compare which grid size performs
better. Both studies use identical search ranges for the head widths, so you'll see if
grid=6's reduced bottleneck is actually better, or if grid=10 still wins despite the
severe compression.

Searches head widths with spatial_grid_size FIXED at 10, because head width more
strongly influences optimal LR/dropout than the reverse.

translation_head_mode is NOT searched here -- fix it to "joint" in --template-config
(per current research direction) rather than letting Optuna pick it; joint vs. separate
is being evaluated separately via the round-1/round-2 comparison grid, not folded into
this HPO search.

Everything else -- lr, regularization, schedule -- is held fixed at whatever
--template-config specifies (should be a "pure_fixed", no-schedule config so this
phase isn't confounded by a schedule already compensating for architecture choice).

Search space (144 combinations, identical to Phase 1A for fair comparison):
    model.spatial_grid_size  fixed at 10
    model.mlp_quat_1         categorical(1024, 2048, 4096)
    model.mlp_quat_2         categorical(1024, 2048)
    model.mlp_quat_3         categorical(512, 1024)
    model.mlp_trans_1        categorical(256, 512, 1024)
    model.mlp_trans_2        categorical(128, 256)
    model.mlp_trans_3        categorical(64, 128)

Usage:
    python scripts/optuna_tuning/optuna_phase1b_head_architecture_grid10.py \\
        --template-config configs/generated/lr_schedule_head_mode_round2/001_pure_fixed_joint_seed43.yml \\
        --n-trials 20 \\
        --study-name phase1b_head_architecture_grid10 \\
        --output-dir runs/optuna_phase1b_head_architecture_grid10

Output: runs/optuna_phase1b_head_architecture_grid10/winner_config.yml -- compare against
Phase 1A (grid=6) winner to decide which grid size is better. Then use the winning
config as --template-config for Phase 2 (regularization search).
"""

from __future__ import annotations

from typing import Any, Dict

import optuna

from optuna_common import build_arg_parser, run_study


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    # spatial_grid_size is FIXED at 10 (current baseline) for this phase; not searched.
    # (Phase 1A searches it at 6 for direct comparison)
    overrides["model.spatial_grid_size"] = 10

    overrides["model.mlp_quat_1"] = trial.suggest_categorical("mlp_quat_1", [1024, 2048, 4096])
    overrides["model.mlp_quat_2"] = trial.suggest_categorical("mlp_quat_2", [1024, 2048])
    overrides["model.mlp_quat_3"] = trial.suggest_categorical("mlp_quat_3", [512, 1024])
    overrides["model.mlp_trans_1"] = trial.suggest_categorical("mlp_trans_1", [256, 512, 1024])
    overrides["model.mlp_trans_2"] = trial.suggest_categorical("mlp_trans_2", [128, 256])
    overrides["model.mlp_trans_3"] = trial.suggest_categorical("mlp_trans_3", [64, 128])

    return overrides


def main():
    parser = build_arg_parser("Phase 1B: prediction-head architecture search (spatial_grid_size=10)")
    args = parser.parse_args()
    run_study(args, suggest_params)


if __name__ == "__main__":
    main()
