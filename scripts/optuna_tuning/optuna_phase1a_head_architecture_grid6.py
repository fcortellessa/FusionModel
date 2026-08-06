#!/usr/bin/env python3
"""
Phase 1A: prediction-head architecture search at spatial_grid_size=6 (reduced bottleneck).

Searches mlp_quat_1/2/3 and mlp_trans_1/2/3 widths with spatial_grid_size FIXED at 6.
This produces a ~16,128-dim fused vector, reducing the bottleneck from 44× to 16×
compared to the grid=10 baseline.

Runs FIRST (ahead of Phase 1B grid=10, Phase 2 regularization, Phase 3 schedule),
because head width more strongly influences optimal LR/dropout than the reverse.

translation_head_mode is NOT searched here -- fix it to "joint" in --template-config
(per current research direction) rather than letting Optuna pick it; joint vs. separate
is being evaluated separately via the round-1/round-2 comparison grid, not folded into
this HPO search.

Everything else -- lr, regularization, schedule -- is held fixed at whatever
--template-config specifies (should be a "pure_fixed", no-schedule config so this
phase isn't confounded by a schedule already compensating for architecture choice).

Search space (144 combinations):
    model.spatial_grid_size  fixed at 6
    model.mlp_quat_1         categorical(1024, 2048, 4096)
    model.mlp_quat_2         categorical(1024, 2048)
    model.mlp_quat_3         categorical(512, 1024)
    model.mlp_trans_1        categorical(256, 512, 1024)
    model.mlp_trans_2        categorical(128, 256)
    model.mlp_trans_3        categorical(64, 128)

Usage:
    python scripts/optuna_tuning/optuna_phase1a_head_architecture_grid6.py \\
        --template-config configs/generated/lr_schedule_head_mode_round2/001_pure_fixed_joint_seed43.yml \\
        --n-trials 20 \\
        --study-name phase1a_head_architecture_grid6 \\
        --output-dir runs/optuna_phase1a_head_architecture_grid6

Output: runs/optuna_phase1a_head_architecture_grid6/winner_config.yml -- compare against
Phase 1B (grid=10) winner to decide which grid size is better.
"""

from __future__ import annotations

from typing import Any, Dict

import optuna

from optuna_common import build_arg_parser, run_study


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    # spatial_grid_size is FIXED at 6 for this phase; not searched.
    # (Phase 1B searches it at 10 for direct comparison)
    overrides["model.spatial_grid_size"] = 6

    overrides["model.mlp_quat_1"] = trial.suggest_categorical("mlp_quat_1", [1024, 2048, 4096])
    overrides["model.mlp_quat_2"] = trial.suggest_categorical("mlp_quat_2", [1024, 2048])
    overrides["model.mlp_quat_3"] = trial.suggest_categorical("mlp_quat_3", [512, 1024])
    overrides["model.mlp_trans_1"] = trial.suggest_categorical("mlp_trans_1", [256, 512, 1024])
    overrides["model.mlp_trans_2"] = trial.suggest_categorical("mlp_trans_2", [128, 256])
    overrides["model.mlp_trans_3"] = trial.suggest_categorical("mlp_trans_3", [64, 128])

    return overrides


def main():
    parser = build_arg_parser("Phase 1A: prediction-head architecture search (spatial_grid_size=6)")
    args = parser.parse_args()
    run_study(args, suggest_params)


if __name__ == "__main__":
    main()
