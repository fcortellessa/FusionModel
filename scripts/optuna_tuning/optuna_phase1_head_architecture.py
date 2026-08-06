#!/usr/bin/env python3
"""
Phase 1: prediction-head architecture search (mlp_quat_1/2/3, mlp_trans_1/2/3 widths,
depth fixed at 3 layers per prior decision). Runs FIRST, ahead of regularization/LR
(Phase 2) and schedule (Phase 3), because head width more strongly influences the
optimal LR/dropout than the reverse (a wider head plausibly wants a smaller LR and
more dropout to avoid overfitting) -- tuning LR/dropout first would risk anchoring to
a network capacity Phase 1 later discards.

translation_head_mode is NOT searched here -- fix it to "joint" in --template-config
(per current research direction) rather than letting Optuna pick it; joint vs. separate
is being evaluated separately via the round-1/round-2 comparison grid, not folded into
this HPO search. If you later want a tuned architecture for separate mode too, rerun
this phase against a separate-mode template as its own study.

Everything else -- lr, regularization, schedule -- is held fixed at whatever
--template-config specifies (should be a "pure_fixed", no-schedule config so this
phase isn't confounded by a schedule already compensating for architecture choice).

This is the highest-dimensional phase (4x4x3x4x4x4 = 3072 width combinations) --
budget more trials here than Phase 2/3, or expect a noisier result.

Search space:
    model.mlp_quat_1   categorical(256, 512, 1024, 2048)
    model.mlp_quat_2   categorical(256, 512, 1024, 2048)
    model.mlp_quat_3   categorical(128, 256, 512)
    model.mlp_trans_1  categorical(64, 128, 256, 512)
    model.mlp_trans_2  categorical(32, 64, 128, 256)
    model.mlp_trans_3  categorical(16, 32, 64, 128)

Usage:
    python scripts/optuna_tuning/optuna_phase1_head_architecture.py \\
        --template-config configs/generated/lr_schedule_head_mode_round2/001_pure_fixed_joint_seed43.yml \\
        --n-trials 40 \\
        --study-name phase1_head_architecture \\
        --output-dir runs/optuna_phase1_head_architecture

Output: runs/optuna_phase1_head_architecture/winner_config.yml -- feed this as
--template-config to optuna_phase2_regularization.py.
"""

from __future__ import annotations

from typing import Any, Dict

import optuna

from optuna_common import build_arg_parser, run_study


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    overrides["model.mlp_quat_1"] = trial.suggest_categorical("mlp_quat_1", [256, 512, 1024, 2048])
    overrides["model.mlp_quat_2"] = trial.suggest_categorical("mlp_quat_2", [256, 512, 1024, 2048])
    overrides["model.mlp_quat_3"] = trial.suggest_categorical("mlp_quat_3", [128, 256, 512])
    overrides["model.mlp_trans_1"] = trial.suggest_categorical("mlp_trans_1", [64, 128, 256, 512])
    overrides["model.mlp_trans_2"] = trial.suggest_categorical("mlp_trans_2", [32, 64, 128, 256])
    overrides["model.mlp_trans_3"] = trial.suggest_categorical("mlp_trans_3", [16, 32, 64, 128])

    return overrides


def main():
    parser = build_arg_parser("Phase 1: prediction-head architecture search")
    args = parser.parse_args()
    run_study(args, suggest_params)


if __name__ == "__main__":
    main()
