#!/usr/bin/env python3
"""
Phase 1: LR + regularization search (lr, weight_decay, dropout_rot, dropout_trans,
use_batchnorm). Schedule/head-decay and head architecture are held fixed at whatever
--template-config specifies -- run this phase against a "pure_fixed" (no lr_schedule,
no head_lr_decay) template so the regularization effect isn't confounded by a schedule
already fighting overfitting on its own.

Search space:
    training.lr               loguniform(1e-5, 1e-3)
    training.weight_decay     0.0 (50% of trials) or loguniform(1e-6, 1e-2)
    training.dropout_rot      uniform(0.0, 0.5)
    training.dropout_trans    uniform(0.0, 0.5)
    training.use_batchnorm    categorical(True, False)

Usage:
    python scripts/tactile_ablation/optuna_phase1_regularization.py \\
        --template-config configs/generated/lr_schedule_head_mode_round2/001_pure_fixed_joint_seed43.yml \\
        --n-trials 25 \\
        --study-name phase1_regularization \\
        --output-dir runs/optuna_phase1_regularization

Output: runs/optuna_phase1_regularization/winner_config.yml -- feed this as
--template-config to optuna_phase2_schedule.py.
"""

from __future__ import annotations

from typing import Any, Dict

import optuna

from optuna_common import build_arg_parser, run_study


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    overrides["training.lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    use_weight_decay = trial.suggest_categorical("use_weight_decay", [True, False])
    overrides["training.weight_decay"] = (
        trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) if use_weight_decay else 0.0
    )

    overrides["training.dropout_rot"] = trial.suggest_float("dropout_rot", 0.0, 0.5)
    overrides["training.dropout_trans"] = trial.suggest_float("dropout_trans", 0.0, 0.5)
    overrides["training.use_batchnorm"] = trial.suggest_categorical("use_batchnorm", [True, False])

    return overrides


def main():
    parser = build_arg_parser("Phase 1: LR + regularization search")
    args = parser.parse_args()
    run_study(args, suggest_params)


if __name__ == "__main__":
    main()
