#!/usr/bin/env python3
"""
Phase 2: LR + regularization search (lr, weight_decay, dropout_rot, dropout_trans).
Runs after Phase 1A and/or 1B (head architecture at grid=6 and/or grid=10) so these
are tuned against the network capacity that phase already settled on, not a capacity
Phase 1 would later discard.

Once you've run Phase 1A (grid=6) and Phase 1B (grid=10) and chosen the better one,
use its winner_config.yml as --template-config for this phase.

use_batchnorm is intentionally NOT searched: fixed to False in every trial here
(BatchNorm's running statistics interact poorly with AdamW's per-parameter adaptive
scaling -- not searched, just disabled). --template-config should already have
training.use_batchnorm: false and training.optimizer: AdamW (the only optimizer in use);
this script does not override either.

Schedule/head-decay is held fixed at whatever --template-config specifies -- run this
phase against a "pure_fixed" (no lr_schedule, no head_lr_decay) template so the
regularization effect isn't confounded by a schedule already fighting overfitting on
its own. --template-config should be Phase 1's (1A or 1B) winner_config.yml.

Search space:
    training.lr               loguniform(1e-5, 1e-3)
    training.weight_decay     0.0 (50% of trials) or loguniform(1e-6, 1e-2)
    training.dropout_rot      uniform(0.0, 0.5)
    training.dropout_trans    uniform(0.0, 0.5)

Usage:
    python scripts/optuna_tuning/optuna_phase2_regularization.py \\
        --template-config runs/optuna_phase1a_head_architecture_grid6/winner_config.yml \\
        --n-trials 25 \\
        --study-name phase2_regularization_from_phase1a \\
        --output-dir runs/optuna_phase2_regularization_from_phase1a

Output: runs/optuna_phase2_regularization/winner_config.yml -- feed this as
--template-config to optuna_phase3_schedule.py.
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

    return overrides


def main():
    parser = build_arg_parser("Phase 2: LR + regularization search")
    args = parser.parse_args()
    run_study(args, suggest_params)


if __name__ == "__main__":
    main()
