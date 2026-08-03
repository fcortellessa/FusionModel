#!/usr/bin/env python3
"""
Phase 2: LR schedule family search. Chooses among no-schedule, warmup_cosine,
global reduce_on_plateau, and per-head reduce_on_plateau, then tunes that branch's
own sub-parameters -- deliberately NOT shared ranges across branches, since they
operate on structurally different signals:

  - warmup_cosine has no patience/factor concept at all (time-based, not loss-based).
  - reduce_on_plateau (global) watches ONE shared val_loss signal -- smoother/less
    noisy, so it can plausibly tolerate lower patience before decaying.
  - per_head_plateau runs two independent schedulers against per-head signals that
    are noisier and can drift apart -- given a wider patience range so it doesn't
    decay prematurely on noise.

training.early_stopping and each schedule's min_delta are searched here too (both
were previously fixed/unsearched despite being directly relevant to the overfitting
this whole search exists to address).

lr, weight_decay, dropout_rot, dropout_trans, use_batchnorm, and head architecture are
held fixed at whatever --template-config specifies -- run this against Phase 1's
winner_config.yml.

Search space:
    training.lr_schedule / training.head_lr_decay -- one of:
      none:
        (nothing else; head_lr_decay and lr_schedule both disabled)
      warmup_cosine:
        pct_start        uniform(0.05, 0.3)
        start_lr_mult    uniform(0.05, 0.5)
        final_lr_mult    loguniform(0.001, 0.1)
      reduce_on_plateau (global):
        patience         int(3, 8)
        factor           uniform(0.1, 0.5)
        min_lr           loguniform(1e-6, 1e-4)
        min_delta        loguniform(1e-4, 1e-2)
      per_head_plateau:
        patience         int(5, 15)
        factor           uniform(0.1, 0.5)
        min_lr           loguniform(1e-6, 1e-4)
        min_delta        loguniform(1e-4, 1e-2)
    training.early_stopping:
      enabled            categorical(True, False)
      patience           int(10, 30)          (only if enabled)
      min_delta          loguniform(1e-4, 1e-2)  (only if enabled)

Usage:
    python scripts/tactile_ablation/optuna_phase2_schedule.py \\
        --template-config runs/optuna_phase1_regularization/winner_config.yml \\
        --n-trials 30 \\
        --study-name phase2_schedule \\
        --output-dir runs/optuna_phase2_schedule

Output: runs/optuna_phase2_schedule/winner_config.yml -- feed this as
--template-config to optuna_phase3_head_architecture.py.
"""

from __future__ import annotations

from typing import Any, Dict

import optuna

from optuna_common import build_arg_parser, run_study


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    schedule_choice = trial.suggest_categorical(
        "schedule_choice", ["none", "warmup_cosine", "reduce_on_plateau", "per_head_plateau"]
    )

    if schedule_choice == "none":
        overrides["training.lr_schedule.enabled"] = False
        overrides["training.head_lr_decay.enabled"] = False

    elif schedule_choice == "warmup_cosine":
        overrides["training.lr_schedule.enabled"] = True
        overrides["training.lr_schedule.type"] = "warmup_cosine"
        overrides["training.lr_schedule.pct_start"] = trial.suggest_float("wc_pct_start", 0.05, 0.3)
        overrides["training.lr_schedule.start_lr_mult"] = trial.suggest_float("wc_start_lr_mult", 0.05, 0.5)
        overrides["training.lr_schedule.final_lr_mult"] = trial.suggest_float("wc_final_lr_mult", 0.001, 0.1, log=True)
        overrides["training.head_lr_decay.enabled"] = False

    elif schedule_choice == "reduce_on_plateau":
        # Global plateau watches one shared, smoother val_loss signal -- can tolerate
        # lower patience than the per-head variant before decaying.
        overrides["training.lr_schedule.enabled"] = True
        overrides["training.lr_schedule.type"] = "reduce_on_plateau"
        overrides["training.lr_schedule.patience"] = trial.suggest_int("gp_patience", 3, 8)
        overrides["training.lr_schedule.factor"] = trial.suggest_float("gp_factor", 0.1, 0.5)
        overrides["training.lr_schedule.min_lr"] = trial.suggest_float("gp_min_lr", 1e-6, 1e-4, log=True)
        overrides["training.lr_schedule.min_delta"] = trial.suggest_float("gp_min_delta", 1e-4, 1e-2, log=True)
        overrides["training.head_lr_decay.enabled"] = False

    elif schedule_choice == "per_head_plateau":
        # Two independent schedulers on noisier per-head signals -- wider patience
        # range so a single head doesn't decay prematurely on noise.
        overrides["training.lr_schedule.enabled"] = False
        overrides["training.head_lr_decay.enabled"] = True
        overrides["training.head_lr_decay.patience"] = trial.suggest_int("hd_patience", 5, 15)
        overrides["training.head_lr_decay.factor"] = trial.suggest_float("hd_factor", 0.1, 0.5)
        overrides["training.head_lr_decay.min_lr"] = trial.suggest_float("hd_min_lr", 1e-6, 1e-4, log=True)
        overrides["training.head_lr_decay.min_delta"] = trial.suggest_float("hd_min_delta", 1e-4, 1e-2, log=True)

    early_stopping_enabled = trial.suggest_categorical("early_stopping_enabled", [True, False])
    overrides["training.early_stopping.enabled"] = early_stopping_enabled
    if early_stopping_enabled:
        overrides["training.early_stopping.patience"] = trial.suggest_int("es_patience", 10, 30)
        overrides["training.early_stopping.min_delta"] = trial.suggest_float("es_min_delta", 1e-4, 1e-2, log=True)

    return overrides


def main():
    parser = build_arg_parser("Phase 2: LR schedule family search")
    args = parser.parse_args()
    run_study(args, suggest_params)


if __name__ == "__main__":
    main()
