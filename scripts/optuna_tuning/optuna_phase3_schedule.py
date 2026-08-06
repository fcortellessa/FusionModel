#!/usr/bin/env python3
"""
Phase 3: LR schedule family search. Chooses among no-schedule ("none" -- your existing
fixed-LR baseline algorithm, kept as a control condition since Phase 1/2 may have
already resolved much of the overfitting a schedule would otherwise be compensating
for; not a redundant arm), warmup_cosine, global reduce_on_plateau, and per-head
reduce_on_plateau, then tunes that branch's own sub-parameters.

global reduce_on_plateau and per_head_plateau share the SAME patience/factor/min_lr/
min_delta ranges here -- an earlier version of this script gave per-head a wider
patience range on the theory that its per-head signals are noisier than the global
signal, but that was an untested hypothesis, not something verified against this
project's actual metrics. Letting Optuna search both branches over identical ranges
means if one genuinely needs different patience, that shows up as a real result in
the trial history instead of being assumed upfront.

training.early_stopping and each schedule's min_delta are searched here too (both
were previously fixed/unsearched despite being directly relevant to the overfitting
this whole search exists to address).

lr, weight_decay, dropout_rot, dropout_trans, use_batchnorm, and head architecture are
held fixed at whatever --template-config specifies -- run this against Phase 2's
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
        patience         int(3, 15)
        factor           uniform(0.1, 0.5)
        min_lr           loguniform(1e-6, 1e-4)
        min_delta        loguniform(1e-4, 1e-2)
      per_head_plateau:
        patience         int(3, 15)          (same range as global -- see note above)
        factor           uniform(0.1, 0.5)
        min_lr           loguniform(1e-6, 1e-4)
        min_delta        loguniform(1e-4, 1e-2)
    training.early_stopping:
      enabled            categorical(True, False)
      patience           int(10, 30)          (only if enabled)
      min_delta          loguniform(1e-4, 1e-2)  (only if enabled)

Usage:
    python scripts/optuna_tuning/optuna_phase3_schedule.py \\
        --template-config runs/optuna_phase2_regularization/winner_config.yml \\
        --n-trials 30 \\
        --study-name phase3_schedule \\
        --output-dir runs/optuna_phase3_schedule

Output: runs/optuna_phase3_schedule/winner_config.yml -- the final tuned config,
combining all three phases' winning values. Run a multi-seed confirmation pass on this
config before treating it as a new baseline (every phase here searched at a single
fixed seed, so the winner should be re-validated across seeds like any other ablation
result in this repo).
"""

from __future__ import annotations

from typing import Any, Dict

import optuna

from optuna_common import build_arg_parser, run_study

PLATEAU_PATIENCE_RANGE = (3, 15)
PLATEAU_FACTOR_RANGE = (0.1, 0.5)
PLATEAU_MIN_LR_RANGE = (1e-6, 1e-4)
PLATEAU_MIN_DELTA_RANGE = (1e-4, 1e-2)


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
        overrides["training.lr_schedule.enabled"] = True
        overrides["training.lr_schedule.type"] = "reduce_on_plateau"
        overrides["training.lr_schedule.patience"] = trial.suggest_int("gp_patience", *PLATEAU_PATIENCE_RANGE)
        overrides["training.lr_schedule.factor"] = trial.suggest_float("gp_factor", *PLATEAU_FACTOR_RANGE)
        overrides["training.lr_schedule.min_lr"] = trial.suggest_float("gp_min_lr", *PLATEAU_MIN_LR_RANGE, log=True)
        overrides["training.lr_schedule.min_delta"] = trial.suggest_float("gp_min_delta", *PLATEAU_MIN_DELTA_RANGE, log=True)
        overrides["training.head_lr_decay.enabled"] = False

    elif schedule_choice == "per_head_plateau":
        overrides["training.lr_schedule.enabled"] = False
        overrides["training.head_lr_decay.enabled"] = True
        overrides["training.head_lr_decay.patience"] = trial.suggest_int("hd_patience", *PLATEAU_PATIENCE_RANGE)
        overrides["training.head_lr_decay.factor"] = trial.suggest_float("hd_factor", *PLATEAU_FACTOR_RANGE)
        overrides["training.head_lr_decay.min_lr"] = trial.suggest_float("hd_min_lr", *PLATEAU_MIN_LR_RANGE, log=True)
        overrides["training.head_lr_decay.min_delta"] = trial.suggest_float("hd_min_delta", *PLATEAU_MIN_DELTA_RANGE, log=True)

    early_stopping_enabled = trial.suggest_categorical("early_stopping_enabled", [True, False])
    overrides["training.early_stopping.enabled"] = early_stopping_enabled
    if early_stopping_enabled:
        overrides["training.early_stopping.patience"] = trial.suggest_int("es_patience", 10, 30)
        overrides["training.early_stopping.min_delta"] = trial.suggest_float("es_min_delta", 1e-4, 1e-2, log=True)

    return overrides


def main():
    parser = build_arg_parser("Phase 3: LR schedule family search")
    args = parser.parse_args()
    run_study(args, suggest_params)


if __name__ == "__main__":
    main()
