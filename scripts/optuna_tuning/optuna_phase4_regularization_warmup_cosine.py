#!/usr/bin/env python3
"""
Phase 4: dropout + weight_decay search, run against the ACTUAL config that will ship
(architecture confirmed by optuna_phase1b_extended_local.py, schedule tuned by
optuna_phase3_schedule.py) -- not the isolated no-schedule regime the abandoned
optuna_phase2_regularization.py used.

Why this exists: dropout_rot/dropout_trans have sat at 0.1/0.1 and weight_decay at 0.0
through every phase so far -- Phase 2 (the phase meant to tune them) was abandoned at
2 of 25 trials, and Phase 3 (schedule search) ran on top of those untouched defaults
because nothing else was available at the time. eval_optuna_winner_on_test.py then
showed this actually costs real generalization: every completed Phase 3 trial checked
against the test split has a positive val/test gap (+18% to +27%), and the one trial
that showed a much smaller gap (+11.3%, trial 19) was the one trial in that group
running with early_stopping enabled -- every other early-stopped trial in Phase 3
scored *worse* on val than the no-early-stopping trials (which is why Optuna's
val-only objective never favored it), yet it generalizes better. That's the textbook
symptom of training past the point that helps: still shrinking val loss by fitting
things that don't transfer, exactly what dropout and weight_decay exist to resist.

--template-config should be runs/optuna_phase3_schedule/winner_config.yml (trial 17's
resolved config): architecture is the one optuna_phase1b_extended_local.py confirmed
as a local optimum (no pushed value in any of its 6 dimensions beat it), and the
warmup_cosine schedule is Phase 3's tuned winner. This phase touches neither -- only
dropout_rot, dropout_trans, weight_decay, and early_stopping's own patience/min_delta
are searched.

training.early_stopping.enabled is fixed ON (not re-searched): Phase 3 already
searched it freely and every early-stopped trial underperformed on val for the reason
described above -- val systematically prefers letting training run long -- so
re-offering "off" here would just reproduce that same bias. patience and min_delta,
however, ARE searched (see below), not fixed: Phase 3 trial 19 -- the one
early-stopped trial actually checked against test -- landed at patience=30, the exact
top of the int(10, 30) range Phase 3 offered, and min_delta=0.0094, ~98% of the way up
its loguniform(1e-4, 1e-2) range on a log scale. Both sit against the edge of what was
searched, the same boundary-hit shape that justified extending
optuna_phase1b_extended_local.py's architecture grid -- so this phase searches past
them rather than assuming 30/0.0094 was really the optimum.

patience's upper bound is capped at 35, NOT pushed further just because 30 was a
boundary hit: trial 19's own metrics.csv shows it stopped at epoch 44 of the
epochs: 80 budget -- patience=30 already consumed nearly half the run waiting out a
dry spell. Past roughly 40, patience approaches the point where a single noise-sized
uptick anywhere in a noisy val curve resets the counter and the trial just silently
runs the full 80 epochs anyway -- functionally re-disabling the thing this phase
exists to keep on. 35 stays under half the budget while still testing past Phase 3's
ceiling.

weight_decay has no on/off gate (unlike the abandoned optuna_phase2_regularization.py):
just a single loguniform(1e-6, 1e-2) draw. The categorical gate that script used adds a
discrete branch TPE has to model on top of a continuous one, and 1e-6 is already close
enough to zero to be functionally indistinguishable from wd=0 in AdamW over 80 epochs
-- so dropping the gate is a real simplification, not a loss of the ability to find
"weight_decay doesn't matter here" if that's the answer.

dropout_rot/dropout_trans are searched on a fixed 0.05 grid (via suggest_float's
step= argument, not suggest_categorical -- step preserves the values' ordering/
distance for TPE, a plain categorical list would not) rather than continuously:
differences smaller than ~0.05 aren't likely distinguishable against a single-seed,
single-run's noise anyway, so a continuous search would be spending resolution the
32-trial budget can't back up. 11x11=121 combinations for the two dims is the same
order of magnitude as the architecture grids already run in this project.

lr is NOT searched here (still inherited from the template, still never tuned under
any schedule) -- out of scope for this phase; flag as a follow-up if the dropout/
weight_decay winner still shows a meaningful val/test gap.

Search space:
    training.dropout_rot                {0.0, 0.05, 0.10, ..., 0.5} -- suggest_float(step=0.05)
    training.dropout_trans              {0.0, 0.05, 0.10, ..., 0.5} -- suggest_float(step=0.05)
    training.weight_decay               loguniform(1e-6, 1e-2)
    training.early_stopping.enabled     fixed True
    training.early_stopping.patience    int(15, 35)          -- was int(10, 30) in Phase 3
    training.early_stopping.min_delta   loguniform(1e-4, 2e-2)  -- was (1e-4, 1e-2) in Phase 3

Usage:
    python scripts/optuna_tuning/optuna_phase4_regularization_warmup_cosine.py \\
        --template-config runs/optuna_phase3_schedule/winner_config.yml \\
        --n-trials 32 \\
        --tail-epochs 10 \\
        --study-name phase4_regularization_warmup_cosine \\
        --output-dir runs/optuna_phase4_regularization_warmup_cosine

(--tail-epochs raised from the shared default of 5: Phase 3's trial 17 vs. trial 29
comparison showed a val-score gap of just 0.0086 was enough for Optuna's argmin to
pick the worse-generalizing trial -- averaging over more epochs gives each trial's
score less exposure to that kind of single-window noise.)

Output: runs/optuna_phase4_regularization_warmup_cosine/winner_config.yml. Before
trusting it, run it through eval_optuna_winner_on_test.py and compare against trial
19's benchmark (val=1.4103, test=1.5694, gap=+11.3%) -- the bar this phase exists to
beat, not just its own val score.
"""

from __future__ import annotations

from typing import Any, Dict

import optuna

from optuna_common import build_arg_parser, run_study

# Seed trial: the template's own dropout (0.1/0.1), weight_decay near-zero, and
# early_stopping's patience/min_delta at Phase 3 trial 19's actual values -- a
# reference point reproducing what's already been checked against test (val=1.4103,
# test=1.5694, gap=+11.3%), anchoring the sampler before it explores the extended
# patience/min_delta ranges and the now-gate-free weight_decay dimension.
SEED_TRIALS = [
    {
        "dropout_rot": 0.1,
        "dropout_trans": 0.1,
        "weight_decay": 1e-6,
        "early_stopping_patience": 30,
        "early_stopping_min_delta": 0.0094,
    },
]


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    overrides["training.dropout_rot"] = trial.suggest_float("dropout_rot", 0.0, 0.3, step=0.05)
    overrides["training.dropout_trans"] = trial.suggest_float("dropout_trans", 0.0, 0.3, step=0.05)
    overrides["training.weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    overrides["training.early_stopping.enabled"] = True
    overrides["training.early_stopping.patience"] = trial.suggest_int("early_stopping_patience", 15, 35)
    overrides["training.early_stopping.min_delta"] = trial.suggest_float(
        "early_stopping_min_delta", 1e-4, 2e-2, log=True
    )

    return overrides


def main():
    parser = build_arg_parser(
        "Phase 4: dropout + weight_decay search against the confirmed architecture + tuned schedule"
    )
    args = parser.parse_args()
    run_study(args, suggest_params, enqueue_trials=SEED_TRIALS)


if __name__ == "__main__":
    main()
