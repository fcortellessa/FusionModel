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
dropout_rot, dropout_trans, and weight_decay are searched.

training.early_stopping is fixed ON (not re-searched): patience=30, min_delta=0.01,
approximating Phase 3 trial 19's values (the one early-stopped trial actually checked
against test). Phase 3 already searched early_stopping.enabled freely and every
early-stopped trial underperformed on val for the same reason described above -- val
systematically prefers letting training run long, so re-offering "off" as an option
here would just reproduce that same bias. Fixing it on spends this phase's trial
budget on dropout/weight_decay instead, which is what's actually unexplored.

lr is NOT searched here (still inherited from the template, still never tuned under
any schedule) -- out of scope for this phase; flag as a follow-up if the dropout/
weight_decay winner still shows a meaningful val/test gap.

Search space:
    training.dropout_rot      uniform(0.0, 0.5)
    training.dropout_trans    uniform(0.0, 0.5)
    training.weight_decay     0.0 (50% of trials) or loguniform(1e-6, 1e-2)
                               -- same 50/50 gate optuna_phase2_regularization.py used,
                               kept deliberately: we have evidence *some* regularization
                               helps, not evidence weight_decay specifically (vs.
                               dropout alone) does, so this still lets Optuna find that
                               wd=0 is fine if dropout alone covers it.
    training.early_stopping.enabled    fixed True
    training.early_stopping.patience   fixed 30
    training.early_stopping.min_delta  fixed 0.01

Usage:
    python scripts/optuna_tuning/optuna_phase4_regularization_warmup_cosine.py \\
        --template-config runs/optuna_phase3_schedule/winner_config.yml \\
        --n-trials 25 \\
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

# Fixed early-stopping settings, approximating Phase 3 trial 19 -- the one
# early-stopped trial actually confirmed (via eval_optuna_winner_on_test.py) to
# generalize better despite a worse val score. See module docstring.
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_MIN_DELTA = 0.01

# Seed trial: the template's own dropout/weight_decay (0.1/0.1/wd=0) but with
# early_stopping now forced on -- a clean "early stopping alone, nothing else changed"
# reference point for this study, comparable to trial 19 without also inheriting
# trial 19's own (searched, not fixed) early-stopping patience/min_delta or its
# slightly different warmup_cosine pct_start.
SEED_TRIALS = [
    {"dropout_rot": 0.1, "dropout_trans": 0.1, "use_weight_decay": False},
]


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    overrides["training.dropout_rot"] = trial.suggest_float("dropout_rot", 0.0, 0.5)
    overrides["training.dropout_trans"] = trial.suggest_float("dropout_trans", 0.0, 0.5)

    use_weight_decay = trial.suggest_categorical("use_weight_decay", [True, False])
    overrides["training.weight_decay"] = (
        trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) if use_weight_decay else 0.0
    )

    overrides["training.early_stopping.enabled"] = True
    overrides["training.early_stopping.patience"] = EARLY_STOPPING_PATIENCE
    overrides["training.early_stopping.min_delta"] = EARLY_STOPPING_MIN_DELTA

    return overrides


def main():
    parser = build_arg_parser(
        "Phase 4: dropout + weight_decay search against the confirmed architecture + tuned schedule"
    )
    args = parser.parse_args()
    run_study(args, suggest_params, enqueue_trials=SEED_TRIALS)


if __name__ == "__main__":
    main()
