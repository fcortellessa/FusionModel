#!/usr/bin/env python3
"""
Phase 1B follow-up: exploitative local search around optuna_phase1b_head_architecture_grid10's
regrouped-split winner (trial 19: mlp_quat_1=4096, mlp_quat_2=2048, mlp_quat_3=512,
mlp_trans_1=1024, mlp_trans_2=256, mlp_trans_3=64, score=1.4444).

Why this exists: that winner sat on a grid boundary in all six searched dimensions --
four at the max value offered, two at the min -- which is the standard signal that the
grid was too narrow, not that the search converged on an interior optimum. On top of
that, the original 20-trial study (TPESampler, no GridSampler) only ever evaluated 11
distinct architectures out of the nominal 144-combination space, and the winning point
has zero replication: the only other trials with mlp_quat_1=4096 paired it with
mlp_quat_3=1024 and scored 2.4-6.9 (two pruned outright), 2-5x worse than the winner's
mlp_quat_3=512 pairing. One sample at the edge of a strong interaction is not enough to
trust as-is.

This is deliberately NOT a repeat of the original broad search. It searches a small,
local neighborhood -- each dimension gets exactly two values: the incumbent winner's
value, and one step further in the direction that was winning (extended past the old
max for the four boundary-max dims, extended past the old min for the two boundary-min
dims). That's 2**6 = 64 combinations, all within one step of a known-good point, instead
of re-covering the full original range.

Sampling is configured to exploit rather than explore that neighborhood:
  - TPESampler(multivariate=True, group=True): models the six params JOINTLY instead of
    independently. Plain per-parameter TPE is exactly what let the original study treat
    mlp_quat_1=4096 as promising in the abstract while missing that it only works
    together with a small mlp_quat_3 -- multivariate TPE can pick up on that interaction
    and stop wasting trials on combinations like quat_1=8192 + quat_3=512-only-by-luck.
  - n_startup_trials=4, matching the 4 enqueued seed points below, so the sampler moves
    to its (multivariate) model immediately after those instead of spending additional
    trials on random startup exploration across the local grid.
  - The incumbent and three seed points are force-enqueued (run_study's enqueue_trials)
    so the study's model is anchored at the known-good region from trial 0:
      1. the incumbent exactly, as a fresh reference row in this study
      2. every dimension pushed one step further (the "does scaling further help
         uniformly" corner)
      3. mlp_quat_1 pushed alone (isolates that dimension's effect, given the
         quat_1 x quat_3 interaction already observed)
      4. mlp_quat_3 pushed alone (same, other side of that interaction)
    Training is fully deterministic (config.deterministic: true), so re-running the
    incumbent's exact combo elsewhere would just reproduce trial 19's score -- these
    seeds exist to anchor the sampler's model, not to generate new information on their
    own beyond point 1's cross-study sanity check.

Search space (64 combinations):
    model.mlp_quat_1   categorical(4096, 8192)   -- was max=4096, extending up
    model.mlp_quat_2   categorical(2048, 4096)   -- was max=2048, extending up
    model.mlp_quat_3   categorical(256, 512)     -- was min=512, extending down
    model.mlp_trans_1  categorical(1024, 2048)   -- was max=1024, extending up
    model.mlp_trans_2  categorical(256, 512)     -- was max=256, extending up
    model.mlp_trans_3  categorical(32, 64)       -- was min=64, extending down

Everything else -- lr, regularization, schedule -- stays fixed at whatever
--template-config specifies, same convention as optuna_phase1b_head_architecture_grid10.py.
Use the same "pure_fixed" template that study used, so this is a clean architecture-only
extension of it.

Usage:
    python scripts/optuna_tuning/optuna_phase1b_extended_local.py \\
        --template-config configs/generated/lr_schedule_head_mode_round2/001_pure_fixed_joint_seed43.yml \\
        --n-trials 24 \\
        --study-name phase1b_extended_local \\
        --output-dir runs/optuna_phase1b_extended_local

Output: runs/optuna_phase1b_extended_local/winner_config.yml. Compare its score against
1.4444 (runs/optuna_phase1b_head_architecture_grid10_regrouped/winner_config.yml):
  - if the incumbent (seed #1) still wins, the boundary hits were noise/interaction
    artifacts, not an under-scoped grid -- keep the original winner and move on.
  - if a pushed value wins, feed THIS script's winner_config.yml forward as
    --template-config to Phase 2 / the lr-dropout-under-warmup_cosine sweep instead of
    the original Phase 1B winner.
"""

from __future__ import annotations

from typing import Any, Dict

import optuna

from optuna_common import build_arg_parser, run_study

# The Phase 1B (regrouped) winner -- trial 19, score 1.4444 -- used both as the
# fixed anchor value on the "keep" side of each dimension and as seed point #1 below.
INCUMBENT = {
    "mlp_quat_1": 4096,
    "mlp_quat_2": 2048,
    "mlp_quat_3": 512,
    "mlp_trans_1": 1024,
    "mlp_trans_2": 256,
    "mlp_trans_3": 64,
}

# One step further, in the direction the boundary hit suggested (up for the four
# max-boundary dims, down for the two min-boundary dims).
PUSHED = {
    "mlp_quat_1": 8192,
    "mlp_quat_2": 4096,
    "mlp_quat_3": 256,
    "mlp_trans_1": 2048,
    "mlp_trans_2": 512,
    "mlp_trans_3": 32,
}

SEED_TRIALS = [
    dict(INCUMBENT),  # 1: reference row for this study
    dict(PUSHED),  # 2: push every dimension at once
    {**INCUMBENT, "mlp_quat_1": PUSHED["mlp_quat_1"]},  # 3: isolate quat_1
    {**INCUMBENT, "mlp_quat_3": PUSHED["mlp_quat_3"]},  # 4: isolate quat_3
]


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    # spatial_grid_size fixed at 10, same as the study this refines.
    overrides["model.spatial_grid_size"] = 10

    overrides["model.mlp_quat_1"] = trial.suggest_categorical("mlp_quat_1", [INCUMBENT["mlp_quat_1"], PUSHED["mlp_quat_1"]])
    overrides["model.mlp_quat_2"] = trial.suggest_categorical("mlp_quat_2", [INCUMBENT["mlp_quat_2"], PUSHED["mlp_quat_2"]])
    overrides["model.mlp_quat_3"] = trial.suggest_categorical("mlp_quat_3", [PUSHED["mlp_quat_3"], INCUMBENT["mlp_quat_3"]])
    overrides["model.mlp_trans_1"] = trial.suggest_categorical("mlp_trans_1", [INCUMBENT["mlp_trans_1"], PUSHED["mlp_trans_1"]])
    overrides["model.mlp_trans_2"] = trial.suggest_categorical("mlp_trans_2", [INCUMBENT["mlp_trans_2"], PUSHED["mlp_trans_2"]])
    overrides["model.mlp_trans_3"] = trial.suggest_categorical("mlp_trans_3", [PUSHED["mlp_trans_3"], INCUMBENT["mlp_trans_3"]])

    return overrides


def main():
    parser = build_arg_parser(
        "Phase 1B follow-up: exploitative local search around the regrouped-split head-architecture winner"
    )
    args = parser.parse_args()

    sampler = optuna.samplers.TPESampler(
        multivariate=True,  # model the 6 dims jointly -- needed to respect the quat_1 x quat_3 interaction
        group=True,
        n_startup_trials=len(SEED_TRIALS),  # switch to the (seeded) model right after the enqueued points
        seed=43,  # match this project's deterministic-training convention
    )

    run_study(args, suggest_params, sampler=sampler, enqueue_trials=SEED_TRIALS)


if __name__ == "__main__":
    main()
