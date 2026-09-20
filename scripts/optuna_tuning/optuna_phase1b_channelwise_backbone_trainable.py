#!/usr/bin/env python3
"""
Head-architecture search for channelwise_gated_concat_local under backbone_trainable=true
-- the first head-architecture search run under a trainable backbone (every prior phase1
search, including optuna_phase1b_head_architecture_grid10 that established the current
production mlp_quat_1=4096/mlp_quat_2=2048/mlp_quat_3=512/mlp_trans_1=1024/
mlp_trans_2=256/mlp_trans_3=64, ran on a frozen backbone).

Motivated by: colleague's much smaller head (mlp_quat_1=256/mlp_quat_2=1024/
mlp_quat_3=0/mlp_trans_1=1024/mlp_trans_2=1024/mlp_trans_3=16, ~65M total params)
beat the current production head on gated_concat, but LOST to it on
channelwise_gated_concat_local (see q2_colleague_baseline_heads /
q2_colleague_baseline_heads_channelwise, both backbone_trainable=true, 3 seeds each) --
i.e. head architecture and fusion mechanism interact, so the head that's right for
channelwise specifically needs its own justification, not inherited from gated_concat
or from the old frozen-backbone search.

mlp_trans_1 is fixed at 1024 (not searched) -- colleague's and current's values agree
exactly here, so there's no known disagreement to explain and no capacity-hypothesis
question to test on this dimension. It's set directly in --template-config instead.

Search space (3 x 2 x 3 x 2 x 2 = 72 combinations over 5 searched dimensions):
    model.mlp_quat_1   categorical(256, 1024, 4096)   -- colleague's / mid / current
    model.mlp_quat_2   categorical(1024, 2048)        -- colleague's / current
    model.mlp_quat_3   categorical(0, 256, 512)       -- colleague's / mid / current
    model.mlp_trans_2  categorical(256, 1024)         -- current's / colleague's
    model.mlp_trans_3  categorical(16, 64)            -- colleague's / current's

Every dimension now brackets or exactly hits both known reference points (colleague's
and current production's), unlike the first version of this script where mlp_trans_3
was (256, 128) and skipped current's actual value of 64 entirely.

Pruning: MedianPruner with n_warmup_steps=50 (via --pruner-warmup-epochs 50), i.e. no
trial can be pruned before epoch 50 -- deliberately conservative, following the
q2_gate_lr_sweep_spec lesson that Optuna's default pruner warmup (15 epochs) killed
every low-start_lr_mult trial before it had a chance to prove itself, systematically
biased against anything slower to show early improvement.

Early stopping (new): re-enabled in --template-config (it was off in the first version
of this script, deliberately, so every trial ran the same fixed 80 epochs for a clean
comparison) with smoothing_alpha=0.3 -- the same fix built for
q2_channelwise_backbone_trainable_smoothed, where raw per-epoch val_abs_rot_x/
val_abs_trans swung 30-50% and triggered early stopping on noise, not genuine
plateaus. Re-enabling it here trades some of that "every trial same epoch count"
cleanliness for real compute savings -- acceptable specifically because smoothing
means a trial should now only stop once it has genuinely converged, so the comparison
should stay fair (score_final still averages each trial's own last --tail-epochs
epochs, whichever epoch that ends up being) while not paying for epochs a trial
doesn't need. Complementary to, not redundant with, the Optuna pruner: pruning
compares a trial against OTHER trials' trajectories, early stopping only looks at a
trial's own history (see optuna_common.py's module docstring).

enqueue_trials now seeds BOTH known reference points as trials 0 and 1 -- with
mlp_trans_3 corrected to include 64, current production's exact config is valid in
this search space too, not just colleague's.

Usage:
    python scripts/optuna_tuning/optuna_phase1b_channelwise_backbone_trainable.py \\
        --template-config configs/generated/q2_optuna_channelwise_backbone_trainable_template.yml \\
        --n-trials 25 \\
        --study-name phase1b_channelwise_backbone_trainable \\
        --output-dir runs/optuna_phase1b_channelwise_backbone_trainable \\
        --pruner-warmup-epochs 50

Output: runs/optuna_phase1b_channelwise_backbone_trainable/winner_config.yml
"""

from __future__ import annotations

from typing import Any, Dict

import optuna

from optuna_common import build_arg_parser, run_study

# Colleague's exact head architecture (Tabelle 6.1, Fusion column) and current
# production's exact head architecture -- both forced as the first two trials so the
# search directly re-confirms both known points under backbone_trainable=true instead
# of leaving it to chance whether the sampler ever revisits either.
COLLEAGUE_ANCHOR = {
    "mlp_quat_1": 256,
    "mlp_quat_2": 1024,
    "mlp_quat_3": 0,
    "mlp_trans_2": 1024,
    "mlp_trans_3": 16,
}

CURRENT_PRODUCTION_ANCHOR = {
    "mlp_quat_1": 4096,
    "mlp_quat_2": 2048,
    "mlp_quat_3": 512,
    "mlp_trans_2": 256,
    "mlp_trans_3": 64,
}


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    overrides["model.mlp_quat_1"] = trial.suggest_categorical("mlp_quat_1", [256, 1024, 4096])
    overrides["model.mlp_quat_2"] = trial.suggest_categorical("mlp_quat_2", [1024, 2048])
    overrides["model.mlp_quat_3"] = trial.suggest_categorical("mlp_quat_3", [0, 256, 512])
    overrides["model.mlp_trans_2"] = trial.suggest_categorical("mlp_trans_2", [256, 1024])
    overrides["model.mlp_trans_3"] = trial.suggest_categorical("mlp_trans_3", [16, 64])

    return overrides


def main():
    parser = build_arg_parser(
        "Head-architecture search for channelwise_gated_concat_local under backbone_trainable=true"
    )
    args = parser.parse_args()
    run_study(args, suggest_params, enqueue_trials=[COLLEAGUE_ANCHOR, CURRENT_PRODUCTION_ANCHOR])


if __name__ == "__main__":
    main()
