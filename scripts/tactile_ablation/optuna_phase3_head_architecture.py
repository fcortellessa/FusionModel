#!/usr/bin/env python3
"""
Phase 3: prediction-head architecture search (mlp_quat_1/2/3, mlp_trans_1/2/3 widths,
depth fixed at 3 layers per prior decision). Everything else -- lr, regularization,
schedule -- is held fixed at whatever --template-config specifies; run this against
Phase 2's winner_config.yml.

This is the highest-dimensional phase (4x4x3x4x4x4 = 3072 width combinations) --
budget more trials here than Phase 1/2, or expect a noisier result.

Search space:
    model.mlp_quat_1   categorical(256, 512, 1024, 2048)
    model.mlp_quat_2   categorical(256, 512, 1024, 2048)
    model.mlp_quat_3   categorical(128, 256, 512)
    model.mlp_trans_1  categorical(64, 128, 256, 512)
    model.mlp_trans_2  categorical(32, 64, 128, 256)
    model.mlp_trans_3  categorical(16, 32, 64, 128)

Usage:
    python scripts/tactile_ablation/optuna_phase3_head_architecture.py \\
        --template-config runs/optuna_phase2_schedule/winner_config.yml \\
        --n-trials 40 \\
        --study-name phase3_head_architecture \\
        --output-dir runs/optuna_phase3_head_architecture

Output: runs/optuna_phase3_head_architecture/winner_config.yml -- the final tuned
config, combining all three phases' winning values. Run a multi-seed confirmation
pass on this config before treating it as a new baseline (every phase here searched
at a single fixed seed, so the winner should be re-validated across seeds like any
other ablation result in this repo).
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
    parser = build_arg_parser("Phase 3: prediction-head architecture search")
    args = parser.parse_args()
    run_study(args, suggest_params)


if __name__ == "__main__":
    main()
