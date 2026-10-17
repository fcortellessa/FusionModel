#!/usr/bin/env python3
"""
Reports live status of an Optuna study started by optuna_phase*.py, without waiting
for the study to finish. Safe to run at any time, including while a search is actively
running in another process/session -- it only reads the SQLite DB and trial folders,
never writes to them.

Usage:
    python scripts/optuna_tuning/optuna_status.py \\
        --study-name phase1_regularization \\
        --storage sqlite:///runs/optuna_phase1_regularization/study.db
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna

from optuna_common import resolve_path


def main():
    parser = argparse.ArgumentParser(description="Check live status of an Optuna study")
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True, help="e.g. sqlite:///runs/optuna_phase1_regularization/study.db")
    parser.add_argument("--output-dir", type=str, default=None, help="Trial output dir, to also report which trial folder is currently in progress (defaults to the directory containing the storage DB)")
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    trials = study.get_trials(deepcopy=False)

    states = {}
    for t in trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1

    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Total trials recorded: {len(trials)}")
    for state, count in sorted(states.items()):
        print(f"  {state}: {count}")

    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = min(completed, key=lambda t: t.value)
        print(f"\nBest so far: trial #{best.number}  score={best.value:.4f}")
        for k, v in best.params.items():
            print(f"    {k}: {v}")

        print("\nAll completed trials, ranked (ascending = better):")
        for t in sorted(completed, key=lambda t: t.value):
            print(f"  #{t.number:04d}  score={t.value:.4f}")

    running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
    if running:
        print(f"\nCurrently running: trial(s) {[t.number for t in running]}")

    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    if pruned:
        print(f"\nPruned trials: {[t.number for t in pruned]}")
        latest_pruned = max(pruned, key=lambda t: t.number)
        last_reported = latest_pruned.intermediate_values
        if last_reported:
            last_epoch = max(last_reported.keys())
            print(f"  (most recent pruned trial #{latest_pruned.number} was killed after epoch {last_epoch})")

    # Best-effort: peek at trial folders on disk for progress within the currently
    # running trial (its own metrics.csv keeps growing even before Optuna scores it).
    output_dir = resolve_path(args.output_dir) if args.output_dir else Path(args.storage.replace("sqlite:///", "")).resolve().parent
    if running and output_dir.exists():
        for t in running:
            trial_dir = output_dir / f"trial_{t.number:04d}"
            metrics_candidates = list(trial_dir.glob("runs/*/metrics.csv"))
            if metrics_candidates:
                import pandas as pd
                df = pd.read_csv(metrics_candidates[0])
                if not df.empty:
                    last_row = df.sort_values("epoch").iloc[-1]
                    print(
                        f"\nTrial #{t.number} in progress: epoch {int(last_row['epoch'])}, "
                        f"val_loss={last_row.get('val_loss', float('nan')):.4f}"
                    )


if __name__ == "__main__":
    main()
