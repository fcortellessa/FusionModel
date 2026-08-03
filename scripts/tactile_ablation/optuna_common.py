#!/usr/bin/env python3
"""
Shared machinery for the phased Optuna hyperparameter search (optuna_phase1_*.py,
optuna_phase2_*.py, optuna_phase3_*.py). Not a standalone entry point.

Design: scripts/train.py has no in-process return value or callback hook (it writes
metrics.csv per epoch and exits), so each trial is run as a subprocess against a
generated config file. A background thread tails metrics.csv as it grows and reports
each new epoch's val_loss to Optuna via trial.report(), so Optuna's pruner can kill
clearly-losing trials early. This is independent of the model's own
training.early_stopping (if enabled in a phase's search space): pruning compares a
trial against OTHER trials' trajectories, early stopping only looks at a trial's own
history -- the two are complementary, not duplicates.

Final objective (minimize): the same "combined score" used by
scripts/evaluate/analysis.py's leaderboard -- val_abs_rot_x (deg) + L2 norm of
(val_abs_y, val_abs_z) in mm, averaged over the last N epochs actually logged.
"""

from __future__ import annotations

import argparse
import copy
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict

import optuna
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_config_variants import set_nested_value  # noqa: E402

TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def build_trial_config(template: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(template)
    for dotted_path, value in overrides.items():
        set_nested_value(cfg, dotted_path, value)
    return cfg


def get_dotted(cfg: Dict[str, Any], dotted_path: str, default=None):
    node = cfg
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def tail_metrics_csv(
    metrics_path: Path,
    report_queue: "queue.Queue[tuple[int, float]]",
    stop_event: threading.Event,
    poll_interval: float = 5.0,
):
    """Background thread: polls metrics_path for new rows, pushes (epoch, val_loss) tuples."""
    seen_epochs = set()
    while not stop_event.is_set():
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                for _, row in df.iterrows():
                    epoch = int(row["epoch"])
                    if epoch not in seen_epochs and "val_loss" in row and pd.notna(row["val_loss"]):
                        seen_epochs.add(epoch)
                        report_queue.put((epoch, float(row["val_loss"])))
            except (pd.errors.EmptyDataError, KeyError, ValueError):
                pass
        time.sleep(poll_interval)


def score_final(metrics_path: Path, tail_epochs: int) -> float:
    df = pd.read_csv(metrics_path)
    final = df.sort_values("epoch").tail(tail_epochs)
    val_rot_x = final["val_abs_rot_x"].mean()
    val_y = final["val_abs_y"].mean()
    val_z = final["val_abs_z"].mean()
    trans_l2_mm = ((val_y * 1000) ** 2 + (val_z * 1000) ** 2) ** 0.5
    return float(val_rot_x + trans_l2_mm)


def run_trial(
    trial: optuna.Trial,
    template: Dict[str, Any],
    suggest_params: Callable[[optuna.Trial], Dict[str, Any]],
    trials_root: Path,
    tail_epochs: int,
    poll_interval: float,
) -> float:
    overrides = suggest_params(trial)
    cfg = build_trial_config(template, overrides)
    cfg.setdefault("training", {}).setdefault("seed", template.get("training", {}).get("seed", 43))

    trial_dir = trials_root / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    run_directory = trial_dir / "runs"
    proc = subprocess.Popen(
        [
            sys.executable, str(TRAIN_SCRIPT),
            "--config", str(config_path),
            "--run_directory", str(run_directory),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    # metrics.csv path is only known once RunSetupManager creates the run folder;
    # poll for it to appear before tailing.
    metrics_path = None
    while metrics_path is None and proc.poll() is None:
        candidates = list(run_directory.glob("*/metrics.csv"))
        if candidates:
            metrics_path = candidates[0]
        else:
            time.sleep(poll_interval)

    report_queue: "queue.Queue[tuple[int, float]]" = queue.Queue()
    stop_event = threading.Event()
    tail_thread = None
    if metrics_path is not None:
        tail_thread = threading.Thread(
            target=tail_metrics_csv, args=(metrics_path, report_queue, stop_event, poll_interval), daemon=True
        )
        tail_thread.start()

    pruned = False
    try:
        while proc.poll() is None:
            try:
                epoch, val_loss = report_queue.get(timeout=poll_interval)
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    pruned = True
                    proc.terminate()
                    break
            except queue.Empty:
                continue
    finally:
        stop_event.set()
        if tail_thread is not None:
            tail_thread.join(timeout=2 * poll_interval)
        if proc.poll() is None:
            proc.wait(timeout=30)

    if pruned:
        raise optuna.TrialPruned()

    if proc.returncode != 0:
        stderr_tail = proc.stderr.read()[-2000:] if proc.stderr else ""
        raise RuntimeError(f"Trial {trial.number} subprocess failed (exit {proc.returncode}):\n{stderr_tail}")

    if metrics_path is None or not metrics_path.exists():
        raise RuntimeError(f"Trial {trial.number}: no metrics.csv produced")

    return score_final(metrics_path, tail_epochs)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--template-config", type=str, required=True, help="Base config providing every key not searched by this phase")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///runs/optuna_study.db. Required to resume a study across processes.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tail-epochs", type=int, default=5, help="Epochs averaged for the final objective, matching the existing leaderboard convention")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between metrics.csv polls")
    parser.add_argument("--pruner-warmup-epochs", type=int, default=15, help="Epochs before pruning is allowed to kick in for a trial (avoids killing runs during early noisy epochs)")
    return parser


def run_study(
    args: argparse.Namespace,
    suggest_params: Callable[[optuna.Trial], Dict[str, Any]],
    winner_out_path_name: str = "winner_config.yml",
):
    template_path = resolve_path(args.template_config)
    template = load_yaml(template_path)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage or f"sqlite:///{output_dir / 'study.db'}"

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=args.pruner_warmup_epochs)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        return run_trial(trial, template, suggest_params, output_dir, args.tail_epochs, args.poll_interval)

    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "=" * 88)
    print(f"Best trial: #{study.best_trial.number}  score={study.best_value:.4f}")
    print("=" * 88)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    trials_df = study.trials_dataframe()
    trials_csv = output_dir / "trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    print(f"\nFull trial history written to {trials_csv}")
    print(f"Study storage: {storage}  (resume with the same --study-name/--storage)")

    # Write the winning trial's full resolved config, so it can be fed directly as
    # --template-config to the next phase (or used as a standalone confirmation run).
    best_trial_dir = output_dir / f"trial_{study.best_trial.number:04d}"
    best_config_path = best_trial_dir / "config.yml"
    winner_path = output_dir / winner_out_path_name
    if best_config_path.exists():
        winner_cfg = load_yaml(best_config_path)
        with open(winner_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(winner_cfg, f, sort_keys=False)
        print(f"Winning config copied to {winner_path} (use as --template-config for the next phase)")
    else:
        print(f"Warning: best trial's config not found at {best_config_path}; winner_config.yml not written")

    return study
