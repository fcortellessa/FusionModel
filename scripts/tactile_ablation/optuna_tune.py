#!/usr/bin/env python3
"""
Optuna hyperparameter search over regularization, LR schedule, and prediction-head
architecture, driven on top of the existing scripts/train.py entry point.

Design: scripts/train.py has no in-process return value or callback hook (it writes
metrics.csv per epoch and exits), so each trial is run as a subprocess against a
generated config file. A background thread tails metrics.csv as it grows and reports
each new epoch's val_loss to Optuna via trial.report(), so Optuna's pruner can kill
clearly-losing trials early (the model's own training.early_stopping, if enabled in the
search space, independently stops a trial once ITS OWN val_loss plateaus -- pruning and
early stopping are complementary, not duplicates: pruning compares against other trials,
early stopping only looks at a trial's own history).

Final objective (minimize): the same "combined score" used by
scripts/evaluate/analysis.py's leaderboard -- val_abs_rot_x (deg) + L2 norm of
(val_abs_y, val_abs_z) in mm, averaged over the last N epochs actually logged.

Search space (fixed at fusion=channelwise_gated_concat, translation_head_mode and
dual_fusion taken from --template-config, epochs/batch_size from --template-config):
    training.lr                          loguniform
    training.weight_decay                loguniform (0 treated as "off" via a separate flag)
    training.dropout_rot                 uniform
    training.dropout_trans               uniform
    training.use_batchnorm               categorical
    training.lr_schedule.enabled/type    categorical (none | warmup_cosine | reduce_on_plateau)
    training.head_lr_decay.enabled       categorical (only meaningful when lr_schedule is off,
                                          matching the "per_head_plateau" arm from prior ablations)
    model.mlp_quat_1/2/3                 categorical (rotation head widths, 3 layers fixed)
    model.mlp_trans_1/2/3                categorical (translation head widths, 3 layers fixed)

Usage:
    python scripts/tactile_ablation/optuna_tune.py \\
        --template-config configs/lr_schedule_ablation_base.yml \\
        --n-trials 30 \\
        --study-name lr_arch_search \\
        --storage sqlite:///runs/optuna_lr_schedule_arch/study.db

Resume an interrupted study by rerunning with the same --study-name/--storage.
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

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


def build_trial_config(template: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    cfg = copy.deepcopy(template)
    for dotted_path, value in overrides.items():
        set_nested_value(cfg, dotted_path, value)
    return cfg


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Returns a dict of dotted-path -> value overrides for this trial."""
    overrides: Dict[str, Any] = {}

    overrides["training.lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    use_weight_decay = trial.suggest_categorical("use_weight_decay", [True, False])
    overrides["training.weight_decay"] = (
        trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) if use_weight_decay else 0.0
    )

    overrides["training.dropout_rot"] = trial.suggest_float("dropout_rot", 0.0, 0.5)
    overrides["training.dropout_trans"] = trial.suggest_float("dropout_trans", 0.0, 0.5)
    overrides["training.use_batchnorm"] = trial.suggest_categorical("use_batchnorm", [True, False])

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
        overrides["training.lr_schedule.patience"] = trial.suggest_int("gp_patience", 3, 10)
        overrides["training.lr_schedule.factor"] = trial.suggest_float("gp_factor", 0.1, 0.5)
        overrides["training.lr_schedule.min_lr"] = trial.suggest_float("gp_min_lr", 1e-6, 1e-4, log=True)
        overrides["training.head_lr_decay.enabled"] = False
    elif schedule_choice == "per_head_plateau":
        overrides["training.lr_schedule.enabled"] = False
        overrides["training.head_lr_decay.enabled"] = True
        overrides["training.head_lr_decay.patience"] = trial.suggest_int("hd_patience", 3, 10)
        overrides["training.head_lr_decay.factor"] = trial.suggest_float("hd_factor", 0.1, 0.5)
        overrides["training.head_lr_decay.min_lr"] = trial.suggest_float("hd_min_lr", 1e-6, 1e-4, log=True)

    overrides["model.mlp_quat_1"] = trial.suggest_categorical("mlp_quat_1", [256, 512, 1024, 2048])
    overrides["model.mlp_quat_2"] = trial.suggest_categorical("mlp_quat_2", [256, 512, 1024, 2048])
    overrides["model.mlp_quat_3"] = trial.suggest_categorical("mlp_quat_3", [128, 256, 512])
    overrides["model.mlp_trans_1"] = trial.suggest_categorical("mlp_trans_1", [64, 128, 256, 512])
    overrides["model.mlp_trans_2"] = trial.suggest_categorical("mlp_trans_2", [32, 64, 128, 256])
    overrides["model.mlp_trans_3"] = trial.suggest_categorical("mlp_trans_3", [16, 32, 64, 128])

    return overrides


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
    trials_root: Path,
    tail_epochs: int,
    poll_interval: float,
) -> float:
    overrides = suggest_params(trial)
    cfg = build_trial_config(template, overrides)
    cfg.setdefault("training", {})["seed"] = template.get("training", {}).get("seed", 43)

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


def main():
    parser = argparse.ArgumentParser(description="Optuna search over regularization/LR-schedule/head-architecture")
    parser.add_argument("--template-config", type=str, required=True, help="Base config providing fixed keys (fusion, epochs, batch_size, translation_head_mode, dual_fusion, data.root, seed, ...)")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", type=str, default="lr_schedule_arch_search")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///runs/optuna_study.db. Required to resume a study across processes.")
    parser.add_argument("--output-dir", type=str, default="runs/optuna_lr_schedule_arch")
    parser.add_argument("--tail-epochs", type=int, default=5, help="Epochs averaged for the final objective, matching the existing leaderboard convention")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between metrics.csv polls")
    parser.add_argument("--pruner-warmup-epochs", type=int, default=15, help="Epochs before pruning is allowed to kick in for a trial (avoids killing runs during early noisy epochs)")
    args = parser.parse_args()

    template_path = Path(args.template_config)
    if not template_path.is_absolute():
        template_path = REPO_ROOT / template_path
    template = load_yaml(template_path)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
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
        return run_trial(trial, template, output_dir, args.tail_epochs, args.poll_interval)

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


if __name__ == "__main__":
    main()
