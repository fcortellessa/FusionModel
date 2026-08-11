#!/usr/bin/env python3
"""
Compares two Optuna head-architecture-grid run directories (e.g. a "leaky" data split
study vs. a "regrouped"/group-aware split study) trial-by-trial -- but matched by the
actual `model.*` hyperparameters each trial used, NOT by trial index.

Why not just compare trial_0000 to trial_0000: these grid searches (see
optuna_phase1b_head_architecture_grid10.py) sample mlp_quat_*/mlp_trans_* combinations
via trial.suggest_categorical(), which Optuna sampler shuffles independently per study.
Two different --output-dir studies almost never enumerate the same combo in the same
order, so "trial_0000" in one directory is very likely a different architecture than
"trial_0000" in the other. Matching on the actual config is the only fair comparison.

For each matched pair, reports the same "combined score" the search itself optimizes
(see optuna_common.score_final): mean(val_abs_rot_x) + L2-norm(mean(val_abs_y),
mean(val_abs_z)) in mm, averaged over the last --tail-epochs epochs. Lower is better.

Usage:
    python scripts/optuna_tuning/compare_split_runs.py \\
        --dir-a runs/optuna_phase1b_head_architecture_grid10 \\
        --dir-b runs/optuna_phase1b_head_architecture_grid10_regrouped \\
        --label-a leaky --label-b regrouped
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Fields that define "the same architecture config" for this comparison. Deliberately
# scoped to scripts/optuna_tuning/optuna_phase1b_head_architecture_grid10.py's search
# space plus the head-shape fields that phase fixes rather than searches -- NOT the
# whole model.* block, since that also includes fields (e.g. backbone) that are
# constant across a study and irrelevant to matching.
MATCH_FIELDS = [
    "fusion",
    "head_type",
    "dual_fusion",
    "translation_head_mode",
    "spatial_grid_size",
    "mlp_quat_1",
    "mlp_quat_2",
    "mlp_quat_3",
    "mlp_trans_1",
    "mlp_trans_2",
    "mlp_trans_3",
]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def find_trial_run_dir(trial_dir: Path) -> Optional[Path]:
    """A trial dir holds runs/<timestamp>_.../{config.yml,metrics.csv} plus a
    runs/tensorboard/ dir to skip."""
    candidates = sorted(trial_dir.glob("runs/*/config.yml"))
    return candidates[0].parent if candidates else None


def config_key(config: Dict[str, Any]) -> tuple:
    model = config.get("model", {})
    return tuple(model.get(f) for f in MATCH_FIELDS)


def score_trial(metrics_path: Path, tail_epochs: int) -> Dict[str, Any]:
    df = pd.read_csv(metrics_path)
    df = df.sort_values("epoch")
    n_epochs = int(df["epoch"].max())
    tail = df.tail(min(tail_epochs, len(df)))
    val_rot_x = tail["val_abs_rot_x"].mean()
    val_y = tail["val_abs_y"].mean()
    val_z = tail["val_abs_z"].mean()
    trans_l2_mm = ((val_y * 1000) ** 2 + (val_z * 1000) ** 2) ** 0.5
    combined_score = float(val_rot_x + trans_l2_mm)
    return {
        "n_epochs": n_epochs,
        "complete": n_epochs >= 80,  # this phase's configs all train for 80 epochs
        "combined_score": combined_score,
        "val_rot_x_deg": float(val_rot_x),
        "trans_l2_mm": float(trans_l2_mm),
        "best_val_loss": float(df["best_loss"].min()) if "best_loss" in df.columns else float("nan"),
    }


def load_trials(study_dir: Path, tail_epochs: int) -> List[Dict[str, Any]]:
    trials = []
    for trial_dir in sorted(study_dir.glob("trial_*")):
        run_dir = find_trial_run_dir(trial_dir)
        if run_dir is None:
            continue  # e.g. trial still setting up, no config/metrics written yet
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists() or metrics_path.stat().st_size == 0:
            continue
        with open(run_dir / "config.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        try:
            scores = score_trial(metrics_path, tail_epochs)
        except (KeyError, ValueError):
            continue
        trials.append({
            "trial": trial_dir.name,
            "run_dir": run_dir,
            "config": config,
            "key": config_key(config),
            **scores,
        })
    return trials


def format_config(key: tuple) -> str:
    return ", ".join(f"{f}={v}" for f, v in zip(MATCH_FIELDS, key))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir-a", required=True, help="First study's output-dir, e.g. runs/optuna_phase1b_head_architecture_grid10")
    parser.add_argument("--dir-b", required=True, help="Second study's output-dir, e.g. runs/optuna_phase1b_head_architecture_grid10_regrouped")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--tail-epochs", type=int, default=5, help="Matches the search's own objective convention")
    parser.add_argument("--include-incomplete", action="store_true", help="Also score trials that haven't finished all 80 epochs yet (default: skip them, since a short/interrupted run isn't a fair comparison)")
    parser.add_argument("--csv-out", default=None, help="Optional path to write the full matched-pair table as CSV")
    args = parser.parse_args()

    dir_a = resolve_path(args.dir_a)
    dir_b = resolve_path(args.dir_b)

    trials_a = load_trials(dir_a, args.tail_epochs)
    trials_b = load_trials(dir_b, args.tail_epochs)

    if not args.include_incomplete:
        trials_a = [t for t in trials_a if t["complete"]]
        trials_b = [t for t in trials_b if t["complete"]]

    print(f"{args.label_a} ({dir_a}): {len(trials_a)} scorable trial(s)")
    print(f"{args.label_b} ({dir_b}): {len(trials_b)} scorable trial(s)")
    print()

    by_key_b: Dict[tuple, List[Dict[str, Any]]] = {}
    for t in trials_b:
        by_key_b.setdefault(t["key"], []).append(t)

    matches = []
    matched_b_trials = set()
    for ta in trials_a:
        candidates = by_key_b.get(ta["key"], [])
        for tb in candidates:
            matches.append((ta, tb))
            matched_b_trials.add(tb["trial"])

    if not matches:
        print("No trials with an IDENTICAL head-architecture config were found in both directories.")
        print("(Each study samples the search space independently/randomly, so trial_0000 vs.")
        print(" trial_0000 is comparing two different architectures -- see script docstring.)")
        print()
        print(f"{args.label_a} trial_0000 config: {format_config(next((t['key'] for t in trials_a if t['trial'] == 'trial_0000'), ()))}" if any(t["trial"] == "trial_0000" for t in trials_a) else "")
        print(f"{args.label_b} trial_0000 config: {format_config(next((t['key'] for t in trials_b if t['trial'] == 'trial_0000'), ()))}" if any(t["trial"] == "trial_0000" for t in trials_b) else "")
        print()
        print("Configs sampled so far:")
        print(f"\n{args.label_a}:")
        for t in trials_a:
            print(f"  {t['trial']}: {format_config(t['key'])}  (score={t['combined_score']:.3f}, {t['n_epochs']} epochs)")
        print(f"\n{args.label_b}:")
        for t in trials_b:
            print(f"  {t['trial']}: {format_config(t['key'])}  (score={t['combined_score']:.3f}, {t['n_epochs']} epochs)")
        return

    rows = []
    for ta, tb in matches:
        delta = tb["combined_score"] - ta["combined_score"]
        pct = 100.0 * delta / ta["combined_score"] if ta["combined_score"] else float("nan")
        rows.append({
            "config": format_config(ta["key"]),
            f"{args.label_a}_trial": ta["trial"],
            f"{args.label_b}_trial": tb["trial"],
            f"{args.label_a}_score": round(ta["combined_score"], 3),
            f"{args.label_b}_score": round(tb["combined_score"], 3),
            "delta_b_minus_a": round(delta, 3),
            "pct_change": round(pct, 1),
            f"{args.label_a}_rot_deg": round(ta["val_rot_x_deg"], 3),
            f"{args.label_b}_rot_deg": round(tb["val_rot_x_deg"], 3),
            f"{args.label_a}_trans_l2_mm": round(ta["trans_l2_mm"], 3),
            f"{args.label_b}_trans_l2_mm": round(tb["trans_l2_mm"], 3),
        })

    df = pd.DataFrame(rows).sort_values("delta_b_minus_a")
    print(f"Matched {len(df)} trial pair(s) with identical head-architecture config:\n")
    print(df.to_string(index=False))

    if args.csv_out:
        out_path = resolve_path(args.csv_out)
        df.to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")

    unmatched_a = [t for t in trials_a if not any(t is ta for ta, _ in matches)]
    unmatched_b = [t for t in trials_b if t["trial"] not in matched_b_trials]
    if unmatched_a or unmatched_b:
        print(f"\n({len(unmatched_a)} {args.label_a} trial(s) and {len(unmatched_b)} {args.label_b} trial(s) had no config match on the other side and were excluded.)")


if __name__ == "__main__":
    main()
