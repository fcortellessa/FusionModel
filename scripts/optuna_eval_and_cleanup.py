"""
Evaluate every run in an ablation/sweep folder with test.py (CSV + summary
plots, incl. signed-error plots), then delete that run's checkpoint_best.pth
to free disk space once its results are confirmed on disk.

Handles both directory layouts seen in this repo's runs/, found by searching
recursively for manifest.json (the layout scripts/train.py writes):
    - Optuna-style:  <sweep_dir>/trial_0000/runs/<run_name>/{config.yml,checkpoint_best.pth,manifest.json,results/}
    - Flat ablation: <sweep_dir>/<run_name>/{config.yml,checkpoint_best.pth,manifest.json,results/}
Any subdirectory without its own manifest.json (e.g. a batch-analysis output
folder with only CSVs/plots) is naturally skipped since nothing matches there.

If interrupted, the script can be resumed by running the same command again —
it skips runs that were already successfully evaluated and cleaned up, tracked
in a .optuna_eval_cleanup_state file in the sweep_dir.

Usage:
    python scripts/optuna_eval_and_cleanup.py --sweep_dir runs/optuna_phase1b_head_architecture_grid10
    python scripts/optuna_eval_and_cleanup.py --sweep_dir runs/lr_schedule_arch
    python scripts/optuna_eval_and_cleanup.py --sweep_dir <dir> --dry-run   # evaluate only, keep checkpoints
    python scripts/optuna_eval_and_cleanup.py --sweep_dir <dir> --reset-state  # clear resume state, restart from scratch
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def find_run_dirs(sweep_dir):
    """Return the list of (label, run_dir) pairs under sweep_dir that have a manifest.json, searched recursively."""
    pairs = []
    manifests = sorted(glob.glob(os.path.join(sweep_dir, "**", "manifest.json"), recursive=True))
    for manifest_path in manifests:
        run_dir = os.path.dirname(manifest_path)
        label = os.path.relpath(run_dir, sweep_dir)
        pairs.append((label, run_dir))
    return pairs


def get_state_file(sweep_dir):
    """Return the path to the state file that tracks processed runs."""
    return os.path.join(sweep_dir, ".optuna_eval_cleanup_state")


def load_state(state_file):
    """Load the set of runs already successfully processed, and total MB freed so far."""
    if not os.path.exists(state_file):
        return set(), 0.0
    try:
        with open(state_file, "r") as f:
            data = json.load(f)
        return set(data.get("processed_runs", [])), data.get("total_freed_mb", 0.0)
    except (json.JSONDecodeError, IOError):
        # Corrupt state file; start fresh
        print(f"[warn] State file {state_file} is corrupt or unreadable; starting fresh.")
        return set(), 0.0


def save_state(state_file, processed_runs, total_freed_mb):
    """Save the set of processed runs and total freed space to the state file."""
    with open(state_file, "w") as f:
        json.dump({
            "processed_runs": sorted(processed_runs),
            "total_freed_mb": total_freed_mb,
        }, f, indent=2)


def reset_state(state_file):
    """Remove the state file to force a restart from scratch."""
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"[info] State file deleted: {state_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True, help="Path to the optuna sweep folder containing trial_* subdirs")
    parser.add_argument("--plot_format", nargs="+", default=["png"], choices=["png", "svg", "html", "all"])
    parser.add_argument("--dry-run", action="store_true", help="Run test.py but do not delete any checkpoint")
    parser.add_argument("--keep-going", action="store_true", help="Continue past a trial that fails instead of aborting")
    parser.add_argument("--reset-state", action="store_true", help="Clear resume state and restart from scratch")
    args = parser.parse_args()

    sweep_dir = os.path.abspath(args.sweep_dir)
    state_file = get_state_file(sweep_dir)

    # Handle --reset-state flag
    if args.reset_state:
        reset_state(state_file)
        return

    pairs = find_run_dirs(sweep_dir)
    if not pairs:
        print(f"No trials with a manifest.json found under {sweep_dir}")
        return

    # Load previously processed runs and total freed space
    processed_runs, total_freed = load_state(state_file)
    if processed_runs:
        print(f"[info] Resuming from saved state: {len(processed_runs)} runs already processed, {total_freed:.0f} MB freed so far.")

    failures = []
    new_processed = 0

    for trial_name, run_dir in pairs:
        ckpt_path = os.path.join(run_dir, "checkpoint_best.pth")
        results_dir = os.path.join(run_dir, "results")

        # Skip already-processed runs (unless in dry-run mode, where we skip nothing)
        if not args.dry_run and trial_name in processed_runs:
            print(f"[resume-skip] {trial_name}: already processed in a previous run")
            continue

        if not os.path.exists(ckpt_path):
            print(f"[skip] {trial_name}: no checkpoint_best.pth at {run_dir} (already cleaned up?)")
            continue

        print(f"\n=== {trial_name} ({run_dir}) ===")
        before_csvs = set(glob.glob(os.path.join(results_dir, "*.csv")))

        cmd = [
            sys.executable, os.path.join(REPO_ROOT, "scripts", "test.py"),
            "--run_folder", run_dir,
            "--plot_format", *args.plot_format,
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT)

        if proc.returncode != 0:
            print(f"[FAIL] {trial_name}: test.py exited with code {proc.returncode}; checkpoint kept.")
            failures.append(trial_name)
            if not args.keep_going:
                break
            continue

        after_csvs = set(glob.glob(os.path.join(results_dir, "*.csv")))
        new_csvs = after_csvs - before_csvs
        new_plots = glob.glob(os.path.join(results_dir, "summary_plots_*"))
        if not new_csvs:
            print(f"[FAIL] {trial_name}: no new results CSV found in {results_dir}; checkpoint kept.")
            failures.append(trial_name)
            if not args.keep_going:
                break
            continue
        if not new_plots:
            print(f"[FAIL] {trial_name}: no summary plots found in {results_dir}; checkpoint kept.")
            failures.append(trial_name)
            if not args.keep_going:
                break
            continue

        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        if args.dry_run:
            print(f"[dry-run] {trial_name}: results OK ({len(new_csvs)} new CSV, {len(new_plots)} plots); would free {size_mb:.0f} MB")
        else:
            os.remove(ckpt_path)
            total_freed += size_mb
            processed_runs.add(trial_name)
            new_processed += 1
            print(f"[ok] {trial_name}: results confirmed, deleted checkpoint_best.pth (freed {size_mb:.0f} MB)")
            # Save state after each successful cleanup to allow resumption if interrupted
            save_state(state_file, processed_runs, total_freed)

    print(f"\nDone. Total freed: {total_freed:.0f} MB.")
    if new_processed > 0:
        print(f"Newly processed runs: {new_processed}. Runs can be resumed by running the same command again.")
    if failures:
        print(f"Trials with failures (checkpoints kept): {failures}")


if __name__ == "__main__":
    main()
