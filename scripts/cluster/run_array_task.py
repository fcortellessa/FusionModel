"""Run one array task: train the config at line SLURM_ARRAY_TASK_ID of a config list, then evaluate it.

    python scripts/cluster/run_array_task.py --config_list lists/my_ablation.txt --out_root runs/my_ablation

Config list: one config path per line (relative to the repo root or absolute); blank lines and lines
starting with '#' are ignored. Task i trains into <out_root>/<config stem>/<run folder>.

Safe to resubmit the whole array: a task whose run already has TRAINING_DONE and TEST_DONE markers is
skipped; a task whose training finished but evaluation did not only re-runs scripts/test.py; anything
else starts a fresh run folder (an interrupted folder is left in place, without markers).
Configs must use `data.pipeline: compact` (frozen splits + shared read-only cache); the legacy pipeline
rebuilds a shared Data_split cache in place and is not safe for concurrent jobs.
"""
import argparse
import os
import subprocess
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TRAINING_DONE, TEST_DONE = "TRAINING_DONE", "TEST_DONE"


def read_config_list(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def run_dirs(parent):
    if not os.path.isdir(parent):
        return []
    return sorted(
        (os.path.join(parent, d) for d in os.listdir(parent)
         if d != "tensorboard" and os.path.isdir(os.path.join(parent, d))),
        key=os.path.getmtime,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_list", required=True)
    ap.add_argument("--out_root", required=True, help="Run output root, e.g. runs/<ablation_name>")
    ap.add_argument("--task_id", type=int, default=None, help="Defaults to $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--force", action="store_true", help="Train again even if a finished run exists")
    args = ap.parse_args()

    task_id = args.task_id if args.task_id is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        sys.exit("No --task_id given and SLURM_ARRAY_TASK_ID is not set.")
    configs = read_config_list(args.config_list)
    task_id = int(task_id)
    if not 0 <= task_id < len(configs):
        sys.exit(f"task_id {task_id} out of range: {args.config_list} lists {len(configs)} configs (array=0-{len(configs) - 1}).")
    cfg_path = configs[task_id]
    cfg_path = cfg_path if os.path.isabs(cfg_path) else os.path.join(REPO_ROOT, cfg_path)
    cfg = yaml.safe_load(open(cfg_path))
    if str(cfg.get("data", {}).get("pipeline", "legacy")).lower() != "compact":
        sys.exit(f"{cfg_path}: set data.pipeline: compact (e.g. as a fixed_override in the generation spec) "
                 "before running it as an array task.")

    out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(REPO_ROOT, args.out_root)
    parent = os.path.join(out_root, os.path.splitext(os.path.basename(cfg_path))[0])
    print(f"[task {task_id}] {os.path.relpath(cfg_path, REPO_ROOT)} -> {os.path.relpath(parent, REPO_ROOT)}", flush=True)

    existing = run_dirs(parent)
    trained = [d for d in existing if os.path.exists(os.path.join(d, TRAINING_DONE))]
    run_dir = None
    if trained and not args.force:
        run_dir = trained[-1]
        if os.path.exists(os.path.join(run_dir, TEST_DONE)) or args.skip_test:
            print(f"[task {task_id}] already finished: {run_dir}", flush=True)
            return
        print(f"[task {task_id}] training already done, evaluating {run_dir}", flush=True)

    if run_dir is None:
        before = set(run_dirs(parent))
        cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "train.py"), "--config", cfg_path,
               "--run_directory", parent, "--no_tensorboard_server"]
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        new = [d for d in run_dirs(parent) if d not in before]
        if rc != 0 or len(new) != 1:
            sys.exit(f"[task {task_id}] training failed (rc={rc}, new run folders: {new})")
        run_dir = new[0]
        open(os.path.join(run_dir, TRAINING_DONE), "w").close()

    if args.skip_test:
        return
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "test.py"), "--run_folder", run_dir]
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    if rc != 0:
        sys.exit(f"[task {task_id}] evaluation failed (rc={rc}) for {run_dir}")
    open(os.path.join(run_dir, TEST_DONE), "w").close()
    print(f"[task {task_id}] done: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
