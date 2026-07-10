import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def discover_configs_from_folder(config_dir):
    config_path = Path(config_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config folder does not exist: {config_dir}")
    if not config_path.is_dir():
        raise NotADirectoryError(f"Expected a directory for --config_dir, got: {config_dir}")

    configs = []
    for path in sorted(config_path.rglob("*.yml")):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception:
            continue

        if isinstance(data, dict) and "model" in data and "training" in data:
            configs.append(str(path))

    if not configs:
        raise ValueError(f"No .yml config files found in: {config_dir}")
    return configs


def run_multiple_trainings(configs, batch_sizes, tb_port, reset_tb_dir=True, cleanup_tb_dir=False):
    session_tag = datetime.now().strftime("multi_run_%Y%m%d_%H%M%S")
    tensorboard_root = os.path.join("logs", "tensorboard", session_tag)

    if reset_tb_dir and os.path.exists(tensorboard_root):
        shutil.rmtree(tensorboard_root)
    os.makedirs(tensorboard_root, exist_ok=True)

    print(f"TensorBoard session root: {tensorboard_root}")
    print(f"Open TensorBoard at: http://127.0.0.1:{tb_port}")

    for batch_size in batch_sizes:
        for config in configs:
            print(f"Running training with config: {config} and batch size: {batch_size}")
            command = [
                sys.executable,
                "scripts/train.py",
                "--config",
                config,
                "--batch_size",
                str(batch_size),
                "--tensorboard_root",
                tensorboard_root,
                "--tensorboard_port",
                str(tb_port),
            ]
            subprocess.run(command, check=True)

    if cleanup_tb_dir:
        shutil.rmtree(tensorboard_root, ignore_errors=True)
        print(f"Removed TensorBoard session root: {tensorboard_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Directory containing .yml config files to run. If set, overrides --configs.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/gated.yml", "configs/default.yml"],
        help="List of config files to run",
    )
    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        type=int,
        default=[16, 32, 64],
        help="List of batch sizes",
    )
    parser.add_argument(
        "--tensorboard_port",
        type=int,
        default=6006,
        help="Port for TensorBoard attach/start behavior",
    )
    parser.add_argument(
        "--no_reset_tb_dir",
        action="store_true",
        help="Do not clear the TensorBoard session root before running",
    )
    parser.add_argument(
        "--cleanup_tb_dir",
        action="store_true",
        help="Delete the TensorBoard session root after the sweep finishes",
    )

    args = parser.parse_args()

    configs = discover_configs_from_folder(args.config_dir) if args.config_dir else args.configs
    run_multiple_trainings(
        configs=configs,
        batch_sizes=args.batch_sizes,
        tb_port=args.tensorboard_port,
        reset_tb_dir=not args.no_reset_tb_dir,
        cleanup_tb_dir=args.cleanup_tb_dir,
    )
