import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime


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
    run_multiple_trainings(
        configs=args.configs,
        batch_sizes=args.batch_sizes,
        tb_port=args.tensorboard_port,
        reset_tb_dir=not args.no_reset_tb_dir,
        cleanup_tb_dir=args.cleanup_tb_dir,
    )
