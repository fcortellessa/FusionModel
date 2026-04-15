import os
import subprocess


if __name__ == "__main__":
    # Define the configurations to run
    configs = [
    "configs/gated.yml",
    "configs/default.yml",
    ]

    batch_sizes = [16, 32, 64]

    for batch_size in batch_sizes:
        for config in configs:
            print(f"Running training with config: {config} and batch size: {batch_size}")
            command = f"python scripts/train.py --config {config} --batch_size {batch_size}"
            subprocess.run(command, shell=True)
