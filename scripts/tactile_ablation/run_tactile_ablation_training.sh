#!/usr/bin/env bash
set -euo pipefail

# Ubuntu-friendly runner for tactile ablation training
# Usage: ./scripts/run_tactile_ablation_training.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

variants=(
  "no_aug"
  "noise_only"
  "brightness_only"
  "contrast_only"
  "blur_only"
  "channel_scaling_only"
  "all"
)

declare -A config_map
config_map[no_aug]="tactile_ablation_baseline.yml"
config_map[noise_only]="tactile_ablation_noise_only.yml"
config_map[brightness_only]="tactile_ablation_brightness_only.yml"
config_map[contrast_only]="tactile_ablation_contrast_only.yml"
config_map[blur_only]="tactile_ablation_blur_only.yml"
config_map[channel_scaling_only]="tactile_ablation_channel_scaling_only.yml"
config_map[all]="tactile_ablation_all.yml"

seeds=(42 43 44)

DEFAULT_PYTHON="python3"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$REPO_ROOT/.venv/bin/python"
fi
PYTHON=${PYTHON:-$DEFAULT_PYTHON}

echo "Repository root: $REPO_ROOT"

# base_folder="${DATA_SPLIT_FOLDER:-data_a_test/Data_split}"
base_folder="${DATA_SPLIT_FOLDER:-Data/Data_split}"

echo "Using data split folder: $base_folder"

for v in "${variants[@]}"; do
  variant_folder="$base_folder/train_augmentation_${v}"
  if [[ ! -d "$variant_folder" ]]; then
    echo "WARNING: variant folder not found: $variant_folder -- skipping"
    echo "         Hint: generate variants first with scripts/tactile_ablation_data_augm.py"
    continue
  fi

  echo "\n=== Running variant: $v ==="
  echo "Using folder: $variant_folder"

  config=${config_map[$v]}
  if [[ -z "$config" ]]; then
    echo "No config mapped for variant $v; skipping"
    continue
  fi

  for seed in "${seeds[@]}"; do
    echo "--> Run for variant=$v seed=$seed"

    # Note: The original PowerShell script loops over seeds but doesn't pass them into train.py.
    # If you want to vary a seed, either edit the configs or modify train.py to accept a --seed flag.
    $PYTHON scripts/train.py --config "configs/$config" --force_preprocess

    if [[ $? -ne 0 ]]; then
      echo "Training failed for variant=$v seed=$seed" >&2
      exit 1
    fi
  done

  echo "Completed variant: $v"
done

echo "\nAll variants processed."
