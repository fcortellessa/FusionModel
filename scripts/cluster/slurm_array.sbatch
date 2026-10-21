#!/bin/bash
# SLURM array job: one task = one config of a config list (train + test).
#
# Submit from the repo root, sizing the array to the list (and capping concurrency with %K):
#   N=$(grep -cvE '^\s*(#|$)' lists/my_ablation.txt)
#   sbatch --array=0-$((N-1))%8 \
#          --export=ALL,CONFIG_LIST=lists/my_ablation.txt,OUT_ROOT=runs/my_ablation \
#          scripts/cluster/slurm_array.sbatch
#
# Resubmitting the same command later is safe: finished tasks are skipped (see run_array_task.py).
# Adjust partition / time / memory / environment activation to your cluster.

#SBATCH --job-name=fusion-array
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
# #SBATCH --partition=<gpu-partition>

set -euo pipefail

: "${CONFIG_LIST:?set CONFIG_LIST (a file with one config path per line)}"
: "${OUT_ROOT:?set OUT_ROOT (run output root, e.g. runs/<ablation_name>)}"

cd "${SLURM_SUBMIT_DIR}"            # the repo root you submitted from
mkdir -p logs/slurm

# Environment: adapt to the cluster (module load ..., conda activate ..., or a venv).
source .venv/bin/activate

# Data root holding splits/ and compact_cache/ (prepared once with scripts/cluster/prepare_data.py).
export FUSIONMODEL_DATA_ROOT="${FUSIONMODEL_DATA_ROOT:-${SLURM_SUBMIT_DIR}/Data}"
# Dataloader workers come from training.num_workers; keep them within the allocated CPUs.
export OMP_NUM_THREADS=1

python scripts/cluster/run_array_task.py --config_list "${CONFIG_LIST}" --out_root "${OUT_ROOT}"
