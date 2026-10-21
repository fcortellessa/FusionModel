#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --gpus=A100:1
#SBATCH --partition=kisski
#SBATCH --cpus-per-task=6
#SBATCH --array=0-1
#SBATCH --job-name=fusion
#SBATCH --output=slurm_out/%x_%A_%a.out

# --------------------------------------------------------------------------
# One array task = one config of a config list: train (scripts/train.py) and
# evaluate (scripts/test.py) it, via scripts/cluster/run_array_task.py.
#
# Submit FROM THE REPO ROOT (the --output path above is relative to it, and
# slurm_out/ must exist before submitting -- SLURM does not create it):
#
#   mkdir -p slurm_out
#   N=$(grep -cvE '^\s*(#|$)' lists/cluster_batch1.txt)
#   sbatch --array=0-$((N-1))%8 \
#          --export=ALL,CONFIG_LIST=lists/cluster_batch1.txt,OUT_ROOT=runs/cluster_batch1 \
#          scripts/cluster/slurm_array.sbatch
#
# The default --array=0-1 above is a 2-task smoke test; always pass the real
# range on the command line. Task i trains line i of CONFIG_LIST into
# OUT_ROOT/<config name>/<run folder>. Resubmitting the same command is safe:
# finished tasks are skipped, unfinished ones are redone.
#
# Data: FUSIONMODEL_DATA_ROOT must hold splits/ and compact_cache/ (uploaded
# once; check with `sha256sum -c SHA256SUMS` in both folders). Default below.
# Configs must use data.pipeline: compact (run_array_task.py refuses others).
# --------------------------------------------------------------------------

set -euo pipefail

# KISSKI only provides Python 3.11: venv built from requirements-cluster-py311.txt (same torch/numpy/
# ultralytics pins as the 3.8 environment of all existing runs; see that file's header).
module load gcc
module load python/3.11.9
source "$HOME/.venv/fusionmodel/bin/activate"

REPO_DIR="$HOME/repos/FusionModel"
: "${CONFIG_LIST:?pass --export=ALL,CONFIG_LIST=lists/<name>.txt,OUT_ROOT=runs/<name>}"
: "${OUT_ROOT:?pass --export=ALL,CONFIG_LIST=lists/<name>.txt,OUT_ROOT=runs/<name>}"
export FUSIONMODEL_DATA_ROOT="${FUSIONMODEL_DATA_ROOT:-$REPO_DIR/Data}"
export OMP_NUM_THREADS=1   # dataloader workers come from training.num_workers (4); keep < cpus-per-task

cd "$REPO_DIR"
mkdir -p slurm_out

echo "FusionModel array task — job=${SLURM_ARRAY_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-?} host=$(hostname)"
echo "  config list: $CONFIG_LIST"
echo "  out root:    $OUT_ROOT"
echo "  data root:   $FUSIONMODEL_DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/cluster/run_array_task.py \
  --config_list "$CONFIG_LIST" \
  --out_root "$OUT_ROOT"
