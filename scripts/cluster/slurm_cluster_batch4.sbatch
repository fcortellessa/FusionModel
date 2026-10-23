#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --gpus=A100:1
#SBATCH --partition=kisski
#SBATCH --cpus-per-task=6
#SBATCH --array=0-96
#SBATCH --job-name=fusion_b4
#SBATCH --output=slurm_out/%x_%A_%a.out

# --------------------------------------------------------------------------
# cluster_batch4: no divisor, splits 36-45, all local-cohort methods not yet run on the cluster (9 methods x 10 splits + se_gated_cross_attention x 6 splits + gated_cross_attention_2d split 36). One array task = one config of lists/cluster_batch4.txt
# (97 configs), trained with scripts/train.py and evaluated with scripts/test.py via
# scripts/cluster/run_array_task.py.
#
# Submit FROM THE REPO ROOT (slurm_out/ must exist; SLURM does not create it):
#
#   mkdir -p slurm_out && sbatch scripts/cluster/slurm_cluster_batch4.sbatch
#
# Resubmitting is safe: finished tasks (TRAINING_DONE/TEST_DONE markers) are skipped.
# Re-run only some tasks with e.g. --array=3,7,12%8.
#
# Data: FUSIONMODEL_DATA_ROOT must hold splits/ and compact_cache/ (uploaded
# once; check with `sha256sum -c SHA256SUMS` in both folders). Default below.
# Configs must use data.pipeline: compact (run_array_task.py refuses others).
# --------------------------------------------------------------------------

set -euo pipefail

# Python environment. KISSKI's module tree is built per CPU architecture: `module load python` on the
# Intel (Sapphire Rapids) login nodes gives an AVX-512 build that crashes with "Illegal instruction" on the
# AMD Milan (zen3) kisski GPU nodes. So no `module load` here: the venv carries its own interpreter, and it
# must be created from the zen2 (AMD Rome, A100 stack) build of Python 3.11.9, which runs on both:
#   PY=/sw/rev/25.04/rome_mofed_cuda80_rocky8/linux-rocky8-zen2/gcc-13.2.0/python-3.11.9-kukywutwnl5yzx55y6qol3awkq2g7vw6/bin/python3.11
#   $PY -m venv ~/.venv/fusionmodel && source ~/.venv/fusionmodel/bin/activate
#   pip install --upgrade pip && pip install -r requirements-cluster-py311.txt
VENV="${FUSION_VENV:-$HOME/.venv/fusionmodel}"
source "$VENV/bin/activate"

# Preflight: fail fast with a clear message instead of a bare core dump.
PY_REAL="$(readlink -f "$VENV/bin/python")"
echo "python: $PY_REAL"
case "$PY_REAL" in
  *sapphirerapids*|*cascadelake*|*skylake*)
    if ! grep -qw avx512f /proc/cpuinfo; then
      echo "ERROR: $VENV was built from an Intel/AVX-512 Python ($PY_REAL), but this node's CPU has no AVX-512." >&2
      echo "       Recreate the venv from the zen2 Python (see the comment above in this script)." >&2
      exit 1
    fi ;;
esac
python -c "import torch, torchvision, numpy, cv2, ultralytics; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| numpy', numpy.__version__)"

REPO_DIR="$HOME/repos/FusionModel"
CONFIG_LIST="${CONFIG_LIST:-lists/cluster_batch4.txt}"
OUT_ROOT="${OUT_ROOT:-runs/cluster_batch4}"
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
