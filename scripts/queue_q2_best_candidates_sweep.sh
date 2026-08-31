#!/bin/bash
# Waits for the currently-running q2_gating_granularity_local_hybrid sweep to
# finish, then launches the q2_best_candidates_splitseed_sweep (queued by
# request on 2026-08-30). Safe to leave running detached (nohup + disown).
set -e
cd /home/MA_Flavia/Documents/FusionModel

echo "[$(date)] Waiting for current sweep (q2_gating_granularity_local_hybrid) to finish..."
while pgrep -f "train_multiple_runs.py --config_dir configs/generated/q2_gating_granularity_local_hybrid" > /dev/null; do
  sleep 60
done
echo "[$(date)] Current sweep finished. Starting q2_best_candidates_splitseed_sweep."

python scripts/train_multiple_runs.py \
  --config_dir configs/generated/q2_best_candidates_splitseed_sweep/ \
  --run_directory runs/q2_best_candidates_splitseed_sweep

echo "[$(date)] q2_best_candidates_splitseed_sweep finished."
