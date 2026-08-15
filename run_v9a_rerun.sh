#!/bin/bash
set -e

# Increase file descriptor limit for parallel environments
ulimit -n 8192
export UV_CACHE_DIR=/tmp/lux-fork-uv-cache

echo "Starting v9a (Categorical) full training with pre-warmed weights..."
uv run --locked python run_monobeast.py \
  --config-name survival_strategic_strength_v9a_collision_categorical_1gpu \
  +load_dir=/home/ueda/workspace/Kaggle_Lux_AI_2021_Fork/outputs/survival_strategic/2026-08-15/09-40-34 \
  +checkpoint_file=100096_weights.pt 2>&1 | tee v9a_training_rerun.log

echo "v9a rerun finished."
