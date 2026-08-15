#!/bin/bash
set -e

# Increase file descriptor limit for parallel environments
ulimit -n 8192
export UV_CACHE_DIR=/tmp/lux-fork-uv-cache

echo "Starting v10 (Curriculum Learning) full training (100k steps)..."
uv run --locked python run_monobeast.py \
  --config-name survival_strategic_strength_v10_curriculum_1gpu \
  +load_dir=/home/ueda/workspace/Kaggle_Lux_AI_2021_Fork/outputs/survival_strategic/2026-08-15/10-39-34 \
  +checkpoint_file=100096_weights.pt 2>&1 | tee v10_curriculum.log

echo "v10 Curriculum training finished."
