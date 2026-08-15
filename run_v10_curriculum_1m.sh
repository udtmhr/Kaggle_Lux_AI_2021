#!/bin/bash
set -e

# Increase file descriptor limit for parallel environments
ulimit -n 8192
export UV_CACHE_DIR=/tmp/lux-fork-uv-cache

echo "Resuming v10 (Curriculum Learning) for up to 1M steps..."
uv run --locked python run_monobeast.py \
  --config-name survival_strategic_strength_v10_curriculum_1m_1gpu \
  +load_dir=/home/ueda/workspace/Kaggle_Lux_AI_2021_Fork/outputs/survival_strategic/2026-08-15/12-02-58 \
  +checkpoint_file=100096.pt \
  ++total_steps=1000000 \
  ++weights_only=false \
  2>&1 | tee v10_curriculum_1m.log

echo "v10 1M Curriculum training finished."
