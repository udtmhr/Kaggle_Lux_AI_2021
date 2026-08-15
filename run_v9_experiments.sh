#!/bin/bash
set -e

# Increase file descriptor limit for parallel environments
ulimit -n 8192
export UV_CACHE_DIR=/tmp/lux-fork-uv-cache

echo "Starting v9a (Categorical + Warmup 15k) full training..."
uv run --locked python run_monobeast.py \
  --config-name survival_strategic_strength_v9a_collision_categorical_1gpu \
  +load_dir=/home/ueda/workspace/Kaggle_Lux_AI_2021_Fork/outputs/eval_behavior_kl \
  +checkpoint_file=100096_weights.pt 2>&1 | tee v9a_training.log

echo "v9a finished. Starting v9b (MSE) full training..."
uv run --locked python run_monobeast.py \
  --config-name survival_strategic_strength_v9b_collision_mse_1gpu \
  +load_dir=/home/ueda/workspace/Kaggle_Lux_AI_2021_Fork/outputs/eval_behavior_kl \
  +checkpoint_file=100096_weights.pt 2>&1 | tee v9b_training.log

echo "All trainings finished."
