#!/bin/bash
set -e

ulimit -n 8192
export UV_CACHE_DIR=/tmp/lux-fork-uv-cache

echo "Starting v10 fine-tuning to beat 1st place model..."
uv run --locked python run_monobeast.py \
  --config-name beat_first_place_v10 \
  +load_dir=/home/ueda/workspace/Kaggle_Lux_AI_2021_Fork/outputs/survival_strategic/2026-08-15/23-58-20 \
  +checkpoint_file=1500160.pt \
  ++total_steps=500000 \
  ++weights_only=true \
  ++checkpoint_freq=5.0 \
  2>&1 | tee rl_finetune_v10.log
