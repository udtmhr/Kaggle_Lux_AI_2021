#!/bin/bash
set -e

# Make sure we use the uv python environment
ulimit -n 8192
UV_CACHE_DIR=/tmp/lux-fork-uv-cache uv run --locked python run_monobeast.py \
  --config-name beat_first_place_small \
  +load_dir=/home/ueda/workspace/Kaggle_Lux_AI_2021_Fork/outputs/distill_small \
  +checkpoint_file=best.pt \
  weights_only=true \
  total_steps=1000000 \
  batch_size=4 \
  checkpoint_freq=5.0

