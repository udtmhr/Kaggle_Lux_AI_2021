#!/bin/bash
set -e

# v11 Long Run Script (5M steps, 2GPU)
uv run --locked python run_monobeast.py \
  --config-name survival_strategic_strength_v11_longrun_1gpu \
  +load_dir=/home/ueda/workspace/Kaggle_Lux_AI_2021_Fork/outputs/survival_strategic/2026-08-15/12-02-58 \
  +checkpoint_file=100096_weights.pt \
  ++total_steps=5000000 \
  ++weights_only=true \
  "$@"
