#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_agent_checkpoint>"
  echo "Example: $0 outputs/survival_strategic/2026-08-15/.../0100096_weights.pt"
  exit 1
fi

AGENT_PATH=$1
OPPONENT_PATH=/home/ueda/workspace/Kaggle_Lux_AI_2021_Fork/internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research/lux_ai/rl_agent/

echo "Evaluating ${AGENT_PATH} against first_place..."

uv run --locked python -m lux_ai.evaluation.eval_model \
  agent=${AGENT_PATH} \
  opponent=${OPPONENT_PATH} \
  n_games=160 \
  seed=42 \
  "$@"
