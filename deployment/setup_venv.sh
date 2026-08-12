#!/usr/bin/env sh
set -eu

agent_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_version=${PYTHON_VERSION:-3.9}
torch_index_url=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}

uv venv --python "$python_version" "$agent_dir/.venv"
uv pip install --python "$agent_dir/.venv/bin/python" \
  numpy==1.24.4 gym==0.26.2 pyyaml==6.0.3 scipy==1.13.1 kaggle-environments==1.12.0
uv pip install --python "$agent_dir/.venv/bin/python" \
  --index-url "$torch_index_url" torch==2.7.1

echo "Agent environment is ready: $agent_dir/.venv"
