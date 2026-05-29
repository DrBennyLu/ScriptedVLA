#!/usr/bin/env bash
# Full fine-tune on libero-object dataset (uv environment)
set -euo pipefail
cd "$(dirname "$0")/../.."
uv sync
uv run python train.py --config libero/config_libero_object.yaml --dataset_path ./dada/libero-object "$@"
