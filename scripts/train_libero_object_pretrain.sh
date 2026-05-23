#!/usr/bin/env bash
# Stage 1: pretrain on all 10 libero-object tasks
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync
uv run python train.py --config config_libero_object_pretrain.yaml --dataset_path ./dada/libero-object "$@"
