#!/usr/bin/env bash
# Stage 2: posttrain on task_index=0, warm-start from pretrain checkpoint
"""
python train.py \
  --config libero/config_libero_object.yaml \
  --dataset_path ./dada/libero-object \
  --init_checkpoint ./checkpoints/libero_object_pretrain/checkpoint_step_100000.pt
"""


set -euo pipefail
cd "$(dirname "$0")/../.."
uv sync

PRETRAIN_CKPT="${1:-./checkpoints/libero_object_pretrain/checkpoint_step_60000.pt}"
if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "Pretrain checkpoint not found: $PRETRAIN_CKPT" >&2
  echo "Usage: $0 [path/to/checkpoint_step_*.pt]" >&2
  exit 1
fi

uv run python train.py \
  --config libero/config_libero_object.yaml \
  --dataset_path ./dada/libero-object \
  --init_checkpoint "$PRETRAIN_CKPT" \
  "${@:2}"
