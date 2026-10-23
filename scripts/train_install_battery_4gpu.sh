#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU (DDP) training for ScriptedVLA on install_battery.
# Designed for 4x NVIDIA H20 on Midea H20 / torchrun.
#
# batch_size in the yaml is PER-GPU.
#   global_batch ≈ batch_size * NUM_GPUS * NNODES
#
# Example (4 GPUs, keep per-GPU bs=32 → global 128):
#   NUM_GPUS=4 bash scripts/train_install_battery_4gpu.sh
#
# Example (match previous global bs=32):
#   NUM_GPUS=4 PER_GPU_BATCH=8 bash scripts/train_install_battery_4gpu.sh
#
# Hosted multi-node (optional):
#   NNODES=$MLP_WORKER_NUM NODE_RANK=$MLP_ROLE_INDEX \
#   MASTER_ADDR=$MLP_WORKER_0_HOST MASTER_PORT=$MLP_WORKER_0_PORT \
#   NUM_GPUS=$MLP_WORKER_GPU bash scripts/train_install_battery_4gpu.sh

USER_ROOT="${USER_ROOT:-/data/share/1919650160032350208/users/lxg}"
SCRIPTED_VLA_ROOT="${SCRIPTED_VLA_ROOT:-${USER_ROOT}/ScriptedVLA}"
LEROBOT_SRC="${LEROBOT_SRC:-${USER_ROOT}/lerobot/src}"
CONDA_ROOT="${CONDA_ROOT:-${USER_ROOT}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-scriptedvla}"
CONFIG="${CONFIG:-${SCRIPTED_VLA_ROOT}/config_install_battery.yaml}"

NUM_GPUS="${NUM_GPUS:-${MLP_WORKER_GPU:-4}}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29501}}"
PER_GPU_BATCH="${PER_GPU_BATCH:-}"   # if set, overrides yaml via --batch_size? train.py may not have it; edit yaml instead

export CRYPTOGRAPHY_OPENSSL_NO_LEGACY="${CRYPTOGRAPHY_OPENSSL_NO_LEGACY:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${LEROBOT_SRC}:${PYTHONPATH:-}"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "[error] conda.sh not found: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${SCRIPTED_VLA_ROOT}"

VISIBLE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
echo "[env] python=$(which python) ($(python -V 2>&1))"
echo "[env] visible_gpus=${VISIBLE} requested_num_gpus=${NUM_GPUS}"
echo "[dist] nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[train] config=${CONFIG}"
echo "[train] PYTHONPATH has lerobot src for v3 dataset"

if [[ "${VISIBLE}" -lt "${NUM_GPUS}" ]]; then
  cat >&2 <<EOF
[error] Only ${VISIBLE} GPU(s) visible, but NUM_GPUS=${NUM_GPUS}.
        Current H20 session must be a multi-GPU instance (e.g. 4x H20).
        Re-provision / switch to a 4-GPU job, then re-run this script.
EOF
  exit 1
fi

if [[ -n "${PER_GPU_BATCH}" ]]; then
  echo "[warn] PER_GPU_BATCH=${PER_GPU_BATCH} is set, but train.py reads batch_size from yaml."
  echo "       Edit training.batch_size in ${CONFIG} (this is per-GPU under DDP)."
fi

# Optional: override max_steps from env
EXTRA_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${SAVE_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--save_steps "${SAVE_STEPS}")
fi
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi

echo "[launch] torchrun nproc_per_node=${NUM_GPUS}"
torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"
