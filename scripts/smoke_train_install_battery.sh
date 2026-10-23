#!/usr/bin/env bash
set -euo pipefail

# Smoke-test ScriptedVLA training on install_battery.
# Requires conda env `scriptedvla` and the local lerobot checkout (v3 dataset support).

USER_ROOT="${USER_ROOT:-/data/share/1919650160032350208/users/lxg}"
SCRIPTED_VLA_ROOT="${SCRIPTED_VLA_ROOT:-${USER_ROOT}/ScriptedVLA}"
LEROBOT_SRC="${LEROBOT_SRC:-${USER_ROOT}/lerobot/src}"
CONDA_ROOT="${CONDA_ROOT:-${USER_ROOT}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-scriptedvla}"
CONFIG="${CONFIG:-${SCRIPTED_VLA_ROOT}/config_install_battery_smoke.yaml}"
CACHE_DIR="${CACHE_DIR:-${SCRIPTED_VLA_ROOT}/cache/models}"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "conda.sh not found: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONPATH="${LEROBOT_SRC}:${PYTHONPATH:-}"
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY="${CRYPTOGRAPHY_OPENSSL_NO_LEGACY:-1}"

cd "${SCRIPTED_VLA_ROOT}"

# Prefetch Qwen2-VL if missing (HF is reachable on this H20)
MARKER="${CACHE_DIR}/.qwen2vl2b.ok"
if [[ ! -f "${MARKER}" ]]; then
  echo "[smoke] downloading Qwen2-VL-2B-Instruct into ${CACHE_DIR}"
  mkdir -p "${CACHE_DIR}"
  python download_model.py --model "Qwen/Qwen2-VL-2B-Instruct" --cache-dir "${CACHE_DIR}"
  touch "${MARKER}"
else
  echo "[smoke] using cached Qwen model (${MARKER})"
fi

echo "[smoke] train with ${CONFIG}"
python train.py --config "${CONFIG}"
echo "[smoke] DONE_OK"
