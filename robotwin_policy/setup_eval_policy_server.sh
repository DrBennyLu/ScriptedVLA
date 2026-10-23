#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"127.0.0.1"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
USER_ROOT=/data/share/1919650160032350208/users/lxg
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[SERVER] policy=scripted_vla task=${task_name} ckpt=${ckpt_name} port=${policy_server_port}"

source "${USER_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

export PYTHONPATH="${USER_ROOT}/ScriptedVLA/src:${XPL_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"

exec python "${XPL_ROOT}/setup_policy_server.py" \
  --config_path "${yaml_file}" \
  --overrides \
  port="${policy_server_port}" \
  host="${policy_server_host}" \
  bench_name="${bench_name}" \
  task_name="${task_name}" \
  ckpt_name="${ckpt_name}" \
  action_horizon="8" \
  env_cfg_type="${env_cfg_type}" \
  seed="${seed}" \
  policy_name="scripted_vla" \
  action_type="${action_type}"
