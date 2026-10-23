#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11}

USER_ROOT=/data/share/1919650160032350208/users/lxg
ROBOTWIN_ROOT="${USER_ROOT}/RoboTwin"

source "${USER_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${eval_env_conda_env}"

cd "${ROBOTWIN_ROOT}"
export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
export PYTHONPATH="${ROBOTWIN_ROOT}/XPolicyLab:${PYTHONPATH:-}"
RESULT_DIR="${RESULT_DIR:-${USER_ROOT}/ScriptedVLA/results/robotwin_smoke}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean_smoke}"
TEST_NUM="${TEST_NUM:-2}"
mkdir -p "${RESULT_DIR}"

exec python scripts/eval_policy_xpolicylab.py \
  --bench_name "${bench_name}" \
  --task_name "${task_name}" \
  --task_config "${TASK_CONFIG}" \
  --policy_name scripted_vla \
  --host "${policy_server_ip}" \
  --port "${policy_server_port}" \
  --protocol ws \
  --seed "${seed}" \
  --device_id "${env_gpu_id}" \
  --env_cfg_type "${env_cfg_type}" \
  --test_num "${TEST_NUM}" \
  --instruction_type seen \
  --additional_info "${additional_info},action_type=${action_type}"
