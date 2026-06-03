#!/usr/bin/env bash
# Online TD3 ablation recipes (plan Step 2). Run from ScriptedVLA repo root with WS server up.
# Usage: bash libero/scripts/run_online_td3_ablations.sh [A|B|C|D|E|all]

set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-.venv/bin/python}"
CFG="libero/config_libero_object.yaml"
COMMON=(
  -m libero.libero_ws_online_td3
  --config "$CFG"
  --task-id 6
  --chunk-steps 10
  --skip-vla-validation
  --max-train-steps 1000
  --save-every-steps 500
)

run_A() {
  echo "=== Ablation A: offline-only mix (no online gradient from rollout data) ==="
  "$PY" "${COMMON[@]}" \
    --online-sample-ratio 0 \
    --online-checkpoint-dir ./checkpoints/ablation_a_offline_only_1k
}

run_B() {
  echo "=== Ablation B: train/deploy aligned rollout (default in config) ==="
  "$PY" "${COMMON[@]}" \
    --rollout-deterministic --no-rollout-ref-mask \
    --online-checkpoint-dir ./checkpoints/ablation_b_aligned_rollout_1k
}

run_C() {
  echo "=== Ablation C: conservative updates (1x/step, lr x0.1) ==="
  "$PY" "${COMMON[@]}" \
    --train-updates-per-step 1 \
    --actor-lr-scale 0.1 --critic-lr-scale 0.1 \
    --online-sample-ratio 0.1 \
    --online-checkpoint-dir ./checkpoints/ablation_c_conservative_1k
}

run_D() {
  echo "=== Ablation D: offline replay only + normal training ==="
  run_A
}

run_E() {
  echo "=== Ablation E: eval offline ckpt only (no training) ==="
  "$PY" -m libero.debug_td3_checkpoint_curve \
    --config "$CFG" --task-id 6 --init-ids 0 1 2 --skip-vla-validation \
    --checkpoints ./checkpoints/libero_object_rl_td3_task6_0602/td3_agent_step_10000.pt \
    --output-dir ./results/checkpoint_curves/ablation_e_offline_baseline
}

run_ratio_sweep() {
  for ratio in 0 0.1 0.3; do
    echo "=== online_sample_ratio=$ratio ==="
    "$PY" "${COMMON[@]}" \
      --rollout-deterministic --no-rollout-ref-mask \
      --train-updates-per-step 1 \
      --actor-lr-scale 0.1 --critic-lr-scale 0.1 \
      --online-sample-ratio "$ratio" \
      --online-checkpoint-dir "./checkpoints/ablation_ratio_${ratio}_1k"
  done
}

case "${1:-all}" in
  A) run_A ;;
  B) run_B ;;
  C) run_C ;;
  D) run_D ;;
  E) run_E ;;
  ratio) run_ratio_sweep ;;
  all) run_E; run_A; run_B; run_C; run_ratio_sweep ;;
  *) echo "Usage: $0 [A|B|C|D|E|ratio|all]" >&2; exit 1 ;;
esac
