#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_valuesketch16_alpha05_generality_128k_6gpu}"
WORKER="${ROOT}/scripts/run_qksieve_valuesketch_alpha_pair_20260801.sh"
VARIANT="qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280"

mkdir -p "${RUN_ROOT}/launcher_logs"

launch_worker() {
  local gpus="$1"
  local topics="$2"
  local label="$3"
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    GPU_IDS="${gpus}" \
    ALPHA="0.5" \
    TOPICS="${topics}" \
    VARIANT="${VARIANT}" \
    bash "${WORKER}" >"${RUN_ROOT}/launcher_logs/${label}.log" 2>&1 &
  WORKER_PID="$!"
}

# The first wave completes the original six-topic, same-seed protocol.
pids=()
launch_worker "0,1" "politics:20260834" "same_politics"
pids+=("${WORKER_PID}")
launch_worker "2,3" "religion:20260835" "same_religion"
pids+=("${WORKER_PID}")
launch_worker "4,5" "mixed_b:20260836" "same_mixed"
pids+=("${WORKER_PID}")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/SAME_SEED_COMPLETE"

# This second wave is held out from the alpha choice.
pids=()
launch_worker "0,1" "sports_both:20260931 politics:20260934" "heldout_a"
pids+=("${WORKER_PID}")
launch_worker "2,3" "medicine:20260932 religion:20260935" "heldout_b"
pids+=("${WORKER_PID}")
launch_worker "4,5" "computer:20260933 mixed_b:20260936" "heldout_c"
pids+=("${WORKER_PID}")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
