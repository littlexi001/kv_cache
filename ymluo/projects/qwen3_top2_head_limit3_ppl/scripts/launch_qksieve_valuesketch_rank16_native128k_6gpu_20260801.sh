#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_valuesketch_rank16_native128k_v25_6gpu}"
RUNNER="${ROOT}/scripts/run_qksieve_native128k_valuesketch_topics_20260801.sh"
VARIANTS="qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280"

mkdir -p "${RUN_ROOT}/launcher_logs"

workers=(
  "0,1|computer:20260833 mixed_b:20260836|computer_mixed"
  "2,3|sports_both:20260831 politics:20260834|sports_politics"
  "4,5|medicine:20260832 religion:20260835|medicine_religion"
)

pids=()
for worker in "${workers[@]}"; do
  IFS="|" read -r gpu_ids topics label <<<"${worker}"
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    GPU_IDS="${gpu_ids}" \
    TOPICS="${topics}" \
    VARIANTS="${VARIANTS}" \
    bash "${RUNNER}" \
    >"${RUN_ROOT}/launcher_logs/${label}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_WORKERS_COMPLETE"
