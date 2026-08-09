#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_valuesketch_computer128k_attribution_gpu45}"
RUNNER="${ROOT}/scripts/run_qksieve_native128k_valuesketch_topics_20260801.sh"
VARIANTS="exact_qk_oracle_k1280,qksieve_keymse_requestlocal_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch32i4_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch32i4_sampled_k1280,qksieve_keymse_requestlocal_valuesketch32i4_fulltopk_k2560,qksieve_keymse_requestlocal_valuesketch32i4_sampled_k2560"

mkdir -p "${RUN_ROOT}/launcher_logs"
env \
  ROOT="${ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  GPU_IDS="4,5" \
  TOPICS="computer:20260833" \
  VARIANTS="${VARIANTS}" \
  bash "${RUNNER}" \
  >"${RUN_ROOT}/launcher_logs/computer.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
