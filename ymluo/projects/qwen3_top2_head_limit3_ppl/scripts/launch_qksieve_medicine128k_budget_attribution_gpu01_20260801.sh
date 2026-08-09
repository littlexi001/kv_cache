#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_medicine128k_budget_attribution_gpu01}"
RUNNER="${ROOT}/scripts/run_qksieve_native128k_valuesketch_topics_20260801.sh"
VARIANTS="exact_qk_oracle_k1280,exact_qk_oracle_k2560,qksieve_keymse_requestlocal_fulltopk_k1280,qksieve_keymse_requestlocal_sampled_k1280_c32,qksieve_keymse_requestlocal_fulltopk_k2560,qksieve_keymse_requestlocal_sampled_k2560_c32"

mkdir -p "${RUN_ROOT}/launcher_logs"
env \
  ROOT="${ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  GPU_IDS="0,1" \
  TOPICS="medicine:20260832" \
  VARIANTS="${VARIANTS}" \
  bash "${RUNNER}" \
  >"${RUN_ROOT}/launcher_logs/medicine.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
