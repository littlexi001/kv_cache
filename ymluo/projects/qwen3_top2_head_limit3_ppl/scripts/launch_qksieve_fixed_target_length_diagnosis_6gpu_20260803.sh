#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260803_qksieve_fixed_target_length_causal_alpha05_6gpu}"
WORKER="${ROOT}/scripts/run_qksieve_fixed_target_length_diagnosis_20260803.sh"
HISTORY_TOKENS="${HISTORY_TOKENS:-32704 65472 98240 131008}"
EVAL_TOKENS="${EVAL_TOKENS:-16}"
REFERENCE_HISTORY_TOKENS="${REFERENCE_HISTORY_TOKENS:-131008}"
VARIANTS="${VARIANTS:-exact_qk_oracle_k1280,qksieve_keymse_requestlocal_fulltopk_k1280,qksieve_keymse_requestlocal_sampled_k1280_c32,qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280}"
VALUE_TAIL_ALPHA="${VALUE_TAIL_ALPHA:-0.5}"

mkdir -p "${RUN_ROOT}/logs"

launch_worker() {
  local gpu_ids="$1"
  local topic="$2"
  local seed="$3"
  env \
    RUN_ROOT="${RUN_ROOT}" \
    GPU_IDS="${gpu_ids}" \
    TOPIC="${topic}" \
    SEED="${seed}" \
    HISTORY_TOKENS="${HISTORY_TOKENS}" \
    REFERENCE_HISTORY_TOKENS="${REFERENCE_HISTORY_TOKENS}" \
    EVAL_TOKENS="${EVAL_TOKENS}" \
    VARIANTS="${VARIANTS}" \
    COLLECT_QK_PRODUCT_SPECTRUM=1 \
    QKSIEVE_VALUE_SKETCH_TAIL_ALPHA="${VALUE_TAIL_ALPHA}" \
    bash "${WORKER}" \
    >"${RUN_ROOT}/logs/worker_${topic}.log" 2>&1 &
  LAST_WORKER_PID="$!"
}

launch_worker 0,1 computer 20260833
pid_computer="${LAST_WORKER_PID}"
launch_worker 2,3 mixed_b 20260836
pid_mixed="${LAST_WORKER_PID}"
launch_worker 4,5 politics 20260834
pid_politics="${LAST_WORKER_PID}"

failed=0
for pid in "${pid_computer}" "${pid_mixed}" "${pid_politics}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi

touch "${RUN_ROOT}/ALL_COMPLETE"
