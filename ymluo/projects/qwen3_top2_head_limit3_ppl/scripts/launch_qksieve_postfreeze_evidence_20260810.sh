#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Yarn-Llama-2-7b-128k}"
DECODE_LAUNCHER="${ROOT}/scripts/launch_qksieve_mha_real_decode_20260809.sh"
PERSISTENT_CASE="${ROOT}/scripts/run_qksieve_persistent_kv_case_20260810.sh"
ATTENTION_RUNNER="${ROOT}/src/benchmark_qksieve_fier_mha_speed_20260808.py"
SUMMARY_RUNNER="${ROOT}/src/summarize_qksieve_persistent_kv_20260810.py"
EVIDENCE_ROOT="${ROOT}/results/20260810_qksieve_postfreeze_evidence_v1"

export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.5
mkdir -p "${EVIDENCE_ROOT}/logs"
touch "${EVIDENCE_ROOT}/RUNNING"
rm -f "${EVIDENCE_ROOT}/ALL_COMPLETE" "${EVIDENCE_ROOT}/FAILED"

status=0

# Clean 128K run after all CUDA extensions have been compiled and cached.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RUN_ROOT="${EVIDENCE_ROOT}/decode_clean" \
MODEL="${MODEL}" \
HISTORY_TOKENS=131072 \
GENERATION_STEPS=64 \
STEADY_START=16 \
GPU_TAG=0-1-2-3-4-5-6-7 \
bash "${DECODE_LAUNCHER}" qksieve_valuesketch_top1280 \
  >"${EVIDENCE_ROOT}/logs/decode_clean_128k.log" 2>&1 || status=1

run_persistent_pair() {
  local devices="$1" length="$2"
  local method
  for method in full qksieve_robust; do
    CUDA_VISIBLE_DEVICES="${devices}" \
    GPU_TAG="${devices//,/-}" \
    ROOT="${ROOT}" \
    RUN_ROOT="${EVIDENCE_ROOT}/persistent" \
    MODEL="${MODEL}" \
    HISTORY_TOKENS="${length}" \
    METHOD="${method}" \
    BRANCH_COUNT=4 \
    BRANCH_STEPS=32 \
    APPEND_STEPS=128 \
    bash "${PERSISTENT_CASE}" || return 1
  done
}

run_attention_8k() {
  local seed
  mkdir -p "${EVIDENCE_ROOT}/attention_8k_sequential"
  for seed in 20260809 20260810 20260811; do
    CUDA_VISIBLE_DEVICES=5 "${PYTHON}" -u "${ATTENTION_RUNNER}" \
      --lengths 8192 \
      --warmup 20 \
      --iterations 80 \
      --value_tail_alpha 0.5 \
      --seed "${seed}" \
      --output "${EVIDENCE_ROOT}/attention_8k_sequential/seed${seed}.json" \
      >"${EVIDENCE_ROOT}/logs/attention_8k_seed${seed}.log" 2>&1 || return 1
  done
}

if [[ ${status} -eq 0 ]]; then
  run_persistent_pair 0,1 32768 & p0=$!
  run_persistent_pair 2,3,4 65536 & p1=$!
  run_attention_8k & p2=$!
  for pid in "${p0}" "${p1}" "${p2}"; do
    wait "${pid}" || status=1
  done
fi

if [[ ${status} -eq 0 ]]; then
  "${PYTHON}" "${SUMMARY_RUNNER}" \
    --run_root "${EVIDENCE_ROOT}/persistent" \
    --output "${EVIDENCE_ROOT}/persistent/summary.json" \
    >"${EVIDENCE_ROOT}/logs/persistent_summary.log" 2>&1 || status=1
fi

rm -f "${EVIDENCE_ROOT}/RUNNING"
if [[ ${status} -eq 0 ]]; then
  touch "${EVIDENCE_ROOT}/ALL_COMPLETE"
else
  touch "${EVIDENCE_ROOT}/FAILED"
fi
exit "${status}"
