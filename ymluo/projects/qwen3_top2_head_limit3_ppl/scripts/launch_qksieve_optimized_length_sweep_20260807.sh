#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/qksieve_optimized_length_sweep_20260807}"
EVAL_TOKENS="${EVAL_TOKENS:-64}"
REPEATS="${REPEATS:-5}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"

mkdir -p "${RUN_ROOT}/logs"
{
  echo "purpose=frozen QKSieve full-optimization length and generation sweep"
  echo "eval_tokens=${EVAL_TOKENS}"
  echo "repeats=${REPEATS}"
  echo "QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1"
  echo "QKSIEVE_BATCH_QMSE_ALLOCATION=1"
  echo "QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=${QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE:-1}"
  echo "QKSIEVE_TILED_VALUE_ATTENTION=${QKSIEVE_TILED_VALUE_ATTENTION:-0}"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/qksieve_valuesketch_cuda_20260801.py" \
    "${ROOT}/src/profile_qksieve_realmodel_index_build_20260807.py" \
    "${ROOT}/scripts/launch_qksieve_optimized_profile_20260807.sh"
} >"${RUN_ROOT}/manifest.txt"

run_one() {
  local history_tokens="$1"
  local repeat="$2"
  local gpu="$3"
  local output="${RUN_ROOT}/n${history_tokens}/r${repeat}"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    return
  fi
  mkdir -p "${output}"
  RUN_ROOT="${output}" \
  GPU="${gpu}" \
  HISTORY_TOKENS="${history_tokens}" \
  EVAL_TOKENS="${EVAL_TOKENS}" \
    bash "${ROOT}/scripts/launch_qksieve_optimized_profile_20260807.sh"
}

run_range() {
  local history_tokens="$1"
  local gpu="$2"
  local first="$3"
  local last="$4"
  local repeat
  for ((repeat=first; repeat<=last; repeat++)); do
    run_one "${history_tokens}" "${repeat}" "${gpu}"
  done
}

run_range 8192 0 1 "${REPEATS}" \
  >"${RUN_ROOT}/logs/n8192.log" 2>&1 &
pid_8k=$!
run_range 16384 1 1 "${REPEATS}" \
  >"${RUN_ROOT}/logs/n16384.log" 2>&1 &
pid_16k=$!
run_range 32768 2 1 "${REPEATS}" \
  >"${RUN_ROOT}/logs/n32768.log" 2>&1 &
pid_32k=$!
run_range 65536 3 1 "${REPEATS}" \
  >"${RUN_ROOT}/logs/n65536.log" 2>&1 &
pid_64k=$!

split=$(( (REPEATS + 1) / 2 ))
run_range 131072 "4,5" 1 "${split}" \
  >"${RUN_ROOT}/logs/n131072_a.log" 2>&1 &
pid_128k_a=$!
if (( split < REPEATS )); then
  run_range 131072 "6,7" "$((split + 1))" "${REPEATS}" \
    >"${RUN_ROOT}/logs/n131072_b.log" 2>&1 &
  pid_128k_b=$!
else
  pid_128k_b=""
fi

status=0
for pid in "${pid_8k}" "${pid_16k}" "${pid_32k}" "${pid_64k}" \
  "${pid_128k_a}" ${pid_128k_b:+"${pid_128k_b}"}; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" "${ROOT}/src/summarize_qksieve_optimized_length_sweep_20260807.py" \
  "${RUN_ROOT}" >"${RUN_ROOT}/logs/summarize.log" 2>&1 || status=1

if (( status == 0 )); then
  touch "${RUN_ROOT}/ALL_COMPLETE"
fi
exit "${status}"
