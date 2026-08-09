#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_robust_alpha05_freeze_validation_v1}"
MODEL="${MODEL:-${ROOT}/models/Yarn-Llama-2-7b-128k}"
DECODE_LAUNCHER="${ROOT}/scripts/launch_qksieve_mha_real_decode_20260809.sh"
ATTENTION_RUNNER="${ROOT}/src/benchmark_qksieve_fier_mha_speed_20260808.py"

export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.5
export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RUN_ROOT}/attention/logs" "${RUN_ROOT}/launcher_logs"

run_attention() {
  local gpu="$1" seed="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ATTENTION_RUNNER}" \
    --lengths 8192,16384,32768,65536,131072 \
    --warmup 10 \
    --iterations 40 \
    --value_tail_alpha 0.5 \
    --seed "${seed}" \
    --output "${RUN_ROOT}/attention/seed${seed}.json" \
    >"${RUN_ROOT}/attention/logs/seed${seed}.log" 2>&1
}

run_decode() {
  local devices="$1" length="$2"
  local tag=${devices//,/-}
  CUDA_VISIBLE_DEVICES="${devices}" \
  QKSIEVE_TRUST_REMOTE_CODE=0 \
  QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.5 \
  RUN_ROOT="${RUN_ROOT}/decode" \
  MODEL="${MODEL}" \
  HISTORY_TOKENS="${length}" \
  GENERATION_STEPS=64 \
  STEADY_START=16 \
  GPU_TAG="${tag}" \
  bash "${DECODE_LAUNCHER}" qksieve_valuesketch_top1280 \
    >"${RUN_ROOT}/launcher_logs/n${length}_robust_gpu${tag}.log" 2>&1
}

status=0
run_decode 0,1 32768 & p0=$!
run_decode 2,3,4 65536 & p1=$!
run_attention 5 20260809 & p2=$!
run_attention 6 20260810 & p3=$!
run_attention 7 20260811 & p4=$!
for pid in "${p0}" "${p1}" "${p2}" "${p3}" "${p4}"; do
  wait "${pid}" || status=1
done

if [[ ${status} -eq 0 ]]; then
  run_decode 0,1,2,3,4,5,6,7 131072 || status=1
fi

if [[ ${status} -eq 0 ]]; then
  rm -f "${RUN_ROOT}/FAILED"
  touch "${RUN_ROOT}/ALL_COMPLETE"
else
  rm -f "${RUN_ROOT}/ALL_COMPLETE"
  touch "${RUN_ROOT}/FAILED"
fi
exit "${status}"
