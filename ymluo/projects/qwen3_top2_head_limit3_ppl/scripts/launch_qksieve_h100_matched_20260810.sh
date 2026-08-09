#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:?set ROOT to the portable QKSieve workspace}"
PYTHON="${PYTHON:?set PYTHON to the experiment interpreter}"
MODEL="${MODEL:-${ROOT}/models/Yarn-Llama-2-7b-128k}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_h100_matched_v1}"
ATTENTION_RUNNER="${ROOT}/src/benchmark_qksieve_fier_mha_speed_20260808.py"
DECODE_LAUNCHER="${ROOT}/scripts/launch_qksieve_mha_real_decode_20260809.sh"
PERSISTENT_CASE="${ROOT}/scripts/run_qksieve_persistent_kv_case_20260810.sh"
SUMMARIZER="${ROOT}/src/summarize_qksieve_h100_20260810.py"
SEEDS="${SEEDS:-20260810,20260811,20260812}"
GPU_64K="${GPU_64K:-0}"
GPU_128K="${GPU_128K:-0,1}"

export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT=512
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.5
export QKSIEVE_TRUST_REMOTE_CODE=0

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/attention"
touch "${RUN_ROOT}/RUNNING"
rm -f "${RUN_ROOT}/ALL_COMPLETE" "${RUN_ROOT}/FAILED"

fail() {
  touch "${RUN_ROOT}/FAILED"
  rm -f "${RUN_ROOT}/RUNNING"
  exit 1
}

required_devices="${GPU_64K},${GPU_128K}"
IFS=',' read -r -a device_list <<<"${required_devices}"
for device in $(printf '%s\n' "${device_list[@]}" | sort -u); do
  gpu_line="$(nvidia-smi -i "${device}" \
    --query-gpu=name,memory.total --format=csv,noheader,nounits)" || fail
  gpu_name="${gpu_line%,*}"
  gpu_memory="${gpu_line##*, }"
  if [[ "${gpu_name}" != *H100* ]] || (( gpu_memory < 75000 )); then
    echo "GPU ${device} is not an >=80GB H100: ${gpu_line}" \
      >"${RUN_ROOT}/logs/hardware_error.log"
    fail
  fi
done

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "missing MHA model: ${MODEL}" >"${RUN_ROOT}/logs/model_error.log"
  fail
fi

IFS=',' read -r -a seed_list <<<"${SEEDS}"
for seed in "${seed_list[@]}"; do
  attention_out="${RUN_ROOT}/attention/seed${seed}.json"
  if [[ ! -s "${attention_out}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_64K%%,*}" "${PYTHON}" -u \
      "${ATTENTION_RUNNER}" \
      --lengths 65536,131072 \
      --max_sample_count 512 \
      --qksieve_split_count 8 \
      --fier_split_count 8 \
      --value_tail_alpha 0.5 \
      --warmup 20 --iterations 80 --seed "${seed}" \
      --output "${attention_out}.tmp" \
      >"${RUN_ROOT}/logs/attention_seed${seed}.log" 2>&1 || fail
    mv "${attention_out}.tmp" "${attention_out}"
  fi
done

run_decode_pair() {
  local length="$1" devices="$2" seed="$3" method
  for method in full qksieve_valuesketch_top1280; do
    CUDA_VISIBLE_DEVICES="${devices}" \
    GPU_TAG="${devices//,/-}" \
    ROOT="${ROOT}" RUN_ROOT="${RUN_ROOT}/decode" MODEL="${MODEL}" \
    HISTORY_TOKENS="${length}" GENERATION_STEPS=256 STEADY_START=32 \
    GLOBAL_MAX_POSITION=262144 MAX_MEMORY_PER_GPU_GIB=76 \
    SEED="${seed}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    QKSIEVE_PRELOAD_EXTENSIONS=1 \
    bash "${DECODE_LAUNCHER}" "${method}" || return 1
  done
}

run_persistent_pair() {
  local length="$1" devices="$2" seed="$3" method
  for method in full qksieve_robust; do
    CUDA_VISIBLE_DEVICES="${devices}" \
    GPU_TAG="${devices//,/-}" \
    ROOT="${ROOT}" RUN_ROOT="${RUN_ROOT}/persistent" MODEL="${MODEL}" \
    HISTORY_TOKENS="${length}" METHOD="${method}" \
    BRANCH_COUNT=4 BRANCH_STEPS=64 APPEND_STEPS=128 \
    MAX_MEMORY_PER_GPU_GIB=76 SEED="${seed}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    bash "${PERSISTENT_CASE}" || return 1
  done
}

for seed in "${seed_list[@]}"; do
  run_decode_pair 65536 "${GPU_64K}" "${seed}" || fail
  run_decode_pair 131072 "${GPU_128K}" "${seed}" || fail
  run_persistent_pair 65536 "${GPU_64K}" "${seed}" || fail
  run_persistent_pair 131072 "${GPU_128K}" "${seed}" || fail
done

"${PYTHON}" "${SUMMARIZER}" \
  --run_root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/summary.json" \
  --expected_seeds "${#seed_list[@]}" \
  >"${RUN_ROOT}/logs/summary.log" 2>&1 || fail

rm -f "${RUN_ROOT}/RUNNING" "${RUN_ROOT}/FAILED"
touch "${RUN_ROOT}/ALL_COMPLETE"
