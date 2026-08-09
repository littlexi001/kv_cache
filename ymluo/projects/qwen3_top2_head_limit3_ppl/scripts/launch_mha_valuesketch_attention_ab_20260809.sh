#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260809_mha_valuesketch_attention_ab_v1}"
SRC_ROOT="${ROOT}/src"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}/logs"

run_seed() {
  local gpu="$1"
  local seed="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    "${SRC_ROOT}/benchmark_qksieve_fier_mha_speed_20260808.py" \
    --lengths 8192,16384,32768,65536,131072 \
    --warmup 10 \
    --iterations 40 \
    --value_tail_alpha 0.5 \
    --seed "${seed}" \
    --output "${RUN_ROOT}/seed${seed}.json" \
    >"${RUN_ROOT}/logs/seed${seed}.log" 2>&1
}

run_seed 5 20260809 & p1=$!
run_seed 6 20260810 & p2=$!
run_seed 7 20260811 & p3=$!
wait "${p1}" "${p2}" "${p3}"

echo "ALL_COMPLETE"
