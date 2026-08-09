#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
RUN_ROOT="${RUN_ROOT:-${BASE}/outputs/20260731_qwen4_prerope_exact_diagnosis_8gpu}"
LENGTHS="${LENGTHS:-32768,65536}"
VARIANTS="${VARIANTS:-full_rope,rope_top2,local_global_postscore,dual_max_postscore}"
SEED_BASE="${SEED_BASE:-264}"

mkdir -p "${RUN_ROOT}/logs"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${BASE}"

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  seed=$((SEED_BASE + gpu))
  output="${RUN_ROOT}/seed${seed}"
  mkdir -p "${output}"
  if [[ -f "${output}/done.txt" ]]; then
    echo "SKIP completed seed ${seed}"
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON}" -u src/run_local_global_rope_probe_8b.py \
      --model-name-or-path "${MODEL}" \
      --output-dir "${output}" \
      --lengths "${LENGTHS}" \
      --seed-start "${seed}" \
      --num-seeds 1 \
      --variants "${VARIANTS}" \
      --ratio 0.02 \
      --local-window 128 \
      --sink-tokens 16 \
      --prefill-chunk-size 128 \
      --dtype float16 \
      --attn-implementation sdpa \
      --original-max-position-embeddings 262144 \
      --global-max-position 262144 \
      >"${output}/run.log" 2>&1
  ) >"${RUN_ROOT}/logs/seed${seed}.log" 2>&1 &
  pids+=("$!")
  echo "seed ${seed}: GPU ${gpu}, PID $!"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
