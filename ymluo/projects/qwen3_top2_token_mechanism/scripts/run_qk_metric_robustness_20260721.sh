#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_token_mechanism
HEAD=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
ARTIFACTS=${ROOT}/artifacts/20260721_numeric_pruning_frontier
SPORTS=${HEAD}/results/20260717_delta_qkv_traces_32k_s16/sports.pt
MEDICINE=${HEAD}/results/20260717_delta_qkv_traces_32k_s16/medicine.pt
QWEN128=${ROOT}/artifacts/20260720_oneshot_combinations/medicine_128k_layer16_s16.pt
export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=${ROOT}/src:${HEAD}/src:${PYTHONPATH:-}
mkdir -p "${ARTIFACTS}"

run_probe() {
  local gpu=$1
  local output=$2
  local layers=$3
  local train_steps=$4
  shift 4
  CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON}" "${ROOT}/src/analyze_qk_metric_lowrank.py" \
    --trace_paths "$@" \
    --output_path "${ARTIFACTS}/${output}.json" \
    --device cuda \
    --layers "${layers}" \
    --rank 64 \
    --train_steps "${train_steps}" \
    --test_start_step 8 \
    --test_steps 8 \
    --key_sample_stride 32 \
    --query_shrinkages 0.5,0.75,1.0 \
    --candidate_fractions 0.04,0.05,0.06,0.08 \
    > "${ARTIFACTS}/${output}.log" 2>&1
}

pids=()
run_probe 0 qkmetric_llama_alllayers_train8 0,8,16,24,31 8 "${SPORTS}" "${MEDICINE}" & pids+=("$!")
run_probe 1 qkmetric_qwen128_train2 16 2 "${QWEN128}" & pids+=("$!")
run_probe 2 qkmetric_qwen128_train4 16 4 "${QWEN128}" & pids+=("$!")
run_probe 3 qkmetric_qwen128_train8 16 8 "${QWEN128}" & pids+=("$!")
run_probe 4 qkmetric_llama_layer16_train2 16 2 "${SPORTS}" "${MEDICINE}" & pids+=("$!")
run_probe 5 qkmetric_llama_layer16_train4 16 4 "${SPORTS}" "${MEDICINE}" & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
exit "${status}"

