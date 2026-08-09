#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260802_qksieve_errorbound_matrix_8gpu}"
TRACES="${ROOT}/results/20260728_qksieve_all_layer_bits_qwen3_32k/traces/qwen3_4b_sports32k_all_layers.pt,${ROOT}/results/20260728_qksieve_all_layer_bits_qwen3_32k/traces/qwen3_4b_medicine32k_all_layers.pt"
SPECS=(
  "fixed400_b80|per_head|sample_ls|qk|value|4|unit|0"
  "fixed210_b80|per_head|sample_ls|qk|value|4|unit|0"
  "auto80|per_head|sample_ls|qk|value|4|unit|0"
  "fixed400_b80|gqa_shared_normalized_max|sample_ls|qk|value|4|unit|0"
  "fixed210_b80|gqa_shared_normalized_max|sample_ls|conditional_value_bound|value|4|spectral|1"
  "auto80|gqa_shared_normalized_max|sample_ls|conditional_value_bound|value|4|spectral|1"
  "auto80|gqa_shared_output_bound|sample_ls|conditional_value_bound|value|4|spectral|1"
  "fixed4421_b240|gqa_shared_normalized_max|sample_ls|conditional_value_bound|value|4|spectral|1"
)

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"
rm -f "${RUN_ROOT}/ALL_COMPLETE" "${RUN_ROOT}/FAILED"

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  IFS='|' read -r profile policy calibration priority leverage_space leverage_bits output_gain leverage_lambda <<<"${SPECS[$gpu]}"
  tag="${profile}_${policy}_${calibration}_${priority}_${leverage_space}_b${leverage_bits}_${output_gain}_l${leverage_lambda}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_layerwise_rate_distortion_20260802.py \
    --traces "${TRACES}" \
    --model_dir "${MODEL}" \
    --output "${RUN_ROOT}/${tag}.json" \
    --layers 0,8,16,24,31 \
    --sample_stride 32 \
    --calibration_steps 8 \
    --max_test_steps 8 \
    --fraction 0.04 \
    --candidate_policy "${policy}" \
    --tail_score_calibration "${calibration}" \
    --tail_score_sample_count 256 \
    --selection_priority "${priority}" \
    --value_leverage_bits "${leverage_bits}" \
    --value_leverage_space "${leverage_space}" \
    --output_group_gain "${output_gain}" \
    --value_leverage_lambda "${leverage_lambda}" \
    --key_profiles "${profile}" \
    --moment_profiles 8x1024x8 \
    --linear_group_blocks 0 \
    --linear_fit_stride 1 \
    --rate_weights 0 \
    --maximum_passes 1 \
    >"${RUN_ROOT}/logs/${tag}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
