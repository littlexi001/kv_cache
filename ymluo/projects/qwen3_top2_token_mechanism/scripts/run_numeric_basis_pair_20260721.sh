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

(
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${ROOT}/src/analyze_basis_sampling_frontier.py" \
    --trace_paths "${SPORTS}" "${MEDICINE}" "${QWEN128}" \
    --output_path "${ARTIFACTS}/basis_sampling_frontier_all.json" \
    --device cuda \
    --layers 16 \
    --max_steps 16 \
    --rank 64 \
    --candidate_fractions 0.04,0.05,0.06,0.08 \
    --methods stride32,stride16,stride8,full,blockmax32,hybrid64,normmix05,normmix10 \
    > "${ARTIFACTS}/basis_sampling_frontier_all.log" 2>&1
) &
basis_pid=$!

(
  CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${ROOT}/src/analyze_qk_metric_lowrank.py" \
    --trace_paths "${SPORTS}" "${MEDICINE}" "${QWEN128}" \
    --output_path "${ARTIFACTS}/qk_metric_lowrank_all.json" \
    --device cuda \
    --layers 16 \
    --rank 64 \
    --train_steps 8 \
    --test_steps 8 \
    --key_sample_stride 32 \
    --query_shrinkages 0.25,0.5,0.75,0.9,0.97,1.0 \
    --candidate_fractions 0.04,0.05,0.06,0.08 \
    > "${ARTIFACTS}/qk_metric_lowrank_all.log" 2>&1
) &
metric_pid=$!

status=0
if ! wait "${basis_pid}"; then status=1; fi
if ! wait "${metric_pid}"; then status=1; fi
exit "${status}"

