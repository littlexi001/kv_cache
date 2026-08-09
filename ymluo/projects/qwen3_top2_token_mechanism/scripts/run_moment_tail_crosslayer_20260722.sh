#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_token_mechanism
TRACE_ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260717_delta_qkv_traces_32k_s16
OUT_ROOT=${ROOT}/artifacts/20260722_microblock_runtime/tail_gate_crosslayer
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
SCRIPT=${ROOT}/src/analyze_block_moment_tail_compensation.py

mkdir -p "${OUT_ROOT}"

run_specs() {
    local gpu=$1
    shift
    for spec in "$@"; do
        local topic=${spec%%:*}
        local layer=${spec##*:}
        CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON}" "${SCRIPT}" \
            --trace_path "${TRACE_ROOT}/${topic}.pt" \
            --output_path "${OUT_ROOT}/${topic}_layer${layer}.json" \
            --device cuda \
            --layer "${layer}" \
            --rank 48 \
            --block_sizes 256 \
            --train_steps 4 \
            --test_start_step 8 \
            --test_steps 8 \
            --query_shrinkage 0.5 \
            --key_sample_stride 32 \
            --affine_sample_stride 400 \
            --candidate_fraction 0.06 \
            --top_fraction 0.02 \
            >"${OUT_ROOT}/${topic}_layer${layer}.log" 2>&1
    done
}

run_specs 0 sports:0 sports:31 &
run_specs 1 medicine:0 medicine:31 &
run_specs 2 sports:8 &
run_specs 3 medicine:8 &
run_specs 4 sports:16 &
run_specs 5 medicine:16 &
run_specs 6 sports:24 &
run_specs 7 medicine:24 &
wait
