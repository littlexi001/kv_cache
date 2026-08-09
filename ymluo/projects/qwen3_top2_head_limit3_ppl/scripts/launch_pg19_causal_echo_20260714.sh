#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
PG19_PARQUET="${PG19_PARQUET:-/home/fdong/ymluo/datasets/pg19/test.parquet}"
BOOK_INDICES="${BOOK_INDICES:-0 1 2 3 4 5}"
RUN_TAG="${RUN_TAG:-20260714_pg19_echo}"

cd "${PROJECT_DIR}"
mkdir -p outputs
for book_index in ${BOOK_INDICES}; do
    gpu_index=$((book_index % 8))
    output_dir="outputs/${RUN_TAG}_book${book_index}"
    log_path="outputs/${RUN_TAG}_book${book_index}.log"
    rm -rf "${output_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_index}" nohup "${PYTHON_BIN}" \
        src/run_pg19_causal_echo_ppl_20260714.py \
        --model_name_or_path "${MODEL_PATH}" \
        --pg19_parquet "${PG19_PARQUET}" \
        --output_dir "${output_dir}" \
        --book_indices "${book_index}" \
        --history_tokens 32000 \
        --query_tokens 256 \
        --eval_tokens 256 \
        --budget_tokens 2048 \
        --recent_tokens 1536 \
        --echo_match_tokens 8 \
        --echo_confirmation_tokens 8 \
        --echo_stability_matches 3 \
        --echo_refresh_tokens 1 \
        --dtype float16 \
        --device cuda \
        --device_map auto \
        --attn_implementation sdpa \
        >"${log_path}" 2>&1 &
    echo "book=${book_index} gpu=${gpu_index} pid=$! log=${log_path}"
done
