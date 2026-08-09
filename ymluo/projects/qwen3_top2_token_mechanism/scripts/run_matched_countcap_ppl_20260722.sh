#!/usr/bin/env bash
set -euo pipefail

GPU_LIST=${1:?'usage: GPU_LIST OUTPUT_ROOT [ORDER] [TOPIC] [HISTORY_TOKENS] [EVAL_TOKENS]'}
OUTPUT_ROOT=${2:?'output root is required'}
ORDER=${3:-full_first}
TOPIC=${4:-medicine}
HISTORY_TOKENS=${5:-128000}
EVAL_TOKENS=${6:-128}

PYTHON=${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}
HEAD_SRC=${HEAD_SRC:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src}
MECH_SRC=${MECH_SRC:-/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/src}
DATASET_CACHE=${DATASET_CACHE:-/home/fdong/ymluo/datasets/sklearn}
COUNTCAP_SCRIPT=${COUNTCAP_SCRIPT:-${MECH_SRC%/src}/scripts/run_countcap_ppl_20260722.sh}

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export PYTHONPATH="${HEAD_SRC}:${MECH_SRC}:${PYTHONPATH:-}"
export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_full() {
    mkdir -p "${OUTPUT_ROOT}/full"
    "${PYTHON}" -u "${HEAD_SRC}/run_full_cache_ppl_baseline_20260715.py" \
        --model_name_or_path "${MODEL}" \
        --output "${OUTPUT_ROOT}/full/result.json" \
        --topic "${TOPIC}" \
        --history_tokens "${HISTORY_TOKENS}" \
        --query_tokens "${EVAL_TOKENS}" \
        --eval_tokens "${EVAL_TOKENS}" \
        --window_stride_tokens $((HISTORY_TOKENS + EVAL_TOKENS + 1024)) \
        --prefill_chunk_tokens 2048 \
        --dataset_cache_dir "${DATASET_CACHE}" \
        --seed 20260714 \
        --dtype float16 \
        --device cuda \
        --device_map auto \
        > "${OUTPUT_ROOT}/full/run.log" 2>&1
}

run_countcap() {
    mkdir -p "${OUTPUT_ROOT}/countcap"
    bash "${COUNTCAP_SCRIPT}" \
        "${GPU_LIST}" "${TOPIC}" "${HISTORY_TOKENS}" "${EVAL_TOKENS}" \
        "${OUTPUT_ROOT}/countcap" \
        > "${OUTPUT_ROOT}/countcap/run.log" 2>&1
}

mkdir -p "${OUTPUT_ROOT}"
if [[ "${ORDER}" == "full_first" ]]; then
    run_full
    run_countcap
elif [[ "${ORDER}" == "countcap_first" ]]; then
    run_countcap
    run_full
else
    echo "ORDER must be full_first or countcap_first" >&2
    exit 2
fi
