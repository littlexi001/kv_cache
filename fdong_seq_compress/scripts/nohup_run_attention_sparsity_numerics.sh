#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt}"
DEVICE="${DEVICE:-mps}"
DTYPE="${DTYPE:-float16}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
LAYERS="${LAYERS:-0,7,14,21,27}"
Q_HEADS="${Q_HEADS:-all}"
QUERY_WINDOW="${QUERY_WINDOW:-512}"
QUERY_STRIDE="${QUERY_STRIDE:-32}"
RATIOS="${RATIOS:-0.001,0.005,0.01,0.02,0.04,0.06,0.1,0.2,0.5}"
SVD_RANKS="${SVD_RANKS:-1,4,8,16}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-fdong_seq_compress/outputs/attention_sparsity_numerics_${TIMESTAMP}}"
LOG_DIR="fdong_seq_compress/logs"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/attention_sparsity_numerics_${TIMESTAMP}.log}"
PID_PATH="${LOG_PATH%.log}.pid"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

nohup python3 -u fdong_seq_compress/src/run_attention_sparsity_numerics.py \
  --model-path "${MODEL_PATH}" \
  --text-path "${TEXT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-tokens "${MAX_TOKENS}" \
  --layers "${LAYERS}" \
  --q-heads "${Q_HEADS}" \
  --query-window "${QUERY_WINDOW}" \
  --query-stride "${QUERY_STRIDE}" \
  --ratios "${RATIOS}" \
  --svd-ranks "${SVD_RANKS}" \
  >"${LOG_PATH}" 2>&1 &

PID=$!
echo "${PID}" >"${PID_PATH}"

echo "Started attention sparsity numerics experiment"
echo "PID: ${PID}"
echo "Log: ${LOG_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Follow: tail -f ${LOG_PATH}"
