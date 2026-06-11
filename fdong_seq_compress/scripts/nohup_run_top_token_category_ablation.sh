#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
DATA_PATH="${DATA_PATH:-ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl}"
SAMPLE_ID="${SAMPLE_ID:-niah_len2000_depth25}"
DEVICE="${DEVICE:-mps}"
DTYPE="${DTYPE:-float16}"
LAYERS="${LAYERS:-all}"
Q_HEADS="${Q_HEADS:-all}"
QUERY_LAST_TOKENS="${QUERY_LAST_TOKENS:-10}"
SCORE_RATIOS="${SCORE_RATIOS:-0.01,0.02,0.04}"
POSITION_RATIOS="${POSITION_RATIOS:-0.01,0.05,0.10}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-fdong_seq_compress/outputs/top_token_category_ablation_${TIMESTAMP}}"
LOG_DIR="fdong_seq_compress/logs"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/top_token_category_ablation_${TIMESTAMP}.log}"
PID_PATH="${LOG_PATH%.log}.pid"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

nohup python3 -u fdong_seq_compress/src/run_top_token_category_ablation.py \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --sample-id "${SAMPLE_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --layers "${LAYERS}" \
  --q-heads "${Q_HEADS}" \
  --query-last-tokens "${QUERY_LAST_TOKENS}" \
  --score-ratios "${SCORE_RATIOS}" \
  --position-ratios "${POSITION_RATIOS}" \
  >"${LOG_PATH}" 2>&1 &

PID=$!
echo "${PID}" >"${PID_PATH}"

echo "Started top-token category profiling and ablation"
echo "PID: ${PID}"
echo "Log: ${LOG_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Follow: tail -f ${LOG_PATH}"
