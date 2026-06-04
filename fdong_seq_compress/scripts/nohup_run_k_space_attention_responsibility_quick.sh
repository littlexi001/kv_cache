#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/biomed_long_range_facts_hard_compact.txt}"
DEVICE="${DEVICE:-mps}"
DTYPE="${DTYPE:-float16}"
MAX_TOKENS="${MAX_TOKENS:-3000}"
QUERY_START="${QUERY_START:-2500}"
QUERY_STRIDE="${QUERY_STRIDE:-16}"
MAX_QUERIES="${MAX_QUERIES:-32}"
LAYERS="${LAYERS:-0,13,27}"
Q_HEADS="${Q_HEADS:-0,4,8,12}"
METHODS="${METHODS:-local,q_l2_nearest,q_dot_nearest,cluster_l2_topn,cluster_dot_topn}"
LOCAL_WINDOW="${LOCAL_WINDOW:-128}"
MAX_CANDIDATES="${MAX_CANDIDATES:-256}"
NUM_CLUSTERS="${NUM_CLUSTERS:-20}"
TOP_CLUSTERS="${TOP_CLUSTERS:-2}"
KMEANS_STEPS="${KMEANS_STEPS:-5}"
TOP_ATTENTION_KS="${TOP_ATTENTION_KS:-1,5,10}"
RANDOM_SEED="${RANDOM_SEED:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-fdong_seq_compress/outputs/k_space_attention_responsibility_${STAMP}}"
LOG_DIR="${LOG_DIR:-fdong_seq_compress/logs}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "output_dir=${OUTPUT_DIR}"
echo "model=${MODEL_PATH} text=${TEXT_PATH}"
echo "device=${DEVICE} dtype=${DTYPE}"
echo "max_tokens=${MAX_TOKENS} query_start=${QUERY_START} query_stride=${QUERY_STRIDE} max_queries=${MAX_QUERIES}"
echo "layers=${LAYERS} q_heads=${Q_HEADS}"
echo "methods=${METHODS}"
echo "clusters=${NUM_CLUSTERS} top_clusters=${TOP_CLUSTERS} max_candidates=${MAX_CANDIDATES}"

python3 fdong_seq_compress/src/run_k_space_attention_responsibility.py \
  --model-path "${MODEL_PATH}" \
  --text-path "${TEXT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --attn-implementation eager \
  --max-tokens "${MAX_TOKENS}" \
  --query-start "${QUERY_START}" \
  --query-stride "${QUERY_STRIDE}" \
  --max-queries "${MAX_QUERIES}" \
  --layers "${LAYERS}" \
  --q-heads "${Q_HEADS}" \
  --methods "${METHODS}" \
  --local-window "${LOCAL_WINDOW}" \
  --max-candidates "${MAX_CANDIDATES}" \
  --num-clusters "${NUM_CLUSTERS}" \
  --top-clusters "${TOP_CLUSTERS}" \
  --kmeans-steps "${KMEANS_STEPS}" \
  --top-attention-ks "${TOP_ATTENTION_KS}" \
  --random-seed "${RANDOM_SEED}"
