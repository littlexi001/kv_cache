#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../../.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/ymluo/models/Qwen3-0.6B}"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/top1_category_ablation}"
PYTHON_EXE="${PYTHON_EXE:-python}"

"${PYTHON_EXE}" "${PROJECT_DIR}/src/run_top1_category_ablation.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_samples "${MAX_SAMPLES:-8}" \
  --max_context_chars "${MAX_CONTEXT_CHARS:-24000}" \
  --top_ratio "${TOP_RATIO:-0.01}" \
  --modes "${MODES:-full_attention,top1_all,answer_only,front_only,end_only,other_only}" \
  --svd_max_vectors "${SVD_MAX_VECTORS:-4096}" \
  --svd_top_k "${SVD_TOP_K:-128}" \
  --dump_top_tokens "${DUMP_TOP_TOKENS:-true}" \
  --dtype "${DTYPE:-float16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --trust_remote_code "${TRUST_REMOTE_CODE:-true}"
