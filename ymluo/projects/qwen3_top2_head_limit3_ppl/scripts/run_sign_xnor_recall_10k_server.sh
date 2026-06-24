#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/sign_xnor_top2_recall_10k}"

TOTAL_TOKENS="${TOTAL_TOKENS:-10000}"
PREFILL_TOKENS="${PREFILL_TOKENS:-9000}"
EVAL_TOKENS="${EVAL_TOKENS:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
TOP_FRACTION="${TOP_FRACTION:-0.02}"
CANDIDATE_FRACTIONS="${CANDIDATE_FRACTIONS:-0.02,0.05,0.10,0.20}"
MAX_QUERY_SAMPLES="${MAX_QUERY_SAMPLES:-32}"
QUERY_STRIDE="${QUERY_STRIDE:-0}"

MAX_CHARS="${MAX_CHARS:-20000000}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
WRITE_PER_QUERY="${WRITE_PER_QUERY:-true}"

if [[ ! -f "${TEXT_PATH}" ]]; then
  echo "Eval text not found: ${TEXT_PATH}" >&2
  echo "Set TEXT_PATH=/path/to/a/long/text/file." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${PROJECT_DIR}/src/analyze_sign_xnor_top2_recall.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --total_tokens "${TOTAL_TOKENS}" \
  --prefill_tokens "${PREFILL_TOKENS}" \
  --eval_tokens "${EVAL_TOKENS}" \
  --chunk_size "${CHUNK_SIZE}" \
  --max_chars "${MAX_CHARS}" \
  --add_special_tokens false \
  --append_eos false \
  --require_total_tokens true \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --device_map "${DEVICE_MAP}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --top_fraction "${TOP_FRACTION}" \
  --candidate_fractions "${CANDIDATE_FRACTIONS}" \
  --max_query_samples "${MAX_QUERY_SAMPLES}" \
  --query_stride "${QUERY_STRIDE}" \
  --write_per_query "${WRITE_PER_QUERY}"
