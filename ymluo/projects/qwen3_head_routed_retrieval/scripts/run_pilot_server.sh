#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
TEXT="${TEXT:-${PROJECT_DIR}/../qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/pilot_war4k_${STAMP}}"

mkdir -p "$OUT_DIR"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PY" -u "$PROJECT_DIR/src/run_head_retriever_imitation.py" \
  --model_name_or_path "$MODEL" \
  --text_path "$TEXT" \
  --output_dir "$OUT_DIR" \
  --max_chars "${MAX_CHARS:-500000}" \
  --prefill_tokens "${PREFILL_TOKENS:-4096}" \
  --train_queries "${TRAIN_QUERIES:-128}" \
  --test_queries "${TEST_QUERIES:-128}" \
  --chunk_size "${CHUNK_SIZE:-16}" \
  --ratio "${RATIO:-0.02}" \
  --query_window "${QUERY_WINDOW:-32}" \
  --block_size "${BLOCK_SIZE:-32}" \
  --repeat_max_n "${REPEAT_MAX_N:-4}" \
  --hybrid_position_fraction "${HYBRID_POSITION_FRACTION:-0.5}" \
  --sink_tokens "${SINK_TOKENS:-4}" \
  --recent_tokens "${RECENT_TOKENS:-256}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation eager \
  --log_every "${LOG_EVERY:-1}" \
  2>&1 | tee "$OUT_DIR/run.log"

echo "[head-routed-retrieval] done: $OUT_DIR"
