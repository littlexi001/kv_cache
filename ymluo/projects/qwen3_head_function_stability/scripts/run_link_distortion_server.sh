#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/qwen3_head_function_stability}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/category_link_distortion_${STAMP}}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"$PY" -u src/run_category_link_distortion.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT_DIR" \
  --device cuda \
  --dtype "${DTYPE:-float16}" \
  --attn_implementation eager \
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}" \
  --sample_limit "${SAMPLE_LIMIT:-0}" \
  --max_seq_length "${MAX_SEQ_LENGTH:-512}" \
  --min_history "${MIN_HISTORY:-16}" \
  --sink_tokens "${SINK_TOKENS:-4}" \
  --recent_window "${RECENT_WINDOW:-16}" \
  --manual_query_tail "${MANUAL_QUERY_TAIL:-1}" \
  --make_plots "${MAKE_PLOTS:-true}" \
  2>&1 | tee "$OUT_DIR/run.log"

echo "[head-link-distortion] done: $OUT_DIR"
