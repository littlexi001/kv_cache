#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/qwen3_head_function_stability}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
RANKINGS="${RANKINGS:-${PROJECT_DIR}/outputs/category_link_distortion_full_v1_20260715/category_link_distortion_rankings.csv}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/top_head_category_nll_${STAMP}}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"$PY" -u src/run_top_head_category_nll_ablation.py \
  --model_name_or_path "$MODEL" \
  --rankings_csv "$RANKINGS" \
  --output_dir "$OUT_DIR" \
  --device cuda \
  --dtype "${DTYPE:-float16}" \
  --attn_implementation eager \
  --top_heads_per_category "${TOP_K:-3}" \
  --sample_limit "${SAMPLE_LIMIT:-0}" \
  2>&1 | tee "$OUT_DIR/run.log"

echo "[top-head-category-nll] done: $OUT_DIR"
