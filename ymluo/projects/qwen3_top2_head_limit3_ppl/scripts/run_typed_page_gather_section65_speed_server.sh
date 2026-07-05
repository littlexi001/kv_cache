#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/bin/activate moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false

STAMP="${STAMP:-20260705_v1}"
OUT="${OUT:-outputs/typed_page_gather_section65_speed_${STAMP}}"
FULL_KV_LENS="${FULL_KV_LENS:-8192,16384,32768}"
BUDGETS="${BUDGETS:-256,512,1024,2048}"
STEPS="${STEPS:-1,16,64,256,1024}"
WARMUP="${WARMUP:-30}"
REPEAT="${REPEAT:-100}"

mkdir -p "$OUT" outputs/logs

python -u src/benchmark_typed_page_gather_section65.py \
  --output_dir "$OUT" \
  --full_kv_lens "$FULL_KV_LENS" \
  --budgets "$BUDGETS" \
  --steps "$STEPS" \
  --page_size "${PAGE_SIZE:-256}" \
  --sink_tokens "${SINK_TOKENS:-64}" \
  --recent_tokens "${RECENT_TOKENS:-256}" \
  --batch_count "${BATCH_COUNT:-1}" \
  --layer_count "${LAYER_COUNT:-32}" \
  --query_head_count "${QUERY_HEAD_COUNT:-32}" \
  --kv_head_count "${KV_HEAD_COUNT:-8}" \
  --head_dim "${HEAD_DIM:-128}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --seed "${SEED:-0}" \
  2>&1 | tee "$OUT/run.log"

echo "output $OUT"
