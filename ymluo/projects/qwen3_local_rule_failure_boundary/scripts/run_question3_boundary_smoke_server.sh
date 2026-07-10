#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
MODEL_LABEL="${MODEL_LABEL:-qwen3_0p6b}"
OUT="${OUT:-$ROOT/outputs/question3_boundary_smoke_20260709}"

mkdir -p "$OUT"
cd "$ROOT"
rm -f "$OUT"/results.csv "$OUT"/candidate_scores.csv "$OUT"/attention_selectivity.csv \
  "$OUT"/summary_by_condition.csv "$OUT"/failure_boundary.csv "$OUT"/summary.md \
  "$OUT"/cases.jsonl "$OUT"/skipped_cases.jsonl "$OUT"/env.json

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PY" -u src/run_local_rule_failure_boundary.py \
  --model_name_or_path "$MODEL" \
  --model_label "$MODEL_LABEL" \
  --output_dir "$OUT" \
  --lengths "${LENGTHS:-512,1024,2048}" \
  --depths "${DEPTHS:-50}" \
  --seeds "${SEEDS:-0}" \
  --distractor_counts "${DISTRACTOR_COUNTS:-0,4}" \
  --distractor_similarities "${DISTRACTOR_SIMILARITIES:-low,high,conflict}" \
  --rule_gap_tokens "${RULE_GAP_TOKENS:-0,128}" \
  --chain_lengths "${CHAIN_LENGTHS:-1,2}" \
  --competitor_counts "${COMPETITOR_COUNTS:-0,1}" \
  --max_cases "${MAX_CASES:-24}" \
  --case_order shuffled \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPL:-sdpa}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-2048}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
  --compute_attention "${COMPUTE_ATTENTION:-true}" \
  --attention_mode "${ATTENTION_MODE:-all}" \
  2>&1 | tee "$OUT/run.log"
