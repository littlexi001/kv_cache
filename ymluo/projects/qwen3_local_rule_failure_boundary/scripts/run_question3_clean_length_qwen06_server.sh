#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
MODEL_LABEL="${MODEL_LABEL:-qwen3_0p6b_clean_length}"
OUT="${OUT:-$PROJECT/outputs/clean_length_qwen06_20260710}"

cd "$PROJECT"
mkdir -p "$OUT"
rm -f "$OUT"/results.csv "$OUT"/candidate_scores.csv "$OUT"/attention_selectivity.csv \
  "$OUT"/summary_by_condition.csv "$OUT"/failure_boundary.csv "$OUT"/summary.md \
  "$OUT"/run.log "$OUT"/metadata.json

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}" nohup "$PY" -u src/run_local_rule_failure_boundary.py \
  --model_name_or_path "$MODEL" \
  --model_label "$MODEL_LABEL" \
  --output_dir "$OUT" \
  --lengths "${LENGTHS:-1024,4096,8192,16384,32768,65536,131072}" \
  --depths "${DEPTHS:-10,50,90}" \
  --seeds "${SEEDS:-0,1,2}" \
  --distractor_counts "${DISTRACTOR_COUNTS:-0}" \
  --distractor_similarities "${DISTRACTOR_SIMILARITIES:-low}" \
  --rule_gap_tokens "${RULE_GAP_TOKENS:-0}" \
  --chain_lengths "${CHAIN_LENGTHS:-1,4}" \
  --competitor_counts "${COMPETITOR_COUNTS:-0}" \
  --max_cases 0 \
  --case_order sequential \
  --dtype "${DTYPE:-float16}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-4096}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
  --compute_attention "${COMPUTE_ATTENTION:-true}" \
  --attention_mode "${ATTENTION_MODE:-all}" \
  --max_attention_prompt_tokens "${MAX_ATTENTION_PROMPT_TOKENS:-32768}" \
  > "$OUT"/run.log 2>&1 &

echo $! > "$OUT"/pid.txt
echo "launched clean length Qwen3-0.6B pid=$(cat "$OUT"/pid.txt)"
echo "log: $OUT/run.log"

