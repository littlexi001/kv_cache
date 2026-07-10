#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
MODEL_LABEL="${MODEL_LABEL:-qwen3_0p6b_clean_interference}"
OUT="${OUT:-$PROJECT/outputs/clean_interference_qwen06_20260710}"

cd "$PROJECT"
mkdir -p "$OUT"
rm -f "$OUT"/results.csv "$OUT"/candidate_scores.csv "$OUT"/attention_selectivity.csv \
  "$OUT"/summary_by_condition.csv "$OUT"/failure_boundary.csv "$OUT"/summary.md \
  "$OUT"/run.log "$OUT"/metadata.json

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}" nohup "$PY" -u src/run_local_rule_failure_boundary.py \
  --model_name_or_path "$MODEL" \
  --model_label "$MODEL_LABEL" \
  --output_dir "$OUT" \
  --lengths "${LENGTHS:-8192,32768}" \
  --depths "${DEPTHS:-50}" \
  --seeds "${SEEDS:-0,1,2,3,4}" \
  --distractor_counts "${DISTRACTOR_COUNTS:-0,4,16,64}" \
  --distractor_similarities "${DISTRACTOR_SIMILARITIES:-low,high,conflict}" \
  --rule_gap_tokens "${RULE_GAP_TOKENS:-512}" \
  --chain_lengths "${CHAIN_LENGTHS:-2}" \
  --competitor_counts "${COMPETITOR_COUNTS:-0,4}" \
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
echo "launched clean interference Qwen3-0.6B pid=$(cat "$OUT"/pid.txt)"
echo "log: $OUT/run.log"

