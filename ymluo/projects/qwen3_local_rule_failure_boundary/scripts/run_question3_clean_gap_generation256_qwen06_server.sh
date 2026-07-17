#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
MODEL_LABEL="${MODEL_LABEL:-qwen3_0p6b_clean_gap_gen256}"
OUT="${OUT:-$PROJECT/outputs/clean_gap_generation256_qwen06_20260710}"

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
  --distractor_counts "${DISTRACTOR_COUNTS:-0}" \
  --distractor_similarities "${DISTRACTOR_SIMILARITIES:-low}" \
  --rule_gap_tokens "${RULE_GAP_TOKENS:-0,512,2048,4096,8192}" \
  --chain_lengths "${CHAIN_LENGTHS:-4}" \
  --competitor_counts "${COMPETITOR_COUNTS:-0}" \
  --max_cases 0 \
  --case_order sequential \
  --dtype "${DTYPE:-float16}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-4096}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --compute_attention "${COMPUTE_ATTENTION:-false}" \
  --attention_mode none \
  > "$OUT"/run.log 2>&1 &

echo $! > "$OUT"/pid.txt
echo "launched clean gap generation256 Qwen3-0.6B pid=$(cat "$OUT"/pid.txt)"
echo "log: $OUT/run.log"
