#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
MODEL_LABEL="${MODEL_LABEL:-qwen3_0p6b}"
OUT="${OUT:-$ROOT/outputs/question3_boundary_qwen06_phase1_20260709}"

mkdir -p "$OUT"
cd "$ROOT"
rm -f "$OUT"/results.csv "$OUT"/candidate_scores.csv "$OUT"/attention_selectivity.csv \
  "$OUT"/summary_by_condition.csv "$OUT"/failure_boundary.csv "$OUT"/summary.md \
  "$OUT"/cases.jsonl "$OUT"/skipped_cases.jsonl "$OUT"/env.json

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup "$PY" -u src/run_local_rule_failure_boundary.py \
  --model_name_or_path "$MODEL" \
  --model_label "$MODEL_LABEL" \
  --output_dir "$OUT" \
  --lengths "${LENGTHS:-1024,4096,8192,16384,32768}" \
  --depths "${DEPTHS:-10,50,90}" \
  --seeds "${SEEDS:-0,1}" \
  --distractor_counts "${DISTRACTOR_COUNTS:-0,16,64}" \
  --distractor_similarities "${DISTRACTOR_SIMILARITIES:-low,high,conflict}" \
  --rule_gap_tokens "${RULE_GAP_TOKENS:-0,512,2048}" \
  --chain_lengths "${CHAIN_LENGTHS:-1,2,4}" \
  --competitor_counts "${COMPETITOR_COUNTS:-0,4}" \
  --max_cases "${MAX_CASES:-420}" \
  --case_order shuffled \
  --case_order_seed "${CASE_ORDER_SEED:-20260709}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPL:-sdpa}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-4096}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
  --compute_attention "${COMPUTE_ATTENTION:-true}" \
  --attention_mode "${ATTENTION_MODE:-failures}" \
  --attention_sample_rate "${ATTENTION_SAMPLE_RATE:-0.10}" \
  --max_attention_prompt_tokens "${MAX_ATTENTION_PROMPT_TOKENS:-65536}" \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started pid=$(cat "$OUT/pid.txt")"
echo "log=$OUT/run.log"
