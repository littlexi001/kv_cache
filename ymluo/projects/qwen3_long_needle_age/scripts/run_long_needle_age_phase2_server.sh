#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_long_needle_age}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
OUT="${OUT:-$ROOT/outputs/long_needle_age_phase2_20260709}"

mkdir -p "$OUT"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_one_length() {
  local length="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  nohup "$PY" -u src/run_long_needle_age.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$out_dir" \
  --lengths "$length" \
  --depths "${DEPTHS:-10,50,90}" \
  --seeds "${SEEDS:-0,1,2,3,4}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPL:-sdpa}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-4096}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
  --compute_attention "${COMPUTE_ATTENTION:-true}" \
  --needle_prompt_lang "${NEEDLE_PROMPT_LANG:-zh}" \
  --filler_lang "${FILLER_LANG:-zh}" \
  > "$out_dir/run.log" 2>&1 < /dev/null &
  echo "$!" > "$out_dir/pid.txt"
  echo "started length=$length pid=$(cat "$out_dir/pid.txt") log=$out_dir/run.log"
}

run_one_length "${LENGTH_64K:-65536}" "$OUT/len65536"
run_one_length "${LENGTH_128K:-131072}" "$OUT/len131072"
