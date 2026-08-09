#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/single_sample_margin_boundary_refine_20260723}"

mkdir -p "$OUT/data"
exec 8>"$OUT/supplement.lock"
flock -n 8 || {
  echo "supplement already running"
  exit 0
}
rm -f "$OUT/supplement.done" "$OUT/supplement.failed"
trap 'date -Is >"$OUT/supplement.failed"' ERR

gpu="$(
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, '{
      gsub(/ /, "", $1);
      gsub(/ /, "", $2);
      gsub(/ /, "", $3);
      if (($1 + 0) >= 4 && ($1 + 0) <= 7 &&
          ($2 + 0) < 2000 && ($3 + 0) < 10) {
        print $1;
        exit;
      }
    }'
)"
if [[ -z "$gpu" ]]; then
  echo "no idle GPU in 4..7"
  exit 1
fi

lengths="$(
  {
    seq 35 49
    seq 76 99
  } | paste -sd, -
)"
echo "$(date -Is) selected GPU $gpu; lengths=$lengths"

CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u \
  "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --lengths "$lengths" \
  --seed 0 \
  --code_mode english_single_token \
  --placement middle \
  --query_mode full2 \
  --prompt_style legacy \
  --max_top 100 \
  --prefill_chunk_size 128 \
  --dtype float16 \
  --device cuda \
  --device_map none \
  --attn_implementation sdpa \
  --original_max_position_embeddings 40960 \
  --global_max_position 130000 \
  --shard_label supplement_35_49_76_99 \
  >"$OUT/supplement.log" 2>&1

"$PY" "$PROJECT/src/analyze_single_sample_failure_trace.py" \
  --input_dir "$OUT/data" \
  --output_dir "$OUT/analysis" \
  >"$OUT/supplement_analysis.log" 2>&1

rm -f "$OUT/supplement.failed"
date -Is >"$OUT/supplement.done"
echo "$(date -Is) complete"
