#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/attention_confidence_qwen3_8b_single_token_20260717}"
STEP="${STEP:-500}"
SHORT_MAX="${SHORT_MAX:-49500}"
MAX_LENGTH="${MAX_LENGTH:-64000}"

mkdir -p "$OUT/data" "$OUT/logs"
rm -f "$OUT"/done_*.txt "$OUT"/manifest.json "$OUT"/launcher.done "$OUT"/launcher.failed
find "$OUT/data" -maxdepth 1 -type f -name 'length_*.json.tmp' -delete

run_stage() {
  local stage="$1"
  local workers="$2"
  local device_map="$3"
  local device_spec="$4"
  local start_length="$5"
  local stop_length="$6"
  declare -a shards pids
  for ((worker=0; worker<workers; worker++)); do shards[$worker]=""; done
  local index=0
  for ((length=start_length; length<=stop_length; length+=STEP)); do
    local worker=$((index % workers))
    [[ -n "${shards[$worker]}" ]] && shards[$worker]+=","
    shards[$worker]+="$length"
    index=$((index + 1))
  done

  for ((worker=0; worker<workers; worker++)); do
    local label
    label=$(printf '%s%02d' "$stage" "$worker")
    local visible
    if [[ "$device_spec" == "single" ]]; then
      visible="$worker"
    else
      local first=$((worker * 2))
      visible="$first,$((first + 1))"
    fi
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    CUDA_VISIBLE_DEVICES="$visible" "$PY" -u "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
      --model_name_or_path "$MODEL" \
      --output_dir "$OUT" \
      --lengths "${shards[$worker]}" \
      --seed 0 \
      --code_mode single_token \
      --max_top 100 \
      --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-128}" \
      --dtype float16 \
      --device_map "$device_map" \
      --attn_implementation sdpa \
      --original_max_position_embeddings 40960 \
      --global_max_position 66000 \
      --shard_label "$label" \
      >"$OUT/logs/$label.log" 2>&1 &
    pids[$worker]=$!
    echo "started stage=$stage worker=$worker devices=$visible pid=${pids[$worker]} lengths=${shards[$worker]}"
    sleep 2
  done

  local failed=0
  for ((worker=0; worker<workers; worker++)); do
    if ! wait "${pids[$worker]}"; then
      echo "stage=$stage worker=$worker failed; inspect $OUT/logs/$(printf '%s%02d' "$stage" "$worker").log" >&2
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]]
}

if ! run_stage short 8 none single 0 "$SHORT_MAX"; then
  date -Is > "$OUT/launcher.failed"
  exit 1
fi

# Exact long-context collection needs two 24 GB cards per process; four
# balanced workers cover the 50K-64K tail without quantization or approximation.
if [[ "$MAX_LENGTH" -gt "$SHORT_MAX" ]]; then
  if ! run_stage long 4 balanced pairs $((SHORT_MAX + STEP)) "$MAX_LENGTH"; then
    date -Is > "$OUT/launcher.failed"
    exit 1
  fi
fi

"$PY" "$PROJECT/src/analyze_attention_confidence_sweep.py" --output_dir "$OUT"
"$PY" "$PROJECT/src/build_attention_confidence_manifest.py" \
  --output_dir "$OUT" \
  --site_data_dir "$OUT/site_data"
date -Is > "$OUT/launcher.done"
echo "completed: $OUT/manifest.json"
