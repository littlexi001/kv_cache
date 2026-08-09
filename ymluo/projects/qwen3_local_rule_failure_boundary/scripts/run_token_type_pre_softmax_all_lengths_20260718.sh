#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/token_type_pre_softmax_all_lengths_20260718}"
EXPORT="$OUT/token_type_lengths"
SITE_DATA="$OUT/site_data/token_type_all_lengths"

mkdir -p "$OUT/logs" "$EXPORT" "$SITE_DATA"
rm -f "$OUT/launcher.done" "$OUT/launcher.failed" "$SITE_DATA/complete.txt"
find "$EXPORT" -type f -name '*.tmp' -delete

failure() {
  date -Is >"$OUT/launcher.failed"
}
trap failure ERR

mapfile -t ALL_LENGTHS < <(seq 0 500 128000)
declare -a SHARD_LENGTHS=("" "" "" "")
for index in "${!ALL_LENGTHS[@]}"; do
  shard=$((index % 4))
  if [[ -z "${SHARD_LENGTHS[$shard]}" ]]; then
    SHARD_LENGTHS[$shard]="${ALL_LENGTHS[$index]}"
  else
    SHARD_LENGTHS[$shard]="${SHARD_LENGTHS[$shard]},${ALL_LENGTHS[$index]}"
  fi
done

declare -a GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
declare -a PIDS=()
for shard in 0 1 2 3; do
  shard_output="$OUT/shard_$shard"
  mkdir -p "$shard_output"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES="${GPU_PAIRS[$shard]}" \
  "$PY" -u "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$shard_output" \
    --lengths "${SHARD_LENGTHS[$shard]}" \
    --seed 0 \
    --code_mode english_single_token \
    --placement middle \
    --query_mode full2 \
    --prompt_style legacy \
    --max_top 100 \
    --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-128}" \
    --dtype float16 \
    --device_map balanced \
    --attn_implementation sdpa \
    --original_max_position_embeddings 40960 \
    --global_max_position 130000 \
    --export_token_type_pre_softmax_dir "$EXPORT" \
    --shard_label "token_type_$shard" \
    >"$OUT/logs/shard_$shard.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more attention shards failed" >&2
  exit 1
fi

"$PY" -u "$PROJECT/src/build_token_type_length_index.py" \
  --input_dir "$EXPORT" \
  --output_dir "$SITE_DATA" \
  --expected_length_count 257 \
  >"$OUT/logs/build_site_index.log" 2>&1

date -Is >"$OUT/launcher.done"
echo "completed: $SITE_DATA/manifest.json"
