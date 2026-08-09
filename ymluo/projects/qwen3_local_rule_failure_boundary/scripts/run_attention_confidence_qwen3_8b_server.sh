#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/attention_confidence_qwen3_8b_20260717}"
GPU_COUNT="${GPU_COUNT:-8}"
MAX_LENGTH="${MAX_LENGTH:-64000}"
STEP="${STEP:-500}"

mkdir -p "$OUT/data" "$OUT/logs"
rm -f "$OUT"/done_*.txt "$OUT"/manifest.json "$OUT"/launcher.done "$OUT"/launcher.failed
find "$OUT/data" -maxdepth 1 -type f -name 'length_*.json.tmp' -delete

declare -a shard_lengths
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
  shard_lengths[$gpu]=""
done
index=0
for ((length=0; length<=MAX_LENGTH; length+=STEP)); do
  gpu=$((index % GPU_COUNT))
  if [[ -n "${shard_lengths[$gpu]}" ]]; then
    shard_lengths[$gpu]+=","
  fi
  shard_lengths[$gpu]+="$length"
  index=$((index + 1))
done

declare -a pids
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
  label=$(printf 'gpu%02d' "$gpu")
  log="$OUT/logs/$label.log"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUT" \
    --lengths "${shard_lengths[$gpu]}" \
    --seed 0 \
    --max_top 100 \
    --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-128}" \
    --dtype float16 \
    --device_map none \
    --attn_implementation sdpa \
    --original_max_position_embeddings 40960 \
    --global_max_position 66000 \
    --shard_label "$label" \
    >"$log" 2>&1 &
  pids[$gpu]=$!
  echo "started gpu=$gpu pid=${pids[$gpu]} lengths=${shard_lengths[$gpu]} log=$log"
  sleep 2
done

failed=0
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
  if ! wait "${pids[$gpu]}"; then
    echo "gpu=$gpu failed; inspect $OUT/logs/gpu$(printf '%02d' "$gpu").log" >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  date -Is > "$OUT/launcher.failed"
  exit 1
fi

"$PY" "$PROJECT/src/analyze_attention_confidence_sweep.py" --output_dir "$OUT"
"$PY" "$PROJECT/src/build_attention_confidence_manifest.py" \
  --output_dir "$OUT" \
  --site_data_dir "$OUT/site_data"
date -Is > "$OUT/launcher.done"
echo "completed: $OUT/manifest.json"
