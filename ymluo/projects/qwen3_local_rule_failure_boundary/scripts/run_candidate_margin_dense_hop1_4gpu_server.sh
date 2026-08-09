#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/candidate_margin_dense_hop1_34_100_20260724}"
START="${START:-34}"
STOP="${STOP:-100}"
GPU_START="${GPU_START:-4}"
GPU_COUNT="${GPU_COUNT:-4}"

mkdir -p "$OUT/data" "$OUT/logs"
exec 8>"$OUT/launcher.lock"
flock -n 8 || {
  echo "single-hop candidate-margin launcher already running"
  exit 0
}
rm -f "$OUT/launcher.done" "$OUT/launcher.failed"
trap 'date -Is >"$OUT/launcher.failed"' ERR

for ((offset=0; offset<GPU_COUNT; offset++)); do
  gpu=$((GPU_START + offset))
  read -r memory utilization < <(
    nvidia-smi -i "$gpu" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
      tr -d ','
  )
  if (( memory >= 2000 || utilization >= 10 )); then
    echo "GPU $gpu is not idle: memory=$memory utilization=$utilization"
    exit 1
  fi
done

declare -a shards
for ((worker=0; worker<GPU_COUNT; worker++)); do
  shards[$worker]=""
done

index=0
for length in $(seq "$START" "$STOP"); do
  worker=$((index % GPU_COUNT))
  [[ -n "${shards[$worker]}" ]] && shards[$worker]+=","
  shards[$worker]+="$length"
  index=$((index + 1))
done

declare -a pids
for ((worker=0; worker<GPU_COUNT; worker++)); do
  gpu=$((GPU_START + worker))
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u \
    "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUT" \
    --lengths "${shards[$worker]}" \
    --seed 0 \
    --code_mode english_single_token \
    --placement middle \
    --query_mode hop1 \
    --prompt_style legacy \
    --max_top 100 \
    --prefill_chunk_size 128 \
    --dtype float16 \
    --device cuda \
    --device_map none \
    --attn_implementation sdpa \
    --original_max_position_embeddings 40960 \
    --global_max_position 130000 \
    --shard_label "gpu${gpu}" \
    >"$OUT/logs/gpu${gpu}.log" 2>&1 &
  pids[$worker]=$!
  echo "started worker=$worker gpu=$gpu pid=${pids[$worker]} lengths=${shards[$worker]}"
done

failed=0
for ((worker=0; worker<GPU_COUNT; worker++)); do
  if ! wait "${pids[$worker]}"; then
    echo "worker $worker failed" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi

rm -f "$OUT/launcher.failed"
date -Is >"$OUT/launcher.done"
echo "$(date -Is) complete: $OUT"
