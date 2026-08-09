#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/full_pre_softmax_qwen3_8b_english_128k_20260718}"
EXPORT="$OUT/full_pre_softmax"

mkdir -p "$OUT/logs" "$EXPORT"
rm -f "$OUT/launcher.done" "$OUT/launcher.failed"
find "$EXPORT" -type f -name '*.tmp' -delete

failure() {
  date -Is >"$OUT/launcher.failed"
}
trap failure ERR

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
"$PY" -u "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --lengths 128000 \
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
  --export_full_pre_softmax_dir "$EXPORT" \
  --shard_label full_pre_softmax_128k \
  >"$OUT/logs/run.log" 2>&1

date -Is >"$OUT/launcher.done"
echo "completed: $EXPORT/length_128000/manifest.json"
