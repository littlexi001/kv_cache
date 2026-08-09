#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/rope_pair_contributions_qwen3_8b_64k_20260720}"

mkdir -p "$OUT/logs"
rm -f "$OUT/done.txt" "$OUT/launcher.done" "$OUT/launcher.failed"

failure() {
  date -Is >"$OUT/launcher.failed"
}
trap failure ERR

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
"$PY" -u "$PROJECT/src/export_rope_pair_contributions_8b.py" \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT/site_data" \
  --length 64000 \
  --placement middle \
  --seed 0 \
  --bin_size 128 \
  --head_batch_size 8 \
  --prefill_chunk_size 128 \
  --dtype float16 \
  --device_map balanced \
  --attn_implementation sdpa \
  --original_max_position_embeddings 40960 \
  --global_max_position 130000 \
  >"$OUT/logs/run.log" 2>&1

date -Is >"$OUT/launcher.done"
echo "completed: $OUT/site_data/manifest.json"
