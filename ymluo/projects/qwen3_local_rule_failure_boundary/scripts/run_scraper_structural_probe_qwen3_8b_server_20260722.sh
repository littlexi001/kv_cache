#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/scraper_structural_probe_qwen3_8b_20260722}"
GPU_ID="${GPU_ID:-7}"

mkdir -p "$OUT/logs"
rm -f "$OUT/done" "$OUT/launcher.done" "$OUT/launcher.failed"
gpu_used_mb="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ -z "$gpu_used_mb" || "$gpu_used_mb" -ge 256 ]]; then
  echo "Refusing to launch: physical GPU $GPU_ID currently uses ${gpu_used_mb:-unknown} MiB" >&2
  date -Is >"$OUT/launcher.failed"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if "$PY" -u "$PROJECT/src/run_scraper_structural_probe_8b.py" \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --body_length 65536 \
  --gap 4096 \
  --filler_types plain,semantic \
  --query_modes lemma,paraphrase \
  --seed 20260722 \
  --prefill_chunk_size 128 \
  --attn_implementation sdpa \
  --original_max_position_embeddings 40960 \
  --load_in_8bit \
  >"$OUT/logs/probe.log" 2>&1; then
  date -Is >"$OUT/launcher.done"
else
  status=$?
  date -Is >"$OUT/launcher.failed"
  exit "$status"
fi

