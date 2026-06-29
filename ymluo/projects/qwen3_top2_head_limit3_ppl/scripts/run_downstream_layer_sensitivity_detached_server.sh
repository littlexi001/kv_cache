#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/downstream_layer_sensitivity_qabs5_single_full_v1
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python -u src/run_downstream_layer_sensitivity.py \
  --model_name_or_path /home/fdong/hrj/prove/Qwen3-0.6B \
  --output_dir "$OUT" \
  --tasks 8 \
  --records_per_task 64 \
  --chunk_size 256 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --attn_implementation eager \
  --top_fraction 0.05 \
  --candidate_fraction 0.05 \
  --protect_sink_tokens 10 \
  --protect_recent_tokens 10 \
  --log_every 2 \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"

