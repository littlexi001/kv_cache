#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/downstream_kv_retrieval_qabs8cand8_tf8_v5
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python -u src/run_qabs_downstream_kv_retrieval.py \
  --model_name_or_path /home/fdong/hrj/prove/Qwen3-0.6B \
  --output_dir "$OUT" \
  --tasks 32 \
  --records_per_task 64 \
  --chunk_size 256 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --attn_implementation eager \
  --top_fraction 0.08 \
  --protect_sink_tokens 10 \
  --protect_recent_tokens 10 \
  --log_every 4 \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
