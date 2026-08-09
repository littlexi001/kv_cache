#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
RUN_ROOT=$ROOT/results/20260727_groupwise_prefill_mechanism_hybrid_gpu1

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT"
cd "$ROOT"
trap 'touch "$RUN_ROOT/TERMINAL"' EXIT

CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u \
  src/analyze_groupwise_prefill_quantization_20260727.py \
  --model_name_or_path "$MODEL" \
  --output "$RUN_ROOT/mixed_a_32k.json" \
  --topic mixed_a \
  --history_tokens 32000 \
  --query_tokens 256 \
  --query_tail_tokens 8 \
  --sample_stride 32 \
  --top_fraction 0.01 \
  --group_size 16 \
  --prefill_chunk_tokens 4096 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$RUN_ROOT/run.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
