#!/usr/bin/env bash
set -euo pipefail

BASE=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUTPUT="$BASE/outputs/first_layer_rope_phase_gpu7_20260730"

mkdir -p "$OUTPUT"
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$BASE/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$BASE"

"$PYTHON" -u src/analyze_qwen3_first_layer_rope_phase.py \
    --model-name-or-path "$MODEL" \
    --output-dir "$OUTPUT" \
    --lengths 8192,16384,32768,65536 \
    --seed 0 \
    --dtype bfloat16 \
    --load-in-4bit \
    --attn-implementation sdpa \
    --original-max-position-embeddings 40960 \
    --global-max-position 70000 \
    > "$OUTPUT/run.log" 2>&1

date --iso-8601=seconds > "$OUTPUT/done.txt"
