#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 GPU_INDEX SEED_START NUM_SEEDS" >&2
    exit 2
fi

GPU_INDEX=$1
SEED_START=$2
NUM_SEEDS=$3

BASE=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
SEED_END=$((SEED_START + NUM_SEEDS - 1))
OUTPUT="$BASE/outputs/local_global_rope_heldout64k_gpu${GPU_INDEX}_seeds${SEED_START}_${SEED_END}_20260730"

mkdir -p "$OUTPUT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$BASE/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$BASE"

printf '%s\n' "$$" > "$OUTPUT/launcher.pid"
rm -f "$OUTPUT/launcher.done" "$OUTPUT/launcher.failed"

if "$PYTHON" -u src/run_local_global_rope_probe_8b.py \
    --model-name-or-path "$MODEL" \
    --output-dir "$OUTPUT" \
    --lengths 65536 \
    --seed-start "$SEED_START" \
    --num-seeds "$NUM_SEEDS" \
    --ratio 0.02 \
    --local-window 128 \
    --sink-tokens 16 \
    --prefill-chunk-size 128 \
    --dtype bfloat16 \
    --load-in-4bit \
    --attn-implementation sdpa \
    --original-max-position-embeddings 40960 \
    --global-max-position 70000 \
    > "$OUTPUT/run.log" 2>&1; then
    date --iso-8601=seconds > "$OUTPUT/launcher.done"
else
    status=$?
    printf '%s exit=%s\n' "$(date --iso-8601=seconds)" "$status" \
        > "$OUTPUT/launcher.failed"
    exit "$status"
fi
