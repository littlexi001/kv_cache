#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-$ROOT/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
OUTPUT="${OUTPUT:-$ROOT/results/20260803_wometric_rank32_hard2_128k_4gpu_v1}"
VARIANTS="qksieve_keymse_requestlocal_valuesketch32i4_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch32i4_wometric_fulltopk_k1280"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
mkdir -p "$OUTPUT"
cd "$ROOT"

run_topic() {
  local devices="$1"
  local topic="$2"
  local seed="$3"
  local output="$OUTPUT/$topic"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "$MODEL" \
    --template "$TEMPLATE" \
    --output_dir "$output" \
    --history_tokens 131008 \
    --stream_reference_history_tokens 131008 \
    --eval_tokens 8 \
    --topic "$topic" \
    --seed "$seed" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants "$VARIANTS" \
    >"$output/run.log" 2>&1
}

run_topic 2,3 medicine 20260832 &
pid0=$!
run_topic 4,5 religion 20260835 &
pid1=$!
wait "$pid0"
wait "$pid1"
touch "$OUTPUT/ALL_COMPLETE"
