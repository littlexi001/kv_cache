#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-$ROOT/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
OUTPUT="${OUTPUT:-$ROOT/results/20260803_globalcal256_smoke4k_gpu0_v1}"
VARIANTS="${VARIANTS:-qksieve_keymse_requestlocal_valuesketch16i4_wometric_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch16i4_wometric_residualrisk4_globalalloc_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch16i4_wometric_residualrisk4_globalcal256_globalalloc_fulltopk_k1280}"
GPU="${GPU:-0}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
mkdir -p "$OUTPUT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u \
  src/run_qksieve_coldskip_longcontext_quality_20260730.py \
  --model_name_or_path "$MODEL" \
  --template "$TEMPLATE" \
  --output_dir "$OUTPUT" \
  --history_tokens 3968 \
  --stream_reference_history_tokens 3968 \
  --eval_tokens 16 \
  --topic religion \
  --seed 20260835 \
  --repeat_topic_stream_if_short \
  --prefill_chunk_tokens 1024 \
  --protect_recent_tokens 0 \
  --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
  --dtype float16 \
  --device cuda \
  --device_map balanced \
  --max_memory_per_gpu_gib 22 \
  --variants "$VARIANTS" \
  >"$OUTPUT/run.log" 2>&1

touch "$OUTPUT/ALL_COMPLETE"
