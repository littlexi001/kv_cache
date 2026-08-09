#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUNNER="$PROJECT/src/run_blockwise_phase_transport_probe_8b.py"
ROOT="$PROJECT/outputs/20260801_blockwise_phase_transport_gpu67"
BASELINES=full_rope,rope_top2,local_global_postscore

mkdir -p "$ROOT/block16_gpu6" "$ROOT/block32_gpu7"

CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$RUNNER" \
  --model-name-or-path "$MODEL" \
  --output-dir "$ROOT/block16_gpu6" \
  --lengths 8192,32768 \
  --seed-start 0 \
  --num-seeds 4 \
  --variants "$BASELINES,block16_selector_only,block16_clipped_consumer,block16_transport,block16_transport_masspreserve,block16_random_matched" \
  --ratio 0.02 \
  --local-window 128 \
  --sink-tokens 16 \
  --prefill-chunk-size 128 \
  --dtype bfloat16 \
  --load-in-4bit \
  --attn-implementation sdpa \
  --original-max-position-embeddings 40960 \
  --global-max-position 70000 \
  >"$ROOT/block16_gpu6.log" 2>&1 &
PID16=$!

CUDA_VISIBLE_DEVICES=7 "$PYTHON" "$RUNNER" \
  --model-name-or-path "$MODEL" \
  --output-dir "$ROOT/block32_gpu7" \
  --lengths 8192,32768 \
  --seed-start 0 \
  --num-seeds 4 \
  --variants "$BASELINES,block32_selector_only,block32_clipped_consumer,block32_transport,block32_transport_masspreserve,block32_random_matched" \
  --ratio 0.02 \
  --local-window 128 \
  --sink-tokens 16 \
  --prefill-chunk-size 128 \
  --dtype bfloat16 \
  --load-in-4bit \
  --attn-implementation sdpa \
  --original-max-position-embeddings 40960 \
  --global-max-position 70000 \
  >"$ROOT/block32_gpu7.log" 2>&1 &
PID32=$!

STATUS=0
wait "$PID16" || STATUS=$?
wait "$PID32" || STATUS=$?
exit "$STATUS"
