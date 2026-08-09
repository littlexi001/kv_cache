#!/usr/bin/env bash
set -euo pipefail

# Minimal causal-closure v2 smoke.  This file is intentionally not launched by
# local tests.  It is hard-limited to remote physical GPUs 6 and 7.

PROJECT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUNNER=${PROJECT}/src/run_value_mediated_rope_probe_8b.py
OUTPUT=${PROJECT}/outputs/20260801_value_mediated_singleton_v2_bf16_noop_smoke_gpu67

mkdir -p "${OUTPUT}/shard_gpu6" "${OUTPUT}/shard_gpu7" "${OUTPUT}/merged"

COMMON=(
  --model-name-or-path "${MODEL}"
  --lengths 8192
  --num-seeds 1
  --class-sample-count 2
  --packet-gap-tokens 16
  --anchor-distances 1,2,4,8,16,32,64,128
  --fixed-anchor-distance 128
  --score-lift 0.25
  --singleton-top-n 1
  --singleton-ranking-metric abs_positive_suppression_x_dm_dscore
  --prefill-chunk-size 64
  --dtype bfloat16
  --attn-implementation eager
  --original-max-position-embeddings 40960
)

# Each process intentionally loads unquantized BF16 weights.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=6 "${PYTHON}" "${RUNNER}" \
  "${COMMON[@]}" \
  --seed-start 0 \
  --output-dir "${OUTPUT}/shard_gpu6" \
  >"${OUTPUT}/shard_gpu6/stdout.log" \
  2>"${OUTPUT}/shard_gpu6/stderr.log" &
PID6=$!

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=7 "${PYTHON}" "${RUNNER}" \
  "${COMMON[@]}" \
  --seed-start 1 \
  --output-dir "${OUTPUT}/shard_gpu7" \
  >"${OUTPUT}/shard_gpu7/stdout.log" \
  2>"${OUTPUT}/shard_gpu7/stderr.log" &
PID7=$!

STATUS=0
wait "${PID6}" || STATUS=$?
wait "${PID7}" || STATUS=$?
if [[ "${STATUS}" -ne 0 ]]; then
  echo "A BF16 shard failed; inspect shard stderr logs." >&2
  exit "${STATUS}"
fi

"${PYTHON}" "${RUNNER}" \
  --output-dir "${OUTPUT}/merged" \
  --merge-shards "${OUTPUT}/shard_gpu6,${OUTPUT}/shard_gpu7" \
  >"${OUTPUT}/merged/stdout.log" \
  2>"${OUTPUT}/merged/stderr.log"

printf 'complete\n' >"${OUTPUT}/COMPLETE"
