#!/usr/bin/env bash
set -euo pipefail

# File-only launcher for the oracle value-mediated mechanism audit.
# It is intentionally hard-limited to physical GPUs 6 and 7 and is not
# launched by creating or testing this file.

PROJECT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUNNER=${PROJECT}/src/run_value_mediated_rope_probe_8b.py
OUTPUT=${PROJECT}/outputs/20260801_value_mediated_rope_probe_singleton_gpu67

mkdir -p "${OUTPUT}/shard_gpu6" "${OUTPUT}/shard_gpu7" "${OUTPUT}/merged"

COMMON=(
  --model-name-or-path "${MODEL}"
  --lengths 8192,32768
  --class-sample-count 8
  --packet-gap-tokens 16
  --anchor-distances 1,2,4,8,16,32,64,128
  --fixed-anchor-distance 128
  --score-lift 0.25
  --singleton-top-n 16
  --singleton-ranking-metric abs_positive_suppression_x_dm_dscore
  --prefill-chunk-size 64
  --dtype bfloat16
  --load-in-4bit
  --attn-implementation eager
  --original-max-position-embeddings 40960
)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=6 "${PYTHON}" "${RUNNER}" \
  "${COMMON[@]}" \
  --seed-start 0 \
  --num-seeds 4 \
  --output-dir "${OUTPUT}/shard_gpu6" \
  >"${OUTPUT}/shard_gpu6/stdout.log" \
  2>"${OUTPUT}/shard_gpu6/stderr.log" &
PID6=$!

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=7 "${PYTHON}" "${RUNNER}" \
  "${COMMON[@]}" \
  --seed-start 4 \
  --num-seeds 4 \
  --output-dir "${OUTPUT}/shard_gpu7" \
  >"${OUTPUT}/shard_gpu7/stdout.log" \
  2>"${OUTPUT}/shard_gpu7/stderr.log" &
PID7=$!

STATUS=0
wait "${PID6}" || STATUS=$?
wait "${PID7}" || STATUS=$?
if [[ "${STATUS}" -ne 0 ]]; then
  echo "A GPU shard failed; inspect shard_gpu6/stderr.log and shard_gpu7/stderr.log." >&2
  exit "${STATUS}"
fi

"${PYTHON}" "${RUNNER}" \
  --output-dir "${OUTPUT}/merged" \
  --merge-shards "${OUTPUT}/shard_gpu6,${OUTPUT}/shard_gpu7" \
  >"${OUTPUT}/merged/stdout.log" \
  2>"${OUTPUT}/merged/stderr.log"

printf 'complete\n' >"${OUTPUT}/COMPLETE"
