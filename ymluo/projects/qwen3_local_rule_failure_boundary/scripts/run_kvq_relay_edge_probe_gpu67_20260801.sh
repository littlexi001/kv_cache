#!/usr/bin/env bash
set -euo pipefail

# Launcher artifact only. It is intentionally hard-limited to physical GPUs
# 6 and 7. Creating and CPU-testing this file does not start either process.

PROJECT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUNNER=${PROJECT}/src/run_kvq_relay_edge_probe_8b.py
OUTPUT=${PROJECT}/outputs/20260801_kvq_relay_edge_probe_gpu67

mkdir -p "${OUTPUT}/shard_gpu6" "${OUTPUT}/shard_gpu7" "${OUTPUT}/merged"

COMMON=(
  --model-name-or-path "${MODEL}"
  --lengths 8192,16384,32768
  --conditions mixed
  --placement prefix
  --code-mode english_single_token
  --max-block-tokens 64
  --max-candidate-blocks 64
  --maximum-matched-negatives 32
  --block-temperature 1.0
  --value-temperature 1.0
  --fd-epsilon 0.05
  --fd-batch-size 8
  --fd-audit-relative-tolerance 0.35
  --fd-audit-cosine-tolerance 0.90
  --baseline-q-max-abs-tolerance 0.0001
  --prefill-chunk-size 64
  --dtype bfloat16
  --load-in-4bit
  --attn-implementation sdpa
  --original-max-position-embeddings 40960
  --global-max-position 70000
)

KVQ_PREKEY_STORAGE=cpu \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=6 \
"${PYTHON}" "${RUNNER}" "${COMMON[@]}" \
  --seed-start 0 --num-seeds 4 \
  --output-dir "${OUTPUT}/shard_gpu6" \
  >"${OUTPUT}/shard_gpu6/run.log" 2>&1 &
PID6=$!

KVQ_PREKEY_STORAGE=cpu \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=7 \
"${PYTHON}" "${RUNNER}" "${COMMON[@]}" \
  --seed-start 4 --num-seeds 4 \
  --output-dir "${OUTPUT}/shard_gpu7" \
  >"${OUTPUT}/shard_gpu7/run.log" 2>&1 &
PID7=$!

STATUS=0
wait "${PID6}" || STATUS=$?
wait "${PID7}" || STATUS=$?
if [[ "${STATUS}" -ne 0 ]]; then
  printf 'failed: %s\n' "${STATUS}" >"${OUTPUT}/failed.txt"
  exit "${STATUS}"
fi

"${PYTHON}" "${RUNNER}" \
  --output-dir "${OUTPUT}/merged" \
  --merge-shards "${OUTPUT}/shard_gpu6,${OUTPUT}/shard_gpu7"

printf 'ok\n' >"${OUTPUT}/done.txt"
