#!/usr/bin/env bash
set -euo pipefail

# Corrective smoke: the first query-span run forced the conflict record into
# the 16-token sink.  This run removes sinks so gold and conflict are both
# genuine remote candidates.  It is hard-limited to physical GPUs 6 and 7.

ROOT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUNNER="$ROOT/src/run_queryspan_prerope_retrieval_probe_8b.py"
OUTPUT="$ROOT/outputs/20260801_queryspan_sink0_smoke_gpu67"

mkdir -p "$OUTPUT/shard_gpu6" "$OUTPUT/shard_gpu7" "$OUTPUT/merged"

COMMON=(
  --model-name-or-path "$MODEL"
  --lengths 8192
  --ratio 0.02
  --variants native_noop,exact_final_pre_top2_postscore,queryspan_block_top2_postscore,queryspan_tokenmax_top2_postscore
  --local-window 128
  --sink-tokens 0
  --block-size 64
  --query-anchor-count 16
  --score-chunk-blocks 32
  --class-sample-count 8
  --packet-gap-tokens 16
  --prefill-chunk-size 64
  --dtype bfloat16
  --load-in-4bit
  --attn-implementation sdpa
  --original-max-position-embeddings 40960
  --global-max-position 70000
)

QUERYSPAN_PREKEY_STORAGE=cuda \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=6 \
"$PYTHON" "$RUNNER" "${COMMON[@]}" \
  --seed-start 0 --num-seeds 2 \
  --output-dir "$OUTPUT/shard_gpu6" \
  >"$OUTPUT/shard_gpu6/stdout.log" 2>"$OUTPUT/shard_gpu6/stderr.log" &
PID6=$!

QUERYSPAN_PREKEY_STORAGE=cuda \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=7 \
"$PYTHON" "$RUNNER" "${COMMON[@]}" \
  --seed-start 2 --num-seeds 2 \
  --output-dir "$OUTPUT/shard_gpu7" \
  >"$OUTPUT/shard_gpu7/stdout.log" 2>"$OUTPUT/shard_gpu7/stderr.log" &
PID7=$!

STATUS=0
wait "$PID6" || STATUS=$?
wait "$PID7" || STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
  printf 'failed:%s\n' "$STATUS" >"$OUTPUT/FAILED"
  exit "$STATUS"
fi

"$PYTHON" "$RUNNER" \
  --output-dir "$OUTPUT/merged" \
  --merge-shards "$OUTPUT/shard_gpu6,$OUTPUT/shard_gpu7"

printf 'complete\n' >"$OUTPUT/COMPLETE"

