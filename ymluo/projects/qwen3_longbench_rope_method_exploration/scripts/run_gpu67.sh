#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_longbench_rope_method_exploration
ORACLE_ROOT=/home/fdong/ymluo/projects/qwen3_longbench_oracle_evidence/outputs/hotpot_semantic_aligned_18_20260802/merged
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench/hotpotqa.jsonl
RUN=${1:-hotpot_strict18_20260803}

mkdir -p "$ROOT/outputs/$RUN"

for SHARD in 0 1; do
  GPU=$((6 + SHARD))
  OUT="$ROOT/outputs/$RUN/shard$SHARD"
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=$GPU \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "$PY" "$ROOT/src/run_longbench_rope_sparse.py" \
    --model-name-or-path "$MODEL" \
    --longbench-jsonl "$DATA" \
    --frozen-manifest "$ORACLE_ROOT/sample_manifest.jsonl" \
    --evidence-mapping "$ORACLE_ROOT/evidence_mapping.jsonl" \
    --frozen-predictions "$ORACLE_ROOT/predictions.jsonl" \
    --output-dir "$OUT" \
    --variants native_full,full_rope_replay,rope_top2,semantic_top2_postscore,local_global_postscore,local_global_blend25 \
    --ratio 0.02 \
    --local-window 128 \
    --sink-tokens 16 \
    --max-new-tokens 32 \
    --prefill-chunk-size 128 \
    --dtype bfloat16 \
    --attn-implementation eager \
    --shard-count 2 \
    --shard-index "$SHARD" \
    >"$OUT/stdout.log" 2>"$OUT/stderr.log" &
  echo $! >"$OUT/pid.txt"
  echo "launched physical GPU $GPU shard $SHARD pid $(cat "$OUT/pid.txt")"
done

