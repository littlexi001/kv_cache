#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_inference_rnope
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DATA=/home/fdong/ymluo/projects/qwen3_ruler32k_rope_method/data/qwen3_8b_ruler13_32k_m2_seed42.jsonl
RUN_NAME=${1:-ruler32k_m2_8gpu_20260804}
RUN_ROOT="$ROOT/outputs/$RUN_NAME"
SMOKE="$RUN_ROOT/smoke"
mkdir -p "$SMOKE"

on_error() {
  touch "$RUN_ROOT/launcher.failed"
}
trap on_error ERR

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u "$ROOT/src/run_inference_rnope_ruler.py" \
  --model-name-or-path "$MODEL" \
  --examples-jsonl "$DATA" \
  --output-dir "$SMOKE" \
  --variants native_rope,native_replay,nope_every4_offset3 \
  --tasks niah_multivalue \
  --target-length 32768 \
  --max-samples-per-task 1 \
  --max-new-tokens-cap 64 \
  --prefill-chunk-size 256 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --load-in-4bit \
  >"$SMOKE/stdout.log" 2>"$SMOKE/stderr.log"

"$PY" - "$SMOKE/rows.jsonl" <<'PY'
import json, math, sys
rows=[json.loads(x) for x in open(sys.argv[1], encoding='utf-8') if x.strip()]
assert len(rows)==3, len(rows)
assert all(r['finite_logits'] and math.isfinite(r['gold_answer_mean_nll']) for r in rows)
replay=[r for r in rows if r['variant']=='native_replay'][0]
assert replay['native_replay_max_logit_error'] is not None
assert replay['native_replay_max_logit_error'] < 1e-4, replay['native_replay_max_logit_error']
PY
touch "$RUN_ROOT/smoke.done"

pids=()
for shard in $(seq 0 7); do
  out="$RUN_ROOT/shard$shard"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$shard" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$ROOT/src/run_inference_rnope_ruler.py" \
    --model-name-or-path "$MODEL" \
    --examples-jsonl "$DATA" \
    --output-dir "$out" \
    --variants native_rope,nope_every4_offset3,nope_every4_offset0,nope_alternating_odd \
    --target-length 32768 \
    --max-samples-per-task 2 \
    --max-new-tokens-cap 128 \
    --prefill-chunk-size 256 \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --load-in-4bit \
    --shard-count 8 \
    --shard-index "$shard" \
    >"$out/stdout.log" 2>"$out/stderr.log" &
  echo $! >"$out/pid.txt"
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" "$ROOT/src/summarize_inference_rnope.py" --run-dir "$RUN_ROOT" \
  >"$RUN_ROOT/summary_stdout.log" 2>"$RUN_ROOT/summary_stderr.log"
touch "$RUN_ROOT/launcher.done"
trap - ERR

