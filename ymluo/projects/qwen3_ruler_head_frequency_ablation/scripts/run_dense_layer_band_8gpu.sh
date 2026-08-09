#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DATA=/home/fdong/ymluo/projects/qwen3_ruler32k_rope_method/data/qwen3_8b_ruler13_32k_m2_seed42.jsonl
SCREEN_IDS=niah_multikey_3_32768_0,fwe_32768_0,cwe_32768_0,niah_multivalue_32768_0,qa_squad_32768_1,qa_hotpot_32768_0
RUN_NAME=${1:-dense_layer_band_sweep_20260806}
RUN=$ROOT/outputs/$RUN_NAME
SPECS=$RUN/specs/dense_layer_band.json
mkdir -p "$RUN/specs"

on_exit() {
  rc=$?
  if test "$rc" -ne 0; then touch "$RUN/launcher.failed"; fi
}
trap on_exit EXIT

"$PY" "$ROOT/src/make_specs.py" \
  --stage dense_layer_band --output "$SPECS"

pids=()
for shard in $(seq 0 7); do
  out="$RUN/shard$shard"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$shard" TOKENIZERS_PARALLELISM=false \
    "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
    --model-name-or-path "$MODEL" \
    --examples-jsonl "$DATA" \
    --specs-json "$SPECS" \
    --output-dir "$out" \
    --sample-ids "$SCREEN_IDS" \
    --target-length 32768 \
    --max-new-tokens-cap 64 \
    --prefill-chunk-size 256 \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --load-in-4bit \
    --spec-shard-count 8 \
    --spec-shard-index "$shard" \
    >"$out/stdout.log" 2>"$out/stderr.log" &
  echo $! >"$out/pid.txt"
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$RUN" \
  >"$RUN/summary_stdout.log" 2>"$RUN/summary_stderr.log"
touch "$RUN/launcher.done"
trap - EXIT
