#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_longrange_highfreq_rope
BASE=/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation
PY=/home/fdong/miniconda3/envs/py312/bin/python
DATA=/home/fdong/ymluo/projects/qwen3_ruler32k_rope_method/data/qwen3_8b_ruler13_32k_m2_seed42.jsonl
MODEL=${MODEL:-$(find /home/fdong/.cache/huggingface/models--Qwen--Qwen3-8B/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)}
SAMPLE_IDS=niah_multikey_3_32768_0,fwe_32768_0,cwe_32768_0,niah_multivalue_32768_0,qa_squad_32768_1,qa_hotpot_32768_0
RUN_NAME=${1:-highfreq_discovery_20260807}
RUN=$PROJECT/outputs/$RUN_NAME
SPECS=$RUN/specs/discovery.json

test -n "$MODEL"
test -f "$MODEL/config.json"
test -f "$DATA"
mkdir -p "$RUN/specs"

on_exit() {
  rc=$?
  if test "$rc" -ne 0; then touch "$RUN/launcher.failed"; fi
}
trap on_exit EXIT

"$PY" "$PROJECT/src/make_highfreq_specs.py" --output "$SPECS"

pids=()
for shard in $(seq 0 7); do
  out="$RUN/shard$shard"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$shard" TOKENIZERS_PARALLELISM=false \
    "$PY" -u "$BASE/src/run_frequency_sweep.py" \
    --model-name-or-path "$MODEL" \
    --examples-jsonl "$DATA" \
    --specs-json "$SPECS" \
    --output-dir "$out" \
    --sample-ids "$SAMPLE_IDS" \
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
"$PY" "$BASE/src/summarize_sweep.py" --run-dir "$RUN" \
  >"$RUN/summary_stdout.log" 2>"$RUN/summary_stderr.log"
touch "$RUN/launcher.done"
trap - EXIT
