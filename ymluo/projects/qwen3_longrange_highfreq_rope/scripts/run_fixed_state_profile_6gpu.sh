#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_longrange_highfreq_rope
PY=/home/fdong/miniconda3/envs/py312/bin/python
DATA=/home/fdong/ymluo/projects/qwen3_ruler32k_rope_method/data/qwen3_8b_ruler13_32k_m2_seed42.jsonl
MODEL=${MODEL:-$(find /home/fdong/.cache/huggingface/models--Qwen--Qwen3-8B/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)}
RUN_NAME=${1:-fixed_state_highfreq_profile_20260807}
RUN=$PROJECT/outputs/$RUN_NAME
IDS=(
  niah_multikey_3_32768_0
  fwe_32768_0
  cwe_32768_0
  niah_multivalue_32768_0
  qa_squad_32768_1
  qa_hotpot_32768_0
)

test -n "$MODEL"
test -f "$MODEL/model-00001-of-00005.safetensors"
mkdir -p "$RUN"

on_exit() {
  rc=$?
  if test "$rc" -ne 0; then touch "$RUN/launcher.failed"; fi
}
trap on_exit EXIT

pids=()
for gpu in $(seq 0 5); do
  out="$RUN/sample$gpu"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$PY" -u "$PROJECT/src/profile_fixed_state_highfreq.py" \
    --model-name-or-path "$MODEL" \
    --examples-jsonl "$DATA" \
    --output-dir "$out" \
    --sample-ids "${IDS[$gpu]}" \
    --target-length 32768 \
    --prefill-chunk-size 256 \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --load-in-4bit \
    --high-frequency-end 8 \
    >"$out/stdout.log" 2>"$out/stderr.log" &
  echo $! >"$out/pid.txt"
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
touch "$RUN/launcher.done"
trap - EXIT
