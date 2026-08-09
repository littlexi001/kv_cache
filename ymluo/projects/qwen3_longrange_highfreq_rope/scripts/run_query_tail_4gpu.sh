#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-query_tail_discovery_20260807}"
PROJECT=/home/fdong/ymluo/projects/qwen3_longrange_highfreq_rope
BASE=/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation
MODEL=/home/fdong/.cache/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DATA=/home/fdong/ymluo/projects/qwen3_ruler32k_rope_method/data/qwen3_8b_ruler13_32k_m2_seed42.jsonl
PYTHON=/home/fdong/miniconda3/envs/py312/bin/python
RUN="$PROJECT/outputs/$RUN_NAME"
SAMPLES=niah_multikey_3_32768_0,fwe_32768_0,cwe_32768_0,niah_multivalue_32768_0,qa_squad_32768_1,qa_hotpot_32768_0

mkdir -p "$RUN/specs"
"$PYTHON" "$PROJECT/src/make_highfreq_specs.py" \
  --output "$RUN/specs/discovery.json"

pids=()
for offset in 0 1 2 3; do
  spec_index=$((11 + offset))
  shard="$RUN/spec${spec_index}"
  mkdir -p "$shard"
  CUDA_VISIBLE_DEVICES="$offset" "$PYTHON" -u "$BASE/src/run_frequency_sweep.py" \
    --model-name-or-path "$MODEL" \
    --examples-jsonl "$DATA" \
    --specs-json "$RUN/specs/discovery.json" \
    --output-dir "$shard" \
    --sample-ids "$SAMPLES" \
    --target-length 32768 \
    --max-new-tokens-cap 64 \
    --prefill-chunk-size 256 \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --load-in-4bit \
    --spec-shard-count 15 \
    --spec-shard-index "$spec_index" \
    > "$shard/stdout.log" 2> "$shard/stderr.log" &
  echo "$!" > "$shard/pid.txt"
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  touch "$RUN/launcher.failed"
  exit 1
fi
touch "$RUN/launcher.done"
