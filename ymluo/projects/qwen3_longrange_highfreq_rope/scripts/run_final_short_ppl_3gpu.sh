#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-final_short_ppl_20260807}"
PROJECT=/home/fdong/ymluo/projects/qwen3_longrange_highfreq_rope
MODEL=/home/fdong/.cache/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DATA=/home/fdong/ymluo/datasets/wikitext103/wikitext-103-raw-v1/test-00000-of-00001.parquet
PYTHON=/home/fdong/miniconda3/envs/py312/bin/python
RUN="$PROJECT/outputs/$RUN_NAME"
LENGTHS="${LENGTHS:-2048,4096}"
COUNT="${COUNT:-64}"

mkdir -p "$RUN/specs"
"$PYTHON" "$PROJECT/src/make_final_candidate_specs.py" \
  --output "$RUN/specs/final.json"

variants=(native_rope late_l24_f00_11_delete late_l30_f00_15_delete)
pids=()
for gpu in 0 1 2; do
  variant="${variants[$gpu]}"
  shard="$RUN/$variant"
  mkdir -p "$shard"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$PROJECT/src/eval_short_ppl.py" \
    --model-name-or-path "$MODEL" \
    --parquet "$DATA" \
    --specs-json "$RUN/specs/final.json" \
    --variants "$variant" \
    --sequence-lengths "$LENGTHS" \
    --sequences-per-length "$COUNT" \
    --output "$shard/results.json" \
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
