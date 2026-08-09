#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
RUN=${RUN:-$PARENT/f47_distance_bf16}
DATA_RUN=${DATA_RUN:-$RUN}
SPECS="$RUN/specs/test.json"

while ! test -f "$RUN/validation.done" || ! test -f "$DATA_RUN/data.done"; do
  sleep 30
done

spec_count=$($PY -c "import json; print(len(json.load(open('$SPECS'))['specs']))")
if test "$spec_count" -lt 2; then
  echo "No distance-conditioned candidate passed validation" >"$RUN/test.no_candidate"
  exit 0
fi

run_seed() {
  local seed=$1
  local gpu0=$2
  local gpu1=$3
  local outroot="$RUN/test/seed${seed}"
  local data="$DATA_RUN/data/ruler32k_seed${seed}_m2.jsonl"
  local pids=()
  mkdir -p "$outroot"
  for shard in 0 1; do
    local gpu=$gpu0
    if test "$shard" -eq 1; then gpu=$gpu1; fi
    local out="$outroot/shard${shard}"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
      --model-name-or-path "$MODEL" \
      --examples-jsonl "$data" \
      --specs-json "$SPECS" \
      --output-dir "$out" \
      --target-length 32768 \
      --max-new-tokens-cap 128 \
      --prefill-chunk-size 128 \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --spec-shard-count 2 \
      --spec-shard-index "$shard" \
      >"$out/stdout.log" 2>"$out/stderr.log" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$outroot" \
    >"$outroot/summary_stdout.log" 2>"$outroot/summary_stderr.log"
  touch "$outroot/stage.done"
}

run_seed 54 2 3 & p0=$!
run_seed 55 4 5 & p1=$!
run_seed 56 6 7 & p2=$!
wait "$p0" "$p1" "$p2"

mkdir -p "$RUN/test/combined"
"$PY" "$ROOT/src/summarize_multiseed.py" \
  --seed-run "54=$RUN/test/seed54" \
  --seed-run "55=$RUN/test/seed55" \
  --seed-run "56=$RUN/test/seed56" \
  --output-dir "$RUN/test/combined" \
  >"$RUN/test/combined_stdout.log" 2>"$RUN/test/combined_stderr.log"
touch "$RUN/test.done"
