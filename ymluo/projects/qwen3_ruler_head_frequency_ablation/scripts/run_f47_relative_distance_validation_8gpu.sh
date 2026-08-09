#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
DATA_ROOT=${DATA_ROOT:-$PARENT/data}
RUN=${RUN:-$PARENT/f47_relative_distance_bf16}
WAIT_FOR=${WAIT_FOR:-$PARENT/f47_relative_smoke/run/done.txt}

if test -n "$WAIT_FOR"; then
  while ! test -f "$WAIT_FOR"; do sleep 10; done
fi

mkdir -p "$RUN/specs"
"$PY" "$ROOT/src/make_f47_relative_distance_specs.py" \
  --output "$RUN/specs/validation.json"

run_seed() {
  local seed=$1
  local first_gpu=$2
  local outroot="$RUN/validation/seed${seed}"
  local data="$DATA_ROOT/ruler32k_seed${seed}_m1.jsonl"
  local pids=()
  mkdir -p "$outroot"
  for shard in 0 1 2 3; do
    local gpu=$((first_gpu + shard))
    local out="$outroot/shard${shard}"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
        --model-name-or-path "$MODEL" \
        --examples-jsonl "$data" \
        --specs-json "$RUN/specs/validation.json" \
        --output-dir "$out" \
        --target-length 32768 \
        --max-new-tokens-cap 128 \
        --prefill-chunk-size 128 \
        --dtype bfloat16 \
        --attn-implementation sdpa \
        --original-max-position-embeddings 40960 \
        --global-max-position 40960 \
        --spec-shard-count 4 \
        --spec-shard-index "$shard" \
        >"$out/stdout.log" 2>"$out/stderr.log" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$outroot" \
    >"$outroot/summary_stdout.log" 2>"$outroot/summary_stderr.log"
}

run_seed 43 0 & p0=$!
run_seed 44 4 & p1=$!
wait "$p0" "$p1"

mkdir -p "$RUN/validation/combined"
"$PY" "$ROOT/src/summarize_multiseed.py" \
  --seed-run "43=$RUN/validation/seed43" \
  --seed-run "44=$RUN/validation/seed44" \
  --output-dir "$RUN/validation/combined" \
  >"$RUN/validation/combined_stdout.log" \
  2>"$RUN/validation/combined_stderr.log"
touch "$RUN/validation.done"
