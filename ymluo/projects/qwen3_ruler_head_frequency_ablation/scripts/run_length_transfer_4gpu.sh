#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
RUN=${RUN:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
DATA=${DATA:-$RUN/length_transfer/ruler_transfer_seed53_m1.jsonl}
SPECS=${SPECS:-$RUN/cross_benchmarks/bf16_smoke/specs.json}
OUT=${OUT:-$RUN/length_transfer}

run_one() {
  local gpu=$1
  local length=$2
  local shard=$3
  local out_dir="$OUT/len${length}/shard${shard}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
    --model-name-or-path "$MODEL" \
    --examples-jsonl "$DATA" \
    --specs-json "$SPECS" \
    --output-dir "$out_dir" \
    --target-length "$length" \
    --max-new-tokens-cap 128 \
    --prefill-chunk-size 256 \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --load-in-4bit \
    --original-max-position-embeddings 40960 \
    --global-max-position 131072 \
    --spec-shard-count 2 \
    --spec-shard-index "$shard" \
    >"$out_dir/run.log" 2>&1
}

# The 8K jobs finish first; reuse the same two GPUs for 64K afterward.
(run_one 0 8192 0 && run_one 0 65536 0) &
p0=$!
(run_one 1 8192 1 && run_one 1 65536 1) &
p1=$!
run_one 2 16384 0 &
p2=$!
run_one 3 16384 1 &
p3=$!

wait "$p0" "$p1" "$p2" "$p3"
touch "$OUT/launcher.done"
