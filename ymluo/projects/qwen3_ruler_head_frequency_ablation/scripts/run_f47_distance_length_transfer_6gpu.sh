#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
METHOD_RUN=${METHOD_RUN:-$PARENT/f47_distance_bf16}
DATA=${DATA:-$PARENT/length_transfer/ruler_transfer_seed53_m1.jsonl}
SPECS=${SPECS:-$METHOD_RUN/specs/test.json}
OUT=${OUT:-$METHOD_RUN/length_transfer}

while ! test -f "$METHOD_RUN/cross_benchmarks/launcher.done"; do
  sleep 30
done

run_one() {
  local gpu=$1
  local length=$2
  local shard=$3
  local out_dir="$OUT/len${length}/shard${shard}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
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
    --spec-shard-count 3 \
    --spec-shard-index "$shard" \
    >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"
}

(run_one 0 8192 0 && run_one 0 65536 0) & p0=$!
(run_one 1 8192 1 && run_one 1 65536 1) & p1=$!
(run_one 2 8192 2 && run_one 2 65536 2) & p2=$!
run_one 3 16384 0 & p3=$!
run_one 4 16384 1 & p4=$!
run_one 5 16384 2 & p5=$!
wait "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"

for length in 8192 16384 65536; do
  "$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$OUT/len${length}" \
    >"$OUT/len${length}/summary_stdout.log" 2>"$OUT/len${length}/summary_stderr.log"
done
touch "$OUT/launcher.done"
