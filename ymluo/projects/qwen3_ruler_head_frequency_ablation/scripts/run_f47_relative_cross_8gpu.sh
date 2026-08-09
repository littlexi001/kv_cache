#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
RUN=${RUN:-$PARENT/f47_relative_distance_bf16}
SPECS="$RUN/specs/test.json"
LONG_DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench/hotpotqa.jsonl
FROZEN=/home/fdong/ymluo/projects/qwen3_longbench_oracle_evidence/outputs/hotpot_semantic_aligned_18_20260802/merged
PG19=/home/fdong/ymluo/datasets/pg19/test.parquet
PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-128}

while ! test -f "$RUN/test.done" && ! test -f "$RUN/test.no_candidate"; do sleep 30; done
if test -f "$RUN/test.no_candidate"; then
  echo "No validated candidate" >"$RUN/cross.no_candidate"
  exit 0
fi

if ! "$PY" - "$RUN/test/combined/summary.json" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
candidate = next(row for row in rows if row["variant"] != "native_rope")
passed = (
    float(candidate["paired_official_delta"]) >= 0.0
    and float(candidate["mean_gold_nll_improvement"]) > 0.0
)
print({
    "variant": candidate["variant"],
    "paired_official_delta": candidate["paired_official_delta"],
    "mean_gold_nll_improvement": candidate["mean_gold_nll_improvement"],
    "passed": passed,
})
raise SystemExit(0 if passed else 1)
PY
then
  echo "Frozen test rejected candidate" >"$RUN/cross.rejected"
  exit 0
fi

run_hotpot() {
  local out="$RUN/cross/longbench_hotpot_strict18"
  local pids=()
  mkdir -p "$out"
  for shard in 0 1 2; do
    mkdir -p "$out/shard${shard}"
    CUDA_VISIBLE_DEVICES="$shard" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_longbench_frequency_scaling.py" \
        --model-name-or-path "$MODEL" \
        --longbench-jsonl "$LONG_DATA" \
        --frozen-manifest "$FROZEN/sample_manifest.jsonl" \
        --frozen-predictions "$FROZEN/predictions.jsonl" \
        --specs-json "$SPECS" \
        --output-dir "$out/shard${shard}" \
        --max-new-tokens 32 \
        --prefill-chunk-size "$PREFILL_CHUNK_SIZE" \
        --dtype bfloat16 \
        --attn-implementation sdpa \
        --shard-count 3 --shard-index "$shard" \
        >"$out/shard${shard}/stdout.log" 2>"$out/shard${shard}/stderr.log" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$ROOT/src/merge_benchmark_shards.py" --run-dir "$out" --mode longbench \
    >"$out/merge_stdout.log" 2>"$out/merge_stderr.log"
}

run_pg19() {
  local out="$RUN/cross/pg19_ppl"
  local pids=()
  mkdir -p "$out"
  for shard in 0 1 2 3 4; do
    local gpu=$((shard + 3))
    mkdir -p "$out/shard${shard}"
    CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_pg19_frequency_ppl.py" \
        --model-name-or-path "$MODEL" \
        --pg19-parquet "$PG19" \
        --specs-json "$SPECS" \
        --output-dir "$out/shard${shard}" \
        --lengths 4096,32768 \
        --books-per-length 8 \
        --token-offset 512 \
        --score-chunk-size 128 \
        --dtype bfloat16 \
        --attn-implementation sdpa \
        --shard-count 5 --shard-index "$shard" \
        >"$out/shard${shard}/stdout.log" 2>"$out/shard${shard}/stderr.log" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$ROOT/src/merge_benchmark_shards.py" --run-dir "$out" --mode pg19 \
    >"$out/merge_stdout.log" 2>"$out/merge_stderr.log"
}

run_hotpot & p0=$!
run_pg19 & p1=$!
wait "$p0" "$p1"
touch "$RUN/cross.done"
