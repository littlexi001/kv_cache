#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
LONG_DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench/hotpotqa.jsonl
LONG_FROZEN=/home/fdong/ymluo/projects/qwen3_longbench_oracle_evidence/outputs/hotpot_semantic_aligned_18_20260802/merged
PG19=/home/fdong/ymluo/datasets/pg19/test.parquet
PARENT=${1:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
RUN=$PARENT/cross_benchmarks
SPECS=$PARENT/specs/test.json
mkdir -p "$RUN"

on_exit() {
  rc=$?
  if test "$rc" -ne 0; then touch "$RUN/launcher.failed"; fi
}
trap on_exit EXIT

while ! test -f "$PARENT/test.done"; do
  if test -f "$PARENT/launcher.failed"; then
    echo "RULER launcher failed; refusing to benchmark an unfrozen spec" >&2
    exit 1
  fi
  sleep 30
done

run_bf16_smoke() {
  local outroot="$RUN/bf16_smoke"
  if test -f "$outroot/stage.done" || test -f "$outroot/stage.failed"; then return; fi
  mkdir -p "$outroot/shard0"
  "$PY" - "$SPECS" "$outroot/specs.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding='utf-8'))
value['specs']=value['specs'][:2]
json.dump(value, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
PY
  if CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
    --model-name-or-path "$MODEL" \
    --examples-jsonl "$PARENT/data/ruler32k_seed45_m2.jsonl" \
    --specs-json "$outroot/specs.json" \
    --output-dir "$outroot/shard0" \
    --sample-ids niah_multikey_3_32768_0 \
    --target-length 32768 \
    --max-new-tokens-cap 32 \
    --prefill-chunk-size 128 \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    >"$outroot/stdout.log" 2>"$outroot/stderr.log"; then
    touch "$outroot/stage.done"
  else
    touch "$outroot/stage.failed"
  fi
}

run_longbench() {
  local outroot="$RUN/longbench_hotpot_strict18"
  if test -f "$outroot/stage.done"; then return; fi
  mkdir -p "$outroot"
  local pids=()
  for shard in $(seq 0 7); do
    mkdir -p "$outroot/shard$shard"
    CUDA_VISIBLE_DEVICES="$shard" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_longbench_frequency_scaling.py" \
      --model-name-or-path "$MODEL" \
      --longbench-jsonl "$LONG_DATA" \
      --frozen-manifest "$LONG_FROZEN/sample_manifest.jsonl" \
      --frozen-predictions "$LONG_FROZEN/predictions.jsonl" \
      --specs-json "$SPECS" \
      --output-dir "$outroot/shard$shard" \
      --max-new-tokens 32 \
      --prefill-chunk-size 256 \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --load-in-4bit \
      --shard-count 8 --shard-index "$shard" \
      >"$outroot/shard$shard/stdout.log" 2>"$outroot/shard$shard/stderr.log" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$ROOT/src/merge_benchmark_shards.py" --run-dir "$outroot" --mode longbench \
    >"$outroot/merge_stdout.log" 2>"$outroot/merge_stderr.log"
  touch "$outroot/stage.done"
}

run_pg19() {
  local outroot="$RUN/pg19_ppl"
  if test -f "$outroot/stage.done"; then return; fi
  mkdir -p "$outroot"
  local pids=()
  for shard in $(seq 0 7); do
    mkdir -p "$outroot/shard$shard"
    CUDA_VISIBLE_DEVICES="$shard" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_pg19_frequency_ppl.py" \
      --model-name-or-path "$MODEL" \
      --pg19-parquet "$PG19" \
      --specs-json "$SPECS" \
      --output-dir "$outroot/shard$shard" \
      --lengths 4096,32768 \
      --books-per-length 8 \
      --token-offset 512 \
      --score-chunk-size 256 \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --load-in-4bit \
      --shard-count 8 --shard-index "$shard" \
      >"$outroot/shard$shard/stdout.log" 2>"$outroot/shard$shard/stderr.log" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$ROOT/src/merge_benchmark_shards.py" --run-dir "$outroot" --mode pg19 \
    >"$outroot/merge_stdout.log" 2>"$outroot/merge_stderr.log"
  touch "$outroot/stage.done"
}

run_longbench_e_panel() {
  local outroot="$RUN/longbench_e_panel"
  if test -f "$outroot/stage.done"; then return; fi
  mkdir -p "$outroot"
  local pids=()
  for shard in $(seq 0 7); do
    mkdir -p "$outroot/shard$shard"
    CUDA_VISIBLE_DEVICES="$shard" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_longbench_e_panel_frequency_scaling.py" \
      --model-name-or-path "$MODEL" \
      --longbench-dir /home/fdong/ymluo/external/KVCache-Factory/data/LongBench \
      --longbench-code-root /home/fdong/ymluo/external/KVCache-Factory \
      --specs-json "$SPECS" \
      --output-dir "$outroot/shard$shard" \
      --datasets qasper,multifieldqa_en,hotpotqa,2wikimqa,passage_retrieval_en,lcc \
      --samples-per-task 6 \
      --selection-seed 20260806 \
      --max-prompt-tokens 39000 \
      --prefill-chunk-size 256 \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --load-in-4bit \
      --shard-count 8 --shard-index "$shard" \
      >"$outroot/shard$shard/stdout.log" 2>"$outroot/shard$shard/stderr.log" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$ROOT/src/merge_benchmark_shards.py" --run-dir "$outroot" --mode longbench_e \
    >"$outroot/merge_stdout.log" 2>"$outroot/merge_stderr.log"
  touch "$outroot/stage.done"
}

run_bf16_smoke
run_longbench
run_longbench_e_panel
run_pg19
touch "$RUN/launcher.done"
trap - EXIT
