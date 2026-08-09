#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
RUN=${RUN:-$ROOT/outputs/multiseed_frequency_scaling_20260806/f47_distance_bf16_exactprefix}
SPECS="$RUN/specs/test.json"
OUT="$RUN/bf16_cross/longbench_hotpot_strict18"
LONG_DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench/hotpotqa.jsonl
FROZEN=/home/fdong/ymluo/projects/qwen3_longbench_oracle_evidence/outputs/hotpot_semantic_aligned_18_20260802/merged
GPU_IDS=(0 1 3)

mkdir -p "$OUT"
pids=()
for shard in 0 1 2; do
  gpu=${GPU_IDS[$shard]}
  mkdir -p "$OUT/shard${shard}"
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$PY" -u "$ROOT/src/run_longbench_frequency_scaling.py" \
    --model-name-or-path "$MODEL" \
    --longbench-jsonl "$LONG_DATA" \
    --frozen-manifest "$FROZEN/sample_manifest.jsonl" \
    --frozen-predictions "$FROZEN/predictions.jsonl" \
    --specs-json "$SPECS" \
    --output-dir "$OUT/shard${shard}" \
    --max-new-tokens 32 \
    --prefill-chunk-size 128 \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --shard-count 3 --shard-index "$shard" \
    >"$OUT/shard${shard}/stdout.log" 2>"$OUT/shard${shard}/stderr.log" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$ROOT/src/merge_benchmark_shards.py" --run-dir "$OUT" --mode longbench \
  >"$OUT/merge_stdout.log" 2>"$OUT/merge_stderr.log"
touch "$OUT/stage.done"
