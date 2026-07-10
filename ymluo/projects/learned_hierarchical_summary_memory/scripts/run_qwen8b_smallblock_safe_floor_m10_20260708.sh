#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong

export TRANSFORMERS_VERBOSITY=error

BASE_OUT="ymluo/projects/learned_hierarchical_summary_memory/outputs"
SCRIPT="ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py"
MODEL="/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER="/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705/adapter"
ROUTER="${ROUTER:-/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/smallblock_router_from_sweeps_m3_20260707/router.pt}"
RUN_TAG="${RUN_TAG:-smallblock_safe_floor_m10_20260708}"
GPU_LIST="${GPU_LIST:-4 5 6 7}"

COMMON=(
  --model_name_or_path "$MODEL"
  --adapter_path "$ADAPTER"
  --router_path "$ROUTER"
  --longbench_data_dir ymluo/external/KVCache-Factory/data/LongBench
  --ruler_data_dir ymluo/external/KVCache-Factory/data/RULER
  --max_examples_per_task 10
  --block_tokens 512
  --recent_tokens 512
  --max_input_tokens 24000
  --max_new_tokens_exact 48
  --max_new_tokens_summary 120
  --dtype float16
  --attn_implementation sdpa
  --device_map auto
)

declare -a JOBS=(
  "longbench|qwen8b_${RUN_TAG}_longbench|2026070821|full_raw,recent_plus_b128_span_top6_b0_a0,recent_plus_b128_span_top12_b0_a0,recent_plus_b256_span_top4_b0_a0,router_blocksize"
  "ruler4k|qwen8b_${RUN_TAG}_ruler4k|2026070822|full_raw,recent_plus_b128_span_top3_b0_a0,recent_plus_b256_span_top3_b0_a0,router_blocksize"
  "ruler8k|qwen8b_${RUN_TAG}_ruler8k|2026070823|full_raw,recent_plus_b128_span_top3_b0_a0,recent_plus_b256_span_top3_b0_a0,router_blocksize"
  "ruler16k|qwen8b_${RUN_TAG}_ruler16k|2026070824|full_raw,recent_plus_b64_span_top4_b0_a0,recent_plus_b256_span_top3_b0_a0,recent_plus_b512_span_top3_b0_a0,router_blocksize"
)

read -r -a GPUS <<< "$GPU_LIST"

run_job() {
  local gpu="$1"
  local spec="$2"
  IFS='|' read -r group out_name seed methods <<< "$spec"
  local out="$BASE_OUT/$out_name"
  local group_args=()
  case "$group" in
    longbench)
      group_args=(
        --longbench_tasks hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,qasper,gov_report,multi_news
        --ruler_tasks ""
        --ruler_context_lengths ""
      )
      ;;
    ruler4k)
      group_args=(
        --longbench_tasks ""
        --ruler_tasks niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe
        --ruler_context_lengths 4096
      )
      ;;
    ruler8k)
      group_args=(
        --longbench_tasks ""
        --ruler_tasks niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe
        --ruler_context_lengths 8192
      )
      ;;
    ruler16k)
      group_args=(
        --longbench_tasks ""
        --ruler_tasks niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe
        --ruler_context_lengths 16384
      )
      ;;
    *)
      echo "unknown group: $group" >&2
      exit 1
      ;;
  esac
  mkdir -p "$out"
  echo "START group=$group gpu=$gpu out=$out methods=$methods $(date -Is)"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python "$SCRIPT" \
      --output_dir "$out" \
      "${COMMON[@]}" \
      --methods "$methods" \
      --seed "$seed" \
      "${group_args[@]}" \
      > "$out/run_outer.log" 2>&1
    echo "DONE group=$group gpu=$gpu $(date -Is)" | tee -a "$out/run_outer.log"
  )
}

idx=0
for spec in "${JOBS[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_job "$gpu" "$spec" &
  idx=$((idx + 1))
done
wait

echo "ALL SMALLBLOCK SAFE-FLOOR M10 JOBS DONE $(date -Is)"
