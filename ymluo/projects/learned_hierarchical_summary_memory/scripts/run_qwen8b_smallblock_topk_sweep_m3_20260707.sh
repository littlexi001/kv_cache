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
ROUTER="/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_safe_topk_router_v5_nonbench_20260707/router.pt"

BLOCK_SIZES="${BLOCK_SIZES:-32 64 128}"
GPU_LIST="${GPU_LIST:-0 1 2 4 5 6 7}"
METHODS="${METHODS:-full_raw,recent_plus_span_top1_b0_a0,recent_plus_span_top2_b0_a0,recent_plus_span_top3_b0_a0,recent_plus_span_top4_b0_a0,recent_plus_span_top6_b0_a0,recent_plus_span_top8_b0_a0,recent_plus_span_top12_b0_a0,recent_plus_span_top16_b0_a0,recent_plus_span_top24_b0_a0,recent_plus_span_top32_b0_a0}"

COMMON=(
  --model_name_or_path "$MODEL"
  --adapter_path "$ADAPTER"
  --router_path "$ROUTER"
  --longbench_data_dir ymluo/external/KVCache-Factory/data/LongBench
  --ruler_data_dir ymluo/external/KVCache-Factory/data/RULER
  --methods "$METHODS"
  --max_examples_per_task 3
  --recent_tokens 512
  --max_input_tokens 24000
  --max_new_tokens_exact 48
  --max_new_tokens_summary 120
  --dtype float16
  --attn_implementation sdpa
  --device_map auto
)

declare -a JOBS=()
for block in $BLOCK_SIZES; do
  JOBS+=("$block|longbench|qwen8b_block${block}_topk_sweep_longbench_m3_20260707|$((2026070900 + block + 1))")
  JOBS+=("$block|ruler4k|qwen8b_block${block}_topk_sweep_ruler4k_m3_20260707|$((2026070900 + block + 2))")
  JOBS+=("$block|ruler8k|qwen8b_block${block}_topk_sweep_ruler8k_m3_20260707|$((2026070900 + block + 3))")
  JOBS+=("$block|ruler16k|qwen8b_block${block}_topk_sweep_ruler16k_m3_20260707|$((2026070900 + block + 4))")
done

read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 1
fi

run_job() {
  local gpu="$1"
  local spec="$2"
  IFS='|' read -r block group out_name seed <<< "$spec"
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
  echo "START block=$block group=$group gpu=$gpu out=$out $(date -Is)"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python "$SCRIPT" \
      --output_dir "$out" \
      "${COMMON[@]}" \
      --block_tokens "$block" \
      --seed "$seed" \
      "${group_args[@]}" \
      > "$out/run_outer.log" 2>&1
    echo "DONE block=$block group=$group gpu=$gpu $(date -Is)" | tee -a "$out/run_outer.log"
  )
}

batch_size="${#GPUS[@]}"
job_idx=0
for spec in "${JOBS[@]}"; do
  gpu="${GPUS[$((job_idx % batch_size))]}"
  run_job "$gpu" "$spec" &
  job_idx=$((job_idx + 1))
  if (( job_idx % batch_size == 0 )); then
    wait
  fi
done
wait

echo "ALL SMALL-BLOCK SWEEPS DONE $(date -Is)"
