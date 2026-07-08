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
RUN_TAG="${RUN_TAG:-blocksize_floor_targeted_m20_20260708}"
GPU_LIST="${GPU_LIST:-2 3}"
GPU_LIST_CSV="${GPU_LIST_CSV:-}"
METHODS="${METHODS:-full_raw,recent_plus_b128_span_top12_b0_a0,recent_plus_b128_span_top16_b0_a0,recent_plus_b256_span_top3_b0_a0,recent_plus_b256_span_top4_b0_a0,recent_plus_b256_span_top8_b0_a0,recent_plus_b256_span_top12_b0_a0,recent_plus_b512_span_top3_b0_a0,recent_plus_b512_span_top4_b0_a0,recent_plus_b512_span_top8_b0_a0}"
MAX_EXAMPLES_PER_TASK="${MAX_EXAMPLES_PER_TASK:-20}"
TARGET_RULER_TASKS="${TARGET_RULER_TASKS:-niah_single_1,niah_single_2,vt}"

COMMON=(
  --model_name_or_path "$MODEL"
  --adapter_path "$ADAPTER"
  --router_path "$ROUTER"
  --longbench_data_dir ymluo/external/KVCache-Factory/data/LongBench
  --ruler_data_dir ymluo/external/KVCache-Factory/data/RULER
  --methods "$METHODS"
  --max_examples_per_task "$MAX_EXAMPLES_PER_TASK"
  --block_tokens 512
  --recent_tokens 512
  --max_input_tokens 24000
  --max_new_tokens_exact 48
  --max_new_tokens_summary 120
  --dtype float16
  --attn_implementation sdpa
  --device_map auto
  --longbench_tasks ""
  --ruler_tasks "$TARGET_RULER_TASKS"
)

if [[ -n "$GPU_LIST_CSV" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST_CSV"
else
  read -r -a GPUS <<< "$GPU_LIST"
fi

run_length() {
  local gpu="$1"
  local length="$2"
  local seed="$3"
  local out="$BASE_OUT/qwen8b_${RUN_TAG}_ruler${length}"
  mkdir -p "$out"
  echo "START targeted length=$length gpu=$gpu out=$out tasks=$TARGET_RULER_TASKS methods=$METHODS max_examples=$MAX_EXAMPLES_PER_TASK $(date -Is)"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python "$SCRIPT" \
      --output_dir "$out" \
      "${COMMON[@]}" \
      --seed "$seed" \
      --ruler_context_lengths "$length" \
      > "$out/run_outer.log" 2>&1
    echo "DONE targeted length=$length gpu=$gpu $(date -Is)" | tee -a "$out/run_outer.log"
  )
}

gpu4="${GPUS[0]}"
gpu8="${GPUS[$((1 % ${#GPUS[@]}))]}"
run_length "$gpu4" 4096 2026070851 &
run_length "$gpu8" 8192 2026070852 &
wait

echo "ALL BLOCKSIZE FLOOR TARGETED JOBS DONE $(date -Is)"
