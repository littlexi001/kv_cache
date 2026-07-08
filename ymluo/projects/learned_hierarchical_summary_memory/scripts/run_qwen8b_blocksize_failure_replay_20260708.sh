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
RUN_TAG="${RUN_TAG:-blocksize_failure_replay_20260708}"
GPU_LIST="${GPU_LIST:-4 5}"
GPU_LIST_CSV="${GPU_LIST_CSV:-}"
METHODS="${METHODS:-full_raw,recent_plus_b128_span_top12_b0_a0,recent_plus_b128_span_top16_b0_a0,recent_plus_b256_span_top3_b0_a0,recent_plus_b256_span_top4_b0_a0,recent_plus_b256_span_top8_b0_a0,recent_plus_b256_span_top12_b0_a0,recent_plus_b512_span_top3_b0_a0,recent_plus_b512_span_top4_b0_a0,recent_plus_b512_span_top8_b0_a0}"

COMMON=(
  --model_name_or_path "$MODEL"
  --adapter_path "$ADAPTER"
  --router_path "$ROUTER"
  --longbench_data_dir ymluo/external/KVCache-Factory/data/LongBench
  --ruler_data_dir ymluo/external/KVCache-Factory/data/RULER
  --methods "$METHODS"
  --max_examples_per_task 20
  --case_ids 11,14,17
  --block_tokens 512
  --recent_tokens 512
  --max_input_tokens 24000
  --max_new_tokens_exact 48
  --max_new_tokens_summary 120
  --dtype float16
  --attn_implementation sdpa
  --device_map auto
  --longbench_tasks ""
)

if [[ -n "$GPU_LIST_CSV" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST_CSV"
else
  read -r -a GPUS <<< "$GPU_LIST"
fi

run_replay() {
  local gpu="$1"
  local length="$2"
  local tasks="$3"
  local seed="$4"
  local out="$BASE_OUT/qwen8b_${RUN_TAG}_ruler${length}"
  mkdir -p "$out"
  echo "START replay length=$length gpu=$gpu out=$out tasks=$tasks case_ids=11,14,17 methods=$METHODS $(date -Is)"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python "$SCRIPT" \
      --output_dir "$out" \
      "${COMMON[@]}" \
      --seed "$seed" \
      --ruler_tasks "$tasks" \
      --ruler_context_lengths "$length" \
      > "$out/run_outer.log" 2>&1
    echo "DONE replay length=$length gpu=$gpu $(date -Is)" | tee -a "$out/run_outer.log"
  )
}

gpu4="${GPUS[0]}"
gpu8="${GPUS[$((1 % ${#GPUS[@]}))]}"
run_replay "$gpu4" 4096 niah_single_1 2026070861 &
run_replay "$gpu8" 8192 niah_single_2,vt 2026070862 &
wait

echo "ALL BLOCKSIZE FAILURE REPLAY JOBS DONE $(date -Is)"
