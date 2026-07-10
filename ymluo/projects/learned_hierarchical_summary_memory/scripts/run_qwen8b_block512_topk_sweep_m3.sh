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
METHODS="full_raw,recent_plus_span_top1_b0_a0,recent_plus_span_top2_b0_a0,recent_plus_span_top3_b0_a0,recent_plus_span_top4_b0_a0,recent_plus_span_top6_b0_a0,recent_plus_span_top8_b0_a0"
COMMON=(
  --model_name_or_path "$MODEL"
  --adapter_path "$ADAPTER"
  --router_path "$ROUTER"
  --longbench_data_dir ymluo/external/KVCache-Factory/data/LongBench
  --ruler_data_dir ymluo/external/KVCache-Factory/data/RULER
  --methods "$METHODS"
  --max_examples_per_task 3
  --block_tokens 512
  --recent_tokens 512
  --max_input_tokens 24000
  --max_new_tokens_exact 48
  --max_new_tokens_summary 120
  --dtype float16
  --attn_implementation sdpa
  --device_map auto
)

run_split() {
  local gpu="$1"
  local out="$2"
  shift 2
  mkdir -p "$out"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python "$SCRIPT" \
      --output_dir "$out" \
      "${COMMON[@]}" \
      "$@" \
      > "$out/run_outer.log" 2>&1
  )
}

run_split 0 "$BASE_OUT/qwen8b_block512_topk_sweep_longbench_m3_20260707" \
  --longbench_tasks hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,qasper,gov_report,multi_news \
  --ruler_tasks "" \
  --ruler_context_lengths "" \
  --seed 2026070720 &

run_split 1 "$BASE_OUT/qwen8b_block512_topk_sweep_ruler4k_m3_20260707" \
  --longbench_tasks "" \
  --ruler_tasks niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe \
  --ruler_context_lengths 4096 \
  --seed 2026070721 &

run_split 2 "$BASE_OUT/qwen8b_block512_topk_sweep_ruler8k_m3_20260707" \
  --longbench_tasks "" \
  --ruler_tasks niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe \
  --ruler_context_lengths 8192 \
  --seed 2026070722 &

run_split 3 "$BASE_OUT/qwen8b_block512_topk_sweep_ruler16k_m3_20260707" \
  --longbench_tasks "" \
  --ruler_tasks niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe \
  --ruler_context_lengths 16384 \
  --seed 2026070723 &

wait
