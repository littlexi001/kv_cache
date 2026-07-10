#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRANSFORMERS_VERBOSITY=error

OUT="ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_block256_topk_sweep_longbench_m3_20260707"
mkdir -p "$OUT"

python ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py \
  --output_dir "$OUT" \
  --model_name_or_path /home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter_path /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705/adapter \
  --router_path /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_safe_topk_router_v5_nonbench_20260707/router.pt \
  --longbench_data_dir ymluo/external/KVCache-Factory/data/LongBench \
  --ruler_data_dir ymluo/external/KVCache-Factory/data/RULER \
  --longbench_tasks hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,qasper,gov_report,multi_news \
  --ruler_tasks "" \
  --ruler_context_lengths "" \
  --methods full_raw,recent_plus_span_top1_b0_a0,recent_plus_span_top2_b0_a0,recent_plus_span_top3_b0_a0,recent_plus_span_top4_b0_a0,recent_plus_span_top6_b0_a0,recent_plus_span_top8_b0_a0 \
  --max_examples_per_task 3 \
  --block_tokens 256 \
  --recent_tokens 512 \
  --max_input_tokens 24000 \
  --max_new_tokens_exact 48 \
  --max_new_tokens_summary 120 \
  --dtype float16 \
  --attn_implementation sdpa \
  --device_map auto \
  --seed 2026073001 \
  > "$OUT/run_outer.log" 2>&1
