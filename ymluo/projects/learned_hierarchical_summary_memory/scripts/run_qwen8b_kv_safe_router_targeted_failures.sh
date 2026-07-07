#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

python ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_router_targeted_failures_20260707 \
  --model_name_or_path /home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter_path /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705/adapter \
  --router_path /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_safe_topk_router_v3_nonbench_20260707/router.pt \
  --longbench_data_dir ymluo/external/KVCache-Factory/data/LongBench \
  --ruler_data_dir ymluo/external/KVCache-Factory/data/RULER \
  --longbench_tasks hotpotqa \
  --ruler_tasks niah_multikey_1 \
  --ruler_context_lengths 16384 \
  --methods full_raw,router_safe,recent_plus_retrieval_raw_k2,recent_plus_retrieval_raw_k3,recent_plus_prefix_to_farthest_top3,recent_plus_full_old_raw \
  --max_examples_per_task 2 \
  --block_tokens 1024 \
  --recent_tokens 512 \
  --max_input_tokens 24000 \
  --max_new_tokens_exact 48 \
  --max_new_tokens_summary 120 \
  --dtype float16 \
  --attn_implementation sdpa \
  --device_map auto \
  --seed 2026070710

