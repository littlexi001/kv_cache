#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

python ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_actions_small_20260706 \
  --model_name_or_path /home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter_path /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705/adapter \
  --longbench_data_dir ymluo/external/KVCache-Factory/data/LongBench \
  --ruler_data_dir ymluo/external/KVCache-Factory/data/RULER \
  --longbench_tasks hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,qasper,gov_report,multi_news \
  --ruler_tasks niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe \
  --ruler_context_lengths 4096,8192,16384 \
  --methods full_raw,recent_plus_summary1_8,recent_plus_retrieval_raw_k2,recent_plus_retrieval_raw_k3,recent_plus_prefix_to_evidence,recent_plus_span_b0_a0,recent_plus_span_b1_a0,recent_plus_span_b1_a1,recent_plus_full_old_raw \
  --max_examples_per_task 1 \
  --block_tokens 1024 \
  --recent_tokens 512 \
  --max_input_tokens 24000 \
  --max_new_tokens_exact 48 \
  --max_new_tokens_summary 120 \
  --dtype float16 \
  --attn_implementation sdpa \
  --device_map auto \
  --seed 2026070608
