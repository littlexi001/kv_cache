#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/home/fdong/ymluo/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/fdong/ymluo/hf_cache}"
export TOKENIZERS_PARALLELISM=false

python ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen3_8b_paper_benchmarks_overnight_20260704 \
  --model_name_or_path Qwen/Qwen3-8B \
  --longbench_tasks hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,qasper,gov_report,multi_news \
  --ruler_tasks niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe \
  --ruler_context_lengths 4096,8192,16384 \
  --max_examples_per_task 5 \
  --methods full_raw,summary10,summary100,summary1000,static_hier,retrieval_raw_k1,retrieval_raw_k2,router,router_conservative \
  --block_tokens 1024 \
  --recent_tokens 512 \
  --max_input_tokens 12000 \
  --max_new_tokens_exact 48 \
  --max_new_tokens_summary 160 \
  --device_map cuda \
  --dtype float16 \
  --attn_implementation sdpa
