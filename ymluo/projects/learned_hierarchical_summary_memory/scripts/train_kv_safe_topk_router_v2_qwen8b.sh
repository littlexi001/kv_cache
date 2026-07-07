#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong

python ymluo/projects/learned_hierarchical_summary_memory/src/run_fast_recent_plus_router_training.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_safe_topk_router_v2_nonbench_20260707 \
  --model_name_or_path /home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --text_paths /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt,/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/data/count_monte_cristo_pg1184.txt \
  --dataset_names warpeace,montecristo \
  --benchmark_output_dir /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_topk_actions_small_20260707 \
  --candidate_methods full_raw,recent_plus_summary1_8,recent_plus_summary1_4,recent_plus_retrieval_raw_k2,recent_plus_span_top2_b0_a0,recent_plus_span_top3_b0_a0,recent_plus_prefix_to_farthest_top3,recent_plus_full_old_raw \
  --cases_per_dataset 360 \
  --prefill_token_lengths 4096,8192,16384,20000 \
  --sample_stride_tokens 384 \
  --eval_start_tokens 12000 \
  --block_tokens 1024 \
  --recent_tokens 512 \
  --max_text_tokens 260000 \
  --max_input_tokens 24000 \
  --policy kv_safe_topk_v2 \
  --hidden_dim 128 \
  --epochs 1200 \
  --lr 0.002 \
  --weight_decay 0.0001 \
  --test_fraction 0.25 \
  --seed 2026070705

