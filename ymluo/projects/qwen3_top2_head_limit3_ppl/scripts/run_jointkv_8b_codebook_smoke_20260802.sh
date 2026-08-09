#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DATA="$ROOT/data/20260802_jointkv"
OUTPUT="$ROOT/results/20260802_jointkv_8b"
mkdir -p "$OUTPUT"

exec env CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" "$ROOT/src/run_jointkv_residual_ppl_20260802.py" \
  --model "$MODEL" \
  --train_texts \
  "$ROOT/data/war_and_peace_pg2600.txt" \
  "$ROOT/data/count_monte_cristo_pg1184.txt" \
  "$DATA/long_textbook_distributed_systems.txt" \
  "$DATA/long_news_supply_chain_dossier.txt" \
  "$DATA/long_dialogue_tool_transcript.txt" \
  --test_texts "$DATA/qksieve_author_text.txt" \
  --calibration_tokens 1024 \
  --history_tokens 1024 \
  --eval_tokens 2 \
  --query_samples_per_text 256 \
  --key_samples_per_text 512 \
  --fraction 0.20 \
  --recent_tokens 128 \
  --sink_tokens 4 \
  --sparse_layers all \
  --binary_bits 64 \
  --projection_iterations 6 \
  --residual_vq_bits 6 \
  --residual_vq_iterations 6 \
  --residual_binary_bits 48 \
  --residual_binary_iterations 6 \
  --residual_binary_candidate_fraction 1.0 \
  --joint_value_weight 0.5 \
  --risk_lambda 1.0 \
  --priority_mode output_bound \
  --risk_error_bits 4 \
  --risk_error_block_size 256 \
  --metric_shrinkage oas \
  --value_mean_bits 4 \
  --refit_key_bits 8 \
  --tail_mode joint_tail \
  --device cuda \
  --dtype float16 \
  --threads 1 \
  --codebook_cache "$OUTPUT/qwen8b_train5_cal1024_jointkv_binres48_codebooks.pt" \
  --output "$OUTPUT/codebook_smoke_h1024_e2_f020.json"
