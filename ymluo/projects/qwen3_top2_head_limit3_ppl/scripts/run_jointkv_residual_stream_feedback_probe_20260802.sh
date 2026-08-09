#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL06=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
MODEL8=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DATA="$ROOT/data/20260802_jointkv"
OUTPUT="$ROOT/results/20260802_jointkv_residual_stream_feedback_probe"
CACHE06="$ROOT/results/20260802_jointkv_all_layer_ppl/qwen06b_train5_cal1024_jointkv_binres48_codebooks_remote.pt"
CACHE8="$ROOT/results/20260802_jointkv_8b/qwen8b_train5_cal1024_jointkv_binres48_codebooks.pt"
RUNNER="$ROOT/src/run_jointkv_residual_ppl_20260802.py"
mkdir -p "$OUTPUT"

COMMON=(
  --train_texts
  "$ROOT/data/war_and_peace_pg2600.txt"
  "$ROOT/data/count_monte_cristo_pg1184.txt"
  "$DATA/long_textbook_distributed_systems.txt"
  "$DATA/long_news_supply_chain_dossier.txt"
  "$DATA/long_dialogue_tool_transcript.txt"
  --calibration_tokens 1024 --query_samples_per_text 256 --key_samples_per_text 512
  --fraction 0.02 --recent_tokens 128 --sink_tokens 4 --sparse_layers all
  --binary_bits 64 --projection_iterations 6
  --residual_vq_bits 6 --residual_vq_iterations 6
  --residual_binary_bits 48 --residual_binary_iterations 6
  --residual_binary_candidate_fraction 1.0
  --adaptive_error_tolerance 0.05 --adaptive_action_mode full_residual
  --adaptive_budget_coupling previous_layer_quantile --adaptive_budget_quantile -1
  --joint_value_weight 0.5 --risk_lambda 1.0 --priority_mode output_bound
  --risk_error_bits 4 --risk_error_block_size 256 --metric_shrinkage oas
  --value_mean_bits 4 --refit_key_bits 8 --tail_mode joint_tail
  --device cuda --dtype float16 --attn_implementation sdpa --threads 1
  --store_budget_records --one_shot_output_error_feedback
  --one_shot_feedback_norm residual_stream_rss
)

run_one() {
  local gpu=$1 name=$2 model=$3 cache=$4 text=$5 history=$6 eval_tokens=$7
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$RUNNER" --model "$model" "${COMMON[@]}" \
    --codebook_cache "$cache" --test_texts "$DATA/$text" \
    --history_tokens "$history" --eval_tokens "$eval_tokens" \
    --output "$OUTPUT/$name.json" >"$OUTPUT/$name.log" 2>&1
}

run_one 4 q06_author_4k_e16_residual "$MODEL06" "$CACHE06" qksieve_author_text.txt 4096 16 &
run_one 5 q06_biomed_8k_e16_residual "$MODEL06" "$CACHE06" biomed_long_range_facts_hard_compact.txt 8192 16 &
run_one 6 q8_author_4k_e16_residual "$MODEL8" "$CACHE8" qksieve_author_text.txt 4096 16 &
run_one 7 q8_author_8k_e8_residual "$MODEL8" "$CACHE8" qksieve_author_text.txt 8192 8 &
wait
