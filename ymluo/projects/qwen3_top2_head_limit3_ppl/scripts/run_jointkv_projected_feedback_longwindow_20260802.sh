#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
DATA="$ROOT/data/20260802_jointkv"
OUTPUT="$ROOT/results/20260802_jointkv_projected_feedback_longwindow"
CACHE="$ROOT/results/20260802_jointkv_all_layer_ppl/qwen06b_train5_cal1024_jointkv_binres48_codebooks_remote.pt"
RUNNER="$ROOT/src/run_jointkv_residual_ppl_20260802.py"
mkdir -p "$OUTPUT"

COMMON=(
  --model "$MODEL" --train_texts
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
  --store_budget_records --codebook_cache "$CACHE"
)

run_one() {
  local gpu=$1 name=$2 text=$3 history=$4 eval_tokens=$5 mode=$6
  local extra=()
  if [[ "$mode" == "rss" ]]; then
    extra+=(
      --one_shot_output_error_feedback
      --one_shot_feedback_norm projected_layer_rss
    )
  fi
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$RUNNER" "${COMMON[@]}" --test_texts "$DATA/$text" \
    --history_tokens "$history" --eval_tokens "$eval_tokens" "${extra[@]}" \
    --output "$OUTPUT/$name.json" >"$OUTPUT/$name.log" 2>&1
}

run_one 0 q06_author_4k_e64_control qksieve_author_text.txt 4096 64 control &
run_one 1 q06_author_4k_e64_feedback qksieve_author_text.txt 4096 64 rss &
run_one 2 q06_biomed_4k_e64_control biomed_long_range_facts_hard_compact.txt 4096 64 control &
run_one 3 q06_biomed_4k_e64_feedback biomed_long_range_facts_hard_compact.txt 4096 64 rss &
run_one 4 q06_biomed_8k_e32_control biomed_long_range_facts_hard_compact.txt 8192 32 control &
run_one 5 q06_biomed_8k_e32_feedback biomed_long_range_facts_hard_compact.txt 8192 32 rss &
run_one 6 q06_compiler_8k_e32_control compiler_long_range_facts_hard_compact.txt 8192 32 control &
run_one 7 q06_compiler_8k_e32_feedback compiler_long_range_facts_hard_compact.txt 8192 32 rss &
wait
