#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DATA="$ROOT/data/20260802_jointkv"
OUTPUT="$ROOT/results/20260802_jointkv_8b/leaveoneout_probe"
CACHE="$ROOT/results/20260802_jointkv_8b/qwen8b_train5_cal1024_jointkv_binres48_codebooks.pt"
RUNNER="$ROOT/src/run_jointkv_residual_ppl_20260802.py"
mkdir -p "$OUTPUT"

COMMON=(
  "$RUNNER" --model "$MODEL"
  --train_texts
  "$ROOT/data/war_and_peace_pg2600.txt"
  "$ROOT/data/count_monte_cristo_pg1184.txt"
  "$DATA/long_textbook_distributed_systems.txt"
  "$DATA/long_news_supply_chain_dossier.txt"
  "$DATA/long_dialogue_tool_transcript.txt"
  --test_texts "$DATA/qksieve_author_text.txt"
  --calibration_tokens 1024 --query_samples_per_text 256 --key_samples_per_text 512
  --recent_tokens 128 --sink_tokens 4 --sparse_layers all
  --binary_bits 64 --projection_iterations 6
  --residual_vq_bits 6 --residual_vq_iterations 6
  --residual_binary_bits 48 --residual_binary_iterations 6
  --residual_binary_candidate_fraction 1.0
  --joint_value_weight 0.5 --risk_lambda 1.0
  --priority_mode output_bound --risk_error_bits 4
  --risk_error_block_size 256 --metric_shrinkage oas
  --value_mean_bits 4 --refit_key_bits 8 --tail_mode joint_tail
  --device cuda --dtype float16 --attn_implementation sdpa --threads 1
  --codebook_cache "$CACHE"
)

run_one() {
  local gpu=$1 name=$2 history=$3 eval_tokens=$4
  shift 4
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "${COMMON[@]}" --history_tokens "$history" \
    --eval_tokens "$eval_tokens" "$@" --output "$OUTPUT/$name.json" \
    >"$OUTPUT/$name.log" 2>&1
}

run_one 0 leaveoneout_1k_e32 1024 32 \
  --fraction 0.02 --adaptive_error_tolerance 0.05 \
  --adaptive_action_mode full_residual \
  --adaptive_budget_coupling previous_layer_quantile --adaptive_budget_quantile -1 &
run_one 1 leaveoneout_4k_e16 4096 16 \
  --fraction 0.02 --adaptive_error_tolerance 0.05 \
  --adaptive_action_mode full_residual \
  --adaptive_budget_coupling previous_layer_quantile --adaptive_budget_quantile -1 &
run_one 2 leaveoneout_8k_e8 8192 8 \
  --fraction 0.02 --adaptive_error_tolerance 0.05 \
  --adaptive_action_mode full_residual \
  --adaptive_budget_coupling previous_layer_quantile --adaptive_budget_quantile -1 &
run_one 3 fixed20_4k_e16 4096 16 \
  --fraction 0.20 --adaptive_error_tolerance 0.0 &
run_one 4 fixed10_4k_e16 4096 16 \
  --fraction 0.10 --adaptive_error_tolerance 0.0 &
run_one 5 independent_4k_e16 4096 16 \
  --fraction 0.02 --adaptive_error_tolerance 0.05 \
  --adaptive_action_mode full_residual --adaptive_budget_coupling independent &
wait
