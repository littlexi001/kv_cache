#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
DATA="$ROOT/data/20260802_jointkv"
OUTPUT="$ROOT/results/20260802_jointkv_all_layer_ppl/remote_4k_fisher"
CACHE="$ROOT/results/20260802_jointkv_all_layer_ppl/qwen06b_train5_cal1024_jointkv_binres48_codebooks_remote.pt"
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
  --calibration_tokens 1024 --history_tokens 4096 --eval_tokens 16
  --query_samples_per_text 256 --key_samples_per_text 512
  --fraction 0.02 --recent_tokens 128 --sink_tokens 4
  --sparse_layers all --binary_bits 64 --projection_iterations 6
  --residual_vq_bits 6 --residual_vq_iterations 6
  --residual_binary_bits 48 --residual_binary_iterations 6
  --residual_binary_candidate_fraction 1.0
  --adaptive_action_mode full_residual
  --joint_value_weight 0.5 --risk_lambda 1.0
  --priority_mode output_bound --risk_error_bits 4
  --risk_error_block_size 256 --metric_shrinkage oas
  --value_mean_bits 4 --refit_key_bits 8 --tail_mode joint_tail
  --head_sensitivity_mode one_shot_fisher
  --device cuda --dtype float16 --threads 1 --codebook_cache "$CACHE"
)

run_one() {
  local gpu=$1 name=$2 tolerance=$3 probes=$4
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "${COMMON[@]}" \
    --adaptive_error_tolerance "$tolerance" \
    --head_sensitivity_probes "$probes" \
    --output "$OUTPUT/$name.json" >"$OUTPUT/$name.log" 2>&1
}

run_one 0 tau0025_probe1 0.025 1 &
run_one 1 tau0025_probe2 0.025 2 &
run_one 2 tau0035_probe1 0.035 1 &
run_one 3 tau0035_probe2 0.035 2 &
wait
