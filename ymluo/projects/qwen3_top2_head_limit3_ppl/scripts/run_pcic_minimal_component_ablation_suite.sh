#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
HARD=${HARD:-/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt}
RUN_EXPERIMENTS=${RUN_EXPERIMENTS:-0}
GPU_COUNT=${GPU_COUNT:-8}
ANCHOR_ACCEPT_MARGIN=${ANCHOR_ACCEPT_MARGIN:-0.012}
ONLY_CASES=${ONLY_CASES:-}

cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs docs

should_run_case() {
  local name=$1
  if [[ -z "$ONLY_CASES" ]]; then
    return 0
  fi
  local pattern
  for pattern in $ONLY_CASES; do
    if [[ "$name" == *"$pattern"* ]]; then
      return 0
    fi
  done
  return 1
}

run_case() {
  local gpu=$1
  local name=$2
  local text=$3
  local blocks=$4
  local eval_tokens=$5
  local sentinel_tokens=$6
  local initial_tokens=$7
  local combos=$8
  shift 8
  local out="outputs/${name}"
  if ! should_run_case "$name"; then
    echo "[filter-skip] $name ONLY_CASES=$ONLY_CASES"
    return 0
  fi
  if [[ -f "$out/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  if [[ "$RUN_EXPERIMENTS" != "1" ]]; then
    echo "[dry-run-missing] $name -> set RUN_EXPERIMENTS=1 to run"
    return 0
  fi
  if [[ ! -f "$text" ]]; then
    echo "[missing-text] $text"
    return 1
  fi
  echo "[start] gpu=$gpu name=$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/run_pcic_rescue_blockwise_local.py \
    --model_name_or_path "$MODEL" \
    --text_path "$text" \
    --output_dir "$out" \
    --prefill_tokens 2048 \
    --num_blocks "$blocks" \
    --calibration_tokens 16 \
    --eval_tokens_per_block "$eval_tokens" \
    --dtype float16 \
    --device cuda:0 \
    --attn_implementation eager \
    --recent_tokens 512 \
    --landmark_stride 64 \
    --combos "$combos" \
    --rescue_strategy none \
    --sentinel_tokens "$sentinel_tokens" \
    --sentinel_cascade_initial_tokens "$initial_tokens" \
    --sentinel_cascade_accept_margin "$ANCHOR_ACCEPT_MARGIN" \
    --sentinel_cascade_extend_topk 2 \
    --sentinel_loss_slack 0.0 \
    --sentinel_all_min_margin 0.0 \
    --horizon_gate_min_gain 0.0 \
    --horizon_gate_min_ratio 0.0 \
    --horizon_gate_uncertainty_floor 0.005 \
    "$@" \
    > "outputs/logs/${name}.log" 2>&1
  echo "[done] $name"
}

HARD_COMBOS='0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'
MONTE_COMBOS='2,0,7,12;7,13;2,7;2,0'
RULER_VAR=${RULER_VAR:-data/pcic_ruler_style_variable_eval_2026_06_29.txt}

gpu=0

# no validation-prior anchor: still uses pairwise/horizon top-k rescue.
run_case "$gpu" server_pcic_ablate_hard_noanchor_top2_eval128_seed64_eager \
  "$HARD" 4 128 128 64 "$HARD_COMBOS" \
  --combo_select_policy risk_memory_horizon_gate \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 64
gpu=$(((gpu + 1) % GPU_COUNT))

# strict no-rescue: memory selector only, no sentinel/horizon candidate arbitration.
run_case "$gpu" server_pcic_ablate_hard_memoryonly_eval128_seed64_eager \
  "$HARD" 4 128 128 64 "$HARD_COMBOS" \
  --combo_select_policy risk_memory \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 64
gpu=$(((gpu + 1) % GPU_COUNT))

# main method: conditional validation-prior anchor.
run_case "$gpu" server_pcic_ablate_hard_condanchor_eval128_seed64_eager \
  "$HARD" 4 128 128 64 "$HARD_COMBOS" \
  --combo_select_policy risk_memory_horizon_gate \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 64 \
  --sentinel_cascade_anchor_combos '0,6'
gpu=$(((gpu + 1) % GPU_COUNT))

# no-history-memory: same selector, but without prefill-tail memory seed.
run_case "$gpu" server_pcic_ablate_hard_nohistory_eval128_seed64_eager \
  "$HARD" 4 128 128 64 "$HARD_COMBOS" \
  --combo_select_policy risk_memory_horizon_gate \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 0 \
  --risk_memory_use_history false \
  --sentinel_cascade_anchor_combos '0,6'
gpu=$(((gpu + 1) % GPU_COUNT))

# no-pairwise proxy: calibration min-loss selector without sentinel pairwise/horizon arbitration.
run_case "$gpu" server_pcic_ablate_hard_nopairwise_eval128_seed64_eager \
  "$HARD" 4 128 128 64 "$HARD_COMBOS" \
  --combo_select_policy risk_memory_horizon_gate \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 64 \
  --pairwise_candidate_probe false \
  --sentinel_cascade_anchor_combos '0,6'
gpu=$(((gpu + 1) % GPU_COUNT))

run_case "$gpu" server_pcic_ablate_hard_minloss_eval128_seed64_eager \
  "$HARD" 4 128 128 64 "$HARD_COMBOS" \
  --combo_select_policy min_loss
gpu=$(((gpu + 1) % GPU_COUNT))

run_case "$gpu" server_pcic_ablate_monte_noanchor_seed64_eager \
  data/count_monte_cristo_pg1184.txt 2 64 64 32 "$MONTE_COMBOS" \
  --combo_select_policy risk_memory_horizon_gate \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 64
gpu=$(((gpu + 1) % GPU_COUNT))

run_case "$gpu" server_pcic_ablate_monte_condanchor_seed64_eager \
  data/count_monte_cristo_pg1184.txt 2 64 64 32 "$MONTE_COMBOS" \
  --combo_select_policy risk_memory_horizon_gate \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 64 \
  --sentinel_cascade_anchor_combos '2,0'
gpu=$(((gpu + 1) % GPU_COUNT))

run_case "$gpu" server_pcic_ablate_monte_memoryonly_seed64_eager \
  data/count_monte_cristo_pg1184.txt 2 64 64 32 "$MONTE_COMBOS" \
  --combo_select_policy risk_memory \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 64
gpu=$(((gpu + 1) % GPU_COUNT))

run_case "$gpu" server_pcic_ablate_monte_nohistory_seed64_eager \
  data/count_monte_cristo_pg1184.txt 2 64 64 32 "$MONTE_COMBOS" \
  --combo_select_policy risk_memory_horizon_gate \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 0 \
  --risk_memory_use_history false \
  --sentinel_cascade_anchor_combos '2,0'
gpu=$(((gpu + 1) % GPU_COUNT))

run_case "$gpu" server_pcic_ablate_monte_nopairwise_seed64_eager \
  data/count_monte_cristo_pg1184.txt 2 64 64 32 "$MONTE_COMBOS" \
  --combo_select_policy risk_memory_horizon_gate \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 64 \
  --pairwise_candidate_probe false \
  --sentinel_cascade_anchor_combos '2,0'
gpu=$(((gpu + 1) % GPU_COUNT))

run_case "$gpu" server_pcic_ablate_monte_minloss_seed64_eager \
  data/count_monte_cristo_pg1184.txt 2 64 64 32 "$MONTE_COMBOS" \
  --combo_select_policy min_loss
gpu=$(((gpu + 1) % GPU_COUNT))

if [[ -f "$RULER_VAR" ]]; then
  run_case "$gpu" server_pcic_ablate_rulervar_noanchor_seed64_eager \
    "$RULER_VAR" 3 64 64 32 "$MONTE_COMBOS" \
    --combo_select_policy risk_memory_horizon_gate \
    --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64
  gpu=$(((gpu + 1) % GPU_COUNT))

  run_case "$gpu" server_pcic_ablate_rulervar_memoryonly_seed64_eager \
    "$RULER_VAR" 3 64 64 32 "$MONTE_COMBOS" \
    --combo_select_policy risk_memory \
    --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64
  gpu=$(((gpu + 1) % GPU_COUNT))

  run_case "$gpu" server_pcic_ablate_rulervar_condanchor_seed64_eager \
    "$RULER_VAR" 3 64 64 32 "$MONTE_COMBOS" \
    --combo_select_policy risk_memory_horizon_gate \
    --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 \
    --sentinel_cascade_anchor_combos '2,0'
  gpu=$(((gpu + 1) % GPU_COUNT))

  run_case "$gpu" server_pcic_ablate_rulervar_nopairwise_seed64_eager \
    "$RULER_VAR" 3 64 64 32 "$MONTE_COMBOS" \
    --combo_select_policy risk_memory_horizon_gate \
    --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 \
    --pairwise_candidate_probe false \
    --sentinel_cascade_anchor_combos '2,0'
else
  echo "[skip-ruler-var] missing $RULER_VAR"
fi

wait
"$PY" scripts/summarize_pcic_minimal_component_ablation.py
