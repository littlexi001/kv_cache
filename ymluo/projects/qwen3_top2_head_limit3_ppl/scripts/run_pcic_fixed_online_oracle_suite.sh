#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
HARD=${HARD:-/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt}
RUN_EXPERIMENTS=${RUN_EXPERIMENTS:-0}
RUN_STATIC=${RUN_STATIC:-$RUN_EXPERIMENTS}
RUN_ONLINE=${RUN_ONLINE:-$RUN_EXPERIMENTS}
GPU_COUNT=${GPU_COUNT:-8}

cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs

run_fixed_one() {
  local gpu=$1
  local name=$2
  local text=$3
  local blocks=$4
  local eval_tokens=$5
  local combo=$6
  local out="outputs/${name}"
  if [[ -f "$out/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  if [[ ! -f "$text" ]]; then
    echo "[missing-text] $text"
    return 1
  fi
  echo "[fixed-start] name=$name gpu=$gpu combo=$combo"
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
    --combos "$combo" \
    --rescue_strategy none \
    --combo_select_policy min_loss \
    > "outputs/logs/${name}.log" 2>&1
  echo "[fixed-done] $name"
}

run_online_one() {
  local name=$1
  local text=$2
  local blocks=$3
  local eval_tokens=$4
  local combos=$5
  local out="outputs/${name}"
  if [[ -f "$out/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  if [[ ! -f "$text" ]]; then
    echo "[missing-text] $text"
    return 1
  fi
  echo "[online-start] name=$name gpu=${CUDA_VISIBLE_DEVICES:-0}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" src/run_pcic_rescue_blockwise_local.py \
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
    --combo_select_policy risk_memory_horizon_gate \
    --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 \
    --sentinel_tokens 64 \
    --sentinel_cascade_initial_tokens 32 \
    --sentinel_cascade_accept_margin 0.01 \
    --sentinel_cascade_extend_topk 2 \
    --sentinel_loss_slack 0.0 \
    --sentinel_all_min_margin 0.0 \
    --horizon_gate_min_gain 0.0 \
    --horizon_gate_min_ratio 0.0 \
    --horizon_gate_uncertainty_floor 0.005 \
    > "outputs/logs/${name}.log" 2>&1
  echo "[online-done] $name"
}

run_fixed_group() {
  local prefix=$1
  local text=$2
  local blocks=$3
  local eval_tokens=$4
  shift 4
  local combos=("$@")
  local gpu=0
  for combo in "${combos[@]}"; do
    local tag=${combo//,/_}
    run_fixed_one "$gpu" "${prefix}_${tag}_eager" "$text" "$blocks" "$eval_tokens" "$combo" &
    gpu=$(((gpu + 1) % GPU_COUNT))
    if [[ "$gpu" -eq 0 ]]; then
      wait
    fi
  done
  wait
}

if [[ "$RUN_STATIC" == "1" ]]; then
  run_fixed_group server_pcic_hardtopic_static_b4_eval64 "$HARD" 4 64 \
    "0,6" "0,7" "0,13" "7,6" "2,0" "2,7" "2,0,7,12" "7,13"
  run_fixed_group server_pcic_hardtopic_static_b4_eval128 "$HARD" 4 128 \
    "0,6" "0,7" "0,13" "7,6" "2,0" "2,7" "2,0,7,12" "7,13"
  run_fixed_group server_pcic_war_static_b2_eval64 data/war_and_peace_pg2600.txt 2 64 \
    "7,6" "0,13" "0,7" "0,6"
  run_fixed_group server_pcic_monte_static_b2_eval64 data/count_monte_cristo_pg1184.txt 2 64 \
    "2,0,7,12" "7,13" "2,7" "2,0"
else
  echo "[summary-only] set RUN_STATIC=1 or RUN_EXPERIMENTS=1 to run fixed-combo jobs"
fi

if [[ "$RUN_ONLINE" == "1" ]]; then
  run_online_one server_pcic_hardtopic_b4_horizongate_top2_timingv2_eval64_seed64_eager \
    "$HARD" 4 64 "0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13"
  run_online_one server_pcic_hardtopic_b4_horizongate_top2_timingv2_eval128_seed64_eager \
    "$HARD" 4 128 "0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13"
  run_online_one server_pcic_war_b2_horizongate_top2_timingv2_seed64_eager \
    data/war_and_peace_pg2600.txt 2 64 "7,6;0,13;0,7;0,6"
  run_online_one server_pcic_monte_b2_horizongate_top2_timingv2_seed64_eager \
    data/count_monte_cristo_pg1184.txt 2 64 "2,0,7,12;7,13;2,7;2,0"
else
  echo "[summary-only] set RUN_ONLINE=1 or RUN_EXPERIMENTS=1 to run online PCIC jobs"
fi

"$PY" scripts/summarize_pcic_fixed_online_oracle.py
