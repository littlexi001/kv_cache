#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
HARD=${HARD:-/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs

run_case() {
  local name=$1
  local text=$2
  local blocks=$3
  local eval_tokens=$4
  local combos=$5
  local out="outputs/${name}"
  if [[ ! -f "$out/pcic_r_blockwise_results.csv" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" src/run_pcic_rescue_blockwise_local.py \
      --model_name_or_path "$MODEL" \
      --text_path "$text" \
      --output_dir "$out" \
      --prefill_tokens 2048 --num_blocks "$blocks" --calibration_tokens 16 --eval_tokens_per_block "$eval_tokens" \
      --dtype float16 --device cuda:0 --attn_implementation eager \
      --recent_tokens 512 --landmark_stride 64 \
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
  fi
}

run_case server_pcic_hardtopic_b4_horizongate_top2_timingv2_eval64_seed64_eager \
  "$HARD" 4 64 '0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'
run_case server_pcic_hardtopic_b4_horizongate_top2_timingv2_eval128_seed64_eager \
  "$HARD" 4 128 '0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'
run_case server_pcic_war_b2_horizongate_top2_timingv2_seed64_eager \
  data/war_and_peace_pg2600.txt 2 64 '7,6;0,13;0,7;0,6'
run_case server_pcic_monte_b2_horizongate_top2_timingv2_seed64_eager \
  data/count_monte_cristo_pg1184.txt 2 64 '2,0,7,12;7,13;2,7;2,0'

"$PY" scripts/summarize_horizon_pcic_results.py
