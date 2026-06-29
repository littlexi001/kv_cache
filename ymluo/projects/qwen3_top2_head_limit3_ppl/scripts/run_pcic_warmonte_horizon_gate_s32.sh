#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs

run_case() {
  local name=$1
  local text=$2
  local combos=$3
  local out="outputs/${name}"
  if [[ ! -f "$out/pcic_r_blockwise_results.csv" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" src/run_pcic_rescue_blockwise_local.py \
      --model_name_or_path "$MODEL" \
      --text_path "$text" \
      --output_dir "$out" \
      --prefill_tokens 2048 --num_blocks 2 --calibration_tokens 16 --eval_tokens_per_block 64 \
      --dtype float16 --device cuda:0 --attn_implementation eager \
      --recent_tokens 512 --landmark_stride 64 \
      --combos "$combos" \
      --rescue_strategy none \
      --combo_select_policy risk_memory_horizon_gate \
      --risk_memory_loss_slack 0.2 \
      --risk_memory_seed_tokens 64 \
      --sentinel_tokens 32 \
      --sentinel_loss_slack 0.0 \
      --sentinel_all_min_margin 0.0 \
      --horizon_gate_min_gain 0.0 \
      --horizon_gate_min_ratio 0.0 \
      --horizon_gate_uncertainty_floor 0.005 \
      > "outputs/logs/${name}.log" 2>&1
  fi
}

run_case server_pcic_war_b2_horizongate_s32_seed64_g0_r0_eager \
  data/war_and_peace_pg2600.txt \
  '7,6;0,13;0,7;0,6'

run_case server_pcic_monte_b2_horizongate_s32_seed64_g0_r0_eager \
  data/count_monte_cristo_pg1184.txt \
  '2,0,7,12;7,13;2,7;2,0'

"$PY" - <<'PY'
import csv
import json
import pathlib

cases = [
    ("war", pathlib.Path("outputs/server_pcic_war_b2_horizongate_s32_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("monte", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_s32_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
]
print("| case | block | combo | delta_ppl | route | gain | ratio | best_combo |")
print("|---|---:|---|---:|---|---:|---:|---|")
for case, path in cases:
    rows = list(csv.DictReader(path.open()))
    evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
    for row in evals:
        rule = json.loads(row.get("rescue_rule") or "{}")
        print(
            f"| {case} | {row['block']} | {row['combo']} | {float(row['delta_ppl']):.6f} | "
            f"{rule.get('sentinel_route')} | {float(rule.get('sentinel_horizon_gain', 0.0)):.6f} | "
            f"{float(rule.get('sentinel_horizon_gain_ratio', 0.0)):.3f} | {rule.get('sentinel_horizon_best_combo')} |"
        )
    ratio = sum(float(row["seconds"]) for row in evals) / sum(float(row["baseline_seconds"]) for row in evals)
    print(
        f"| {case} avg | | {';'.join(row['combo'] for row in evals)} | "
        f"{sum(float(row['delta_ppl']) for row in evals) / len(evals):.6f} | method/base={ratio:.3f} | | | |"
    )
PY
