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
  local sentinel_tokens=$4
  local min_gain=$5
  local cascade_initial=$6
  local cascade_margin=$7
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
      --sentinel_tokens "$sentinel_tokens" \
      --sentinel_cascade_initial_tokens "$cascade_initial" \
      --sentinel_cascade_accept_margin "$cascade_margin" \
      --sentinel_loss_slack 0.0 \
      --sentinel_all_min_margin 0.0 \
      --horizon_gate_min_gain "$min_gain" \
      --horizon_gate_min_ratio 0.0 \
      --horizon_gate_uncertainty_floor 0.005 \
      > "outputs/logs/${name}.log" 2>&1
  fi
}

run_war_monte() {
  local suffix=$1
  local sentinel_tokens=$2
  local min_gain=$3
  local cascade_initial=$4
  local cascade_margin=$5
  run_case "server_pcic_war_b2_horizongate_${suffix}_seed64_eager" \
    data/war_and_peace_pg2600.txt \
    '7,6;0,13;0,7;0,6' \
    "$sentinel_tokens" "$min_gain" "$cascade_initial" "$cascade_margin"
  run_case "server_pcic_monte_b2_horizongate_${suffix}_seed64_eager" \
    data/count_monte_cristo_pg1184.txt \
    '2,0,7,12;7,13;2,7;2,0' \
    "$sentinel_tokens" "$min_gain" "$cascade_initial" "$cascade_margin"
}

run_war_monte s32_g02_r0 32 0.02 0 0.02
run_war_monte cascade32to64_m02_g0_r0 64 0.0 32 0.02

"$PY" - <<'PY'
import csv
import json
import pathlib

cases = [
    "server_pcic_war_b2_horizongate_s32_g02_r0_seed64_eager",
    "server_pcic_monte_b2_horizongate_s32_g02_r0_seed64_eager",
    "server_pcic_war_b2_horizongate_cascade32to64_m02_g0_r0_seed64_eager",
    "server_pcic_monte_b2_horizongate_cascade32to64_m02_g0_r0_seed64_eager",
]
print("| run | avg_delta_ppl | selected_method/base | gate_s | cascade_extended | cascade_early | combos |")
print("|---|---:|---:|---:|---:|---:|---|")
for name in cases:
    path = pathlib.Path("outputs") / name / "pcic_r_blockwise_results.csv"
    rows = list(csv.DictReader(path.open()))
    evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
    avg = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    ratio = sum(float(row["seconds"]) for row in evals) / sum(float(row["baseline_seconds"]) for row in evals)
    gate = sum(float(row.get("gate_seconds") or 0.0) for row in evals)
    extended = 0
    early = 0
    for row in evals:
        rule = json.loads(row.get("rescue_rule") or "{}")
        extended += int(rule.get("sentinel_cascade_extended", 0) or 0)
        early += int(rule.get("sentinel_cascade_accepted_early", 0) or 0)
    print(
        f"| {name} | {avg:.6f} | {ratio:.3f} | {gate:.3f} | "
        f"{extended} | {early} | {';'.join(row['combo'] for row in evals)} |"
    )
PY
