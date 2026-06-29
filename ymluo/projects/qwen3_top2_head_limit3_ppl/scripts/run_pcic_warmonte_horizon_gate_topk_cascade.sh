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
  local topk=$4
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
      --sentinel_tokens 64 \
      --sentinel_cascade_initial_tokens 32 \
      --sentinel_cascade_accept_margin 0.01 \
      --sentinel_cascade_extend_topk "$topk" \
      --sentinel_loss_slack 0.0 \
      --sentinel_all_min_margin 0.0 \
      --horizon_gate_min_gain 0.0 \
      --horizon_gate_min_ratio 0.0 \
      --horizon_gate_uncertainty_floor 0.005 \
      > "outputs/logs/${name}.log" 2>&1
  fi
}

run_case server_pcic_war_b2_horizongate_cascade32to64_m01_top2_seed64_g0_r0_eager \
  data/war_and_peace_pg2600.txt \
  '7,6;0,13;0,7;0,6' \
  2
run_case server_pcic_war_b2_horizongate_cascade32to64_m01_top3_seed64_g0_r0_eager \
  data/war_and_peace_pg2600.txt \
  '7,6;0,13;0,7;0,6' \
  3
run_case server_pcic_monte_b2_horizongate_cascade32to64_m01_top2_seed64_g0_r0_eager \
  data/count_monte_cristo_pg1184.txt \
  '2,0,7,12;7,13;2,7;2,0' \
  2
run_case server_pcic_monte_b2_horizongate_cascade32to64_m01_top3_seed64_g0_r0_eager \
  data/count_monte_cristo_pg1184.txt \
  '2,0,7,12;7,13;2,7;2,0' \
  3

"$PY" - <<'PY'
import csv
import json
import pathlib

cases = [
    ("war raw_s64", pathlib.Path("outputs/server_pcic_war_b2_horizongate_s64_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("war cascade_all_m02", pathlib.Path("outputs/server_pcic_war_b2_horizongate_cascade32to64_m02_g0_r0_seed64_eager/pcic_r_blockwise_results.csv")),
    ("war top2_m01", pathlib.Path("outputs/server_pcic_war_b2_horizongate_cascade32to64_m01_top2_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("war top3_m01", pathlib.Path("outputs/server_pcic_war_b2_horizongate_cascade32to64_m01_top3_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("monte raw_s64", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_s64_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("monte cascade_all_m02", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_cascade32to64_m02_g0_r0_seed64_eager/pcic_r_blockwise_results.csv")),
    ("monte top2_m01", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_cascade32to64_m01_top2_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("monte top3_m01", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_cascade32to64_m01_top3_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
]
print("| run | avg_delta_ppl | selected_method/base | gate_s | cascade_extended | cascade_early | avg_extended_candidates | combos |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for name, path in cases:
    rows = list(csv.DictReader(path.open()))
    evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
    avg = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    ratio = sum(float(row["seconds"]) for row in evals) / sum(float(row["baseline_seconds"]) for row in evals)
    gate = sum(float(row.get("gate_seconds") or 0.0) for row in evals)
    extended = 0
    early = 0
    extended_candidate_counts = []
    for row in evals:
        rule = json.loads(row.get("rescue_rule") or "{}")
        did_extend = int(rule.get("sentinel_cascade_extended", 0) or 0)
        extended += did_extend
        early += int(rule.get("sentinel_cascade_accepted_early", 0) or 0)
        if did_extend:
            extended_candidate_counts.append(len(rule.get("sentinel_cascade_extended_candidates") or []))
    avg_extended = sum(extended_candidate_counts) / len(extended_candidate_counts) if extended_candidate_counts else 0.0
    print(
        f"| {name} | {avg:.6f} | {ratio:.3f} | {gate:.3f} | {extended} | {early} | "
        f"{avg_extended:.2f} | {';'.join(row['combo'] for row in evals)} |"
    )
PY
