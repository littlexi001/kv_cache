#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs docs

select_anchor() {
  local combos=$1
  local pattern=$2
  "$PY" scripts/select_pcic_validation_anchor.py \
    --combos "$combos" \
    --fixed_pattern "$pattern" \
    --topk "${ANCHOR_TOPK:-1}" \
    --score "${ANCHOR_SCORE:-avg_delta_ppl}"
}

run_case() {
  local name=$1
  local text=$2
  local combos=$3
  local anchors=$4
  local out="outputs/${name}"
  if [[ -f "$out/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[start] $name anchors=$anchors"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" src/run_pcic_rescue_blockwise_local.py \
    --model_name_or_path "$MODEL" \
    --text_path "$text" \
    --output_dir "$out" \
    --prefill_tokens 2048 \
    --num_blocks 2 \
    --calibration_tokens 16 \
    --eval_tokens_per_block 64 \
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
    --sentinel_cascade_accept_margin 0.05 \
    --sentinel_cascade_extend_topk 2 \
    --sentinel_cascade_anchor_combos "$anchors" \
    --sentinel_loss_slack 0.0 \
    --sentinel_all_min_margin 0.0 \
    --horizon_gate_min_gain 0.0 \
    --horizon_gate_min_ratio 0.0 \
    --horizon_gate_uncertainty_floor 0.005 \
    > "outputs/logs/${name}.log" 2>&1
  echo "[done] $name"
}

WAR_COMBOS='7,6;0,13;0,7;0,6'
MONTE_COMBOS='2,0,7,12;7,13;2,7;2,0'

WAR_ANCHORS=$(select_anchor "$WAR_COMBOS" 'server_pcic_war_static_b2_eval64_{combo_tag}_eager')
MONTE_ANCHORS=$(select_anchor "$MONTE_COMBOS" 'server_pcic_monte_static_b2_eval64_{combo_tag}_eager')

echo "[auto-anchor] war=${WAR_ANCHORS} monte=${MONTE_ANCHORS}"

"$PY" scripts/select_pcic_validation_anchor.py \
  --combos "$WAR_COMBOS" \
  --fixed_pattern 'server_pcic_war_static_b2_eval64_{combo_tag}_eager' \
  --score "${ANCHOR_SCORE:-avg_delta_ppl}" \
  --print_table > docs/pcic_war_auto_anchor_validation_prior_2026_06_29.md

"$PY" scripts/select_pcic_validation_anchor.py \
  --combos "$MONTE_COMBOS" \
  --fixed_pattern 'server_pcic_monte_static_b2_eval64_{combo_tag}_eager' \
  --score "${ANCHOR_SCORE:-avg_delta_ppl}" \
  --print_table > docs/pcic_monte_auto_anchor_validation_prior_2026_06_29.md

run_case server_pcic_war_b2_horizongate_cascade32to64_top2_autoanchor_seed64_eager \
  data/war_and_peace_pg2600.txt "$WAR_COMBOS" "$WAR_ANCHORS"

run_case server_pcic_monte_b2_horizongate_cascade32to64_top2_autoanchor_seed64_eager \
  data/count_monte_cristo_pg1184.txt "$MONTE_COMBOS" "$MONTE_ANCHORS"

"$PY" - <<'PY'
import csv
import json
import pathlib

cases = [
    ("war_top2", pathlib.Path("outputs/server_pcic_war_b2_horizongate_top2_timingv2_seed64_eager/pcic_r_blockwise_results.csv")),
    ("war_auto_anchor", pathlib.Path("outputs/server_pcic_war_b2_horizongate_cascade32to64_top2_autoanchor_seed64_eager/pcic_r_blockwise_results.csv")),
    ("monte_top2", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_top2_timingv2_seed64_eager/pcic_r_blockwise_results.csv")),
    ("monte_auto_anchor", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_cascade32to64_top2_autoanchor_seed64_eager/pcic_r_blockwise_results.csv")),
]

print("| run | avg_delta_ppl | method/base | gate_s | extended | early | anchors | combos |")
print("|---|---:|---:|---:|---:|---:|---|---|")
for name, path in cases:
    rows = [row for row in csv.DictReader(path.open()) if row.get("kind") == "pcic_r_eval"]
    avg = sum(float(row["delta_ppl"]) for row in rows) / len(rows)
    baseline = sum(float(row.get("baseline_seconds") or 0.0) for row in rows)
    method = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in rows)
    gate = sum(float(row.get("gate_seconds") or 0.0) for row in rows)
    rules = [json.loads(row.get("rescue_rule") or "{}") for row in rows]
    extended = sum(int(rule.get("sentinel_cascade_extended", 0) or 0) for rule in rules)
    early = sum(int(rule.get("sentinel_cascade_accepted_early", 0) or 0) for rule in rules)
    anchors = ";".join(str(anchor) for anchor in (rules[0].get("sentinel_cascade_anchor_combos") or [])) if rules else ""
    combos = ";".join(row["combo"] for row in rows)
    print(f"| {name} | {avg:.6f} | {method / max(baseline, 1e-9):.3f} | {gate:.3f} | {extended} | {early} | `{anchors}` | `{combos}` |")
PY
