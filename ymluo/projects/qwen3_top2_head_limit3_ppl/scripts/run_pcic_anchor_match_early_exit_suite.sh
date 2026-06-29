#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
HARD=${HARD:-/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt}
ANCHOR_ACCEPT_MARGIN=${ANCHOR_ACCEPT_MARGIN:-0.012}
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
  local blocks=$3
  local eval_tokens=$4
  local sentinel_tokens=$5
  local initial_tokens=$6
  local combos=$7
  local anchors=$8
  local out="outputs/${name}"
  if [[ -f "$out/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[start] $name anchors=$anchors margin=$ANCHOR_ACCEPT_MARGIN"
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
    --sentinel_tokens "$sentinel_tokens" \
    --sentinel_cascade_initial_tokens "$initial_tokens" \
    --sentinel_cascade_accept_margin "$ANCHOR_ACCEPT_MARGIN" \
    --sentinel_cascade_extend_topk 2 \
    --sentinel_cascade_anchor_combos "$anchors" \
    --sentinel_cascade_anchor_accept_on_match true \
    --sentinel_loss_slack 0.0 \
    --sentinel_all_min_margin 0.0 \
    --horizon_gate_min_gain 0.0 \
    --horizon_gate_min_ratio 0.0 \
    --horizon_gate_uncertainty_floor 0.005 \
    > "outputs/logs/${name}.log" 2>&1
  echo "[done] $name"
}

HARD_COMBOS='0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'
WAR_COMBOS='7,6;0,13;0,7;0,6'
MONTE_COMBOS='2,0,7,12;7,13;2,7;2,0'
NEEDLE_COMBOS="$HARD_COMBOS"

HARD_ANCHORS=$(select_anchor "$HARD_COMBOS" 'server_pcic_hardtopic_static_b4_eval64_{combo_tag}_eager')
WAR_ANCHORS=$(select_anchor "$WAR_COMBOS" 'server_pcic_war_static_b2_eval64_{combo_tag}_eager')
MONTE_ANCHORS=$(select_anchor "$MONTE_COMBOS" 'server_pcic_monte_static_b2_eval64_{combo_tag}_eager')
NEEDLE_ANCHORS=$(select_anchor "$NEEDLE_COMBOS" 'server_pcic_needleval_static_b2_eval128_{combo_tag}_eager')

echo "[anchor-match-early-exit] hard=${HARD_ANCHORS} war=${WAR_ANCHORS} monte=${MONTE_ANCHORS} needle=${NEEDLE_ANCHORS}"

run_case server_pcic_hardtopic_b4_horizongate_anchoraccept_eval128_seed64_eager \
  "$HARD" 4 128 128 64 "$HARD_COMBOS" "$HARD_ANCHORS"
run_case server_pcic_war_b2_horizongate_anchoraccept_seed64_eager \
  data/war_and_peace_pg2600.txt 2 64 64 32 "$WAR_COMBOS" "$WAR_ANCHORS"
run_case server_pcic_monte_b2_horizongate_anchoraccept_seed64_eager \
  data/count_monte_cristo_pg1184.txt 2 64 64 32 "$MONTE_COMBOS" "$MONTE_ANCHORS"
run_case server_pcic_needle_b4_horizongate_anchoraccept_eval128_seed64_eager \
  data/pcic_needle_style_eval_2026_06_29.txt 4 128 128 64 "$NEEDLE_COMBOS" "$NEEDLE_ANCHORS"

"$PY" - <<'PY'
import csv
import json
import pathlib

cases = [
    ("hard_cond", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_condautoanchor_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("hard_anchor_accept", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_anchoraccept_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("war_cond", pathlib.Path("outputs/server_pcic_war_b2_horizongate_condautoanchor_seed64_eager/pcic_r_blockwise_results.csv")),
    ("war_anchor_accept", pathlib.Path("outputs/server_pcic_war_b2_horizongate_anchoraccept_seed64_eager/pcic_r_blockwise_results.csv")),
    ("monte_cond", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_condautoanchor_seed64_eager/pcic_r_blockwise_results.csv")),
    ("monte_anchor_accept", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_anchoraccept_seed64_eager/pcic_r_blockwise_results.csv")),
    ("needle_cond", pathlib.Path("outputs/server_pcic_needle_b4_horizongate_condautoanchor_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("needle_anchor_accept", pathlib.Path("outputs/server_pcic_needle_b4_horizongate_anchoraccept_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
]

print("| run | avg_delta_ppl | method/base | gate_s | extended | early | anchor_match_early | anchors | combos |")
print("|---|---:|---:|---:|---:|---:|---:|---|---|")
for name, path in cases:
    rows = [row for row in csv.DictReader(path.open()) if row.get("kind") == "pcic_r_eval"]
    baseline = sum(float(row.get("baseline_seconds") or 0.0) for row in rows)
    method = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in rows)
    rules = [json.loads(row.get("rescue_rule") or "{}") for row in rows]
    anchors = ";".join(str(anchor) for anchor in (rules[0].get("sentinel_cascade_anchor_combos") or [])) if rules else ""
    print(
        f"| {name} | {sum(float(row['delta_ppl']) for row in rows) / len(rows):.6f} | "
        f"{method / max(baseline, 1e-9):.3f} | "
        f"{sum(float(row.get('gate_seconds') or 0.0) for row in rows):.3f} | "
        f"{sum(int(rule.get('sentinel_cascade_extended', 0) or 0) for rule in rules)} | "
        f"{sum(int(rule.get('sentinel_cascade_accepted_early', 0) or 0) for rule in rules)} | "
        f"{sum(int(rule.get('sentinel_cascade_accepted_by_anchor_match', 0) or 0) for rule in rules)} | "
        f"`{anchors}` | `{'/'.join(row['combo'] for row in rows)}` |"
    )
PY
