#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
TEXT=${TEXT:-/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt}
ANCHOR_ACCEPT_MARGIN=${ANCHOR_ACCEPT_MARGIN:-0.012}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs docs

COMBOS='0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'
ANCHORS=$("$PY" scripts/select_pcic_validation_anchor.py \
  --combos "$COMBOS" \
  --fixed_pattern 'server_pcic_hardtopic_static_b4_eval64_{combo_tag}_eager' \
  --topk "${ANCHOR_TOPK:-1}" \
  --score "${ANCHOR_SCORE:-avg_delta_ppl}")

echo "[hardtopic-b8] anchors=${ANCHORS} margin=${ANCHOR_ACCEPT_MARGIN}"

run_case() {
  local name=$1
  local sentinel_tokens=$2
  local initial_tokens=$3
  local accept_margin=$4
  local anchors=$5
  local out="outputs/${name}"
  if [[ -f "$out/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[start] $name anchors=$anchors margin=$accept_margin"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" src/run_pcic_rescue_blockwise_local.py \
    --model_name_or_path "$MODEL" \
    --text_path "$TEXT" \
    --output_dir "$out" \
    --prefill_tokens 2048 \
    --num_blocks 8 \
    --calibration_tokens 16 \
    --eval_tokens_per_block 128 \
    --dtype float16 \
    --device cuda:0 \
    --attn_implementation eager \
    --recent_tokens 512 \
    --landmark_stride 64 \
    --combos "$COMBOS" \
    --rescue_strategy none \
    --combo_select_policy risk_memory_horizon_gate \
    --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 \
    --sentinel_tokens "$sentinel_tokens" \
    --sentinel_cascade_initial_tokens "$initial_tokens" \
    --sentinel_cascade_accept_margin "$accept_margin" \
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

run_case server_pcic_hardtopic_b8_horizongate_top2_eval128_seed64_eager \
  64 32 0.01 ""

run_case server_pcic_hardtopic_b8_horizongate_condautoanchor_eval128_seed64_eager \
  128 64 "$ANCHOR_ACCEPT_MARGIN" "$ANCHORS"

"$PY" - <<'PY'
import csv
import json
import pathlib

cases = [
    ("b4_top2", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_top2_timingv2_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("b4_cond_auto_anchor", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_condautoanchor_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("b8_top2", pathlib.Path("outputs/server_pcic_hardtopic_b8_horizongate_top2_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("b8_cond_auto_anchor", pathlib.Path("outputs/server_pcic_hardtopic_b8_horizongate_condautoanchor_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
]

print("| run | blocks | avg_delta_ppl | method/base | gate_s | extended | early | anchors | combos |")
print("|---|---:|---:|---:|---:|---:|---:|---|---|")
for name, path in cases:
    rows = [row for row in csv.DictReader(path.open()) if row.get("kind") == "pcic_r_eval"]
    baseline = sum(float(row.get("baseline_seconds") or 0.0) for row in rows)
    method = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in rows)
    rules = [json.loads(row.get("rescue_rule") or "{}") for row in rows]
    anchors = ";".join(str(anchor) for anchor in (rules[0].get("sentinel_cascade_anchor_combos") or [])) if rules else ""
    print(
        f"| {name} | {len(rows)} | "
        f"{sum(float(row['delta_ppl']) for row in rows) / len(rows):.6f} | "
        f"{method / max(baseline, 1e-9):.3f} | "
        f"{sum(float(row.get('gate_seconds') or 0.0) for row in rows):.3f} | "
        f"{sum(int(rule.get('sentinel_cascade_extended', 0) or 0) for rule in rules)} | "
        f"{sum(int(rule.get('sentinel_cascade_accepted_early', 0) or 0) for rule in rules)} | "
        f"`{anchors}` | `{'/'.join(row['combo'] for row in rows)}` |"
    )
PY
