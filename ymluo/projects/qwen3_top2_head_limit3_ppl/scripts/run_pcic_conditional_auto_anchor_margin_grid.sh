#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
HARD=${HARD:-/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt}
MARGINS=${MARGINS:-"0.008 0.010 0.012 0.015"}
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

margin_tag() {
  local margin=$1
  echo "m${margin/./}"
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
  local margin=$9
  local out="outputs/${name}"
  if [[ -f "$out/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[start] $name anchors=$anchors margin=$margin"
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
    --sentinel_cascade_accept_margin "$margin" \
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

HARD_COMBOS='0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'
WAR_COMBOS='7,6;0,13;0,7;0,6'
MONTE_COMBOS='2,0,7,12;7,13;2,7;2,0'

HARD_ANCHORS=$(select_anchor "$HARD_COMBOS" 'server_pcic_hardtopic_static_b4_eval64_{combo_tag}_eager')
WAR_ANCHORS=$(select_anchor "$WAR_COMBOS" 'server_pcic_war_static_b2_eval64_{combo_tag}_eager')
MONTE_ANCHORS=$(select_anchor "$MONTE_COMBOS" 'server_pcic_monte_static_b2_eval64_{combo_tag}_eager')

echo "[margin-grid] hard=${HARD_ANCHORS} war=${WAR_ANCHORS} monte=${MONTE_ANCHORS} margins=${MARGINS}"

for margin in $MARGINS; do
  tag=$(margin_tag "$margin")
  run_case "server_pcic_hardtopic_b4_horizongate_condautoanchor_${tag}_eval128_seed64_eager" \
    "$HARD" 4 128 128 64 "$HARD_COMBOS" "$HARD_ANCHORS" "$margin"
  run_case "server_pcic_war_b2_horizongate_condautoanchor_${tag}_seed64_eager" \
    data/war_and_peace_pg2600.txt 2 64 64 32 "$WAR_COMBOS" "$WAR_ANCHORS" "$margin"
  run_case "server_pcic_monte_b2_horizongate_condautoanchor_${tag}_seed64_eager" \
    data/count_monte_cristo_pg1184.txt 2 64 64 32 "$MONTE_COMBOS" "$MONTE_ANCHORS" "$margin"
done

"$PY" - <<'PY'
import csv
import json
import os
import pathlib

root = pathlib.Path("outputs")
margins = os.environ.get("MARGINS", "0.008 0.010 0.012 0.015").split()


def margin_tag(margin: str) -> str:
    return "m" + margin.replace(".", "")


def summarize(path: pathlib.Path) -> dict[str, object]:
    rows = [row for row in csv.DictReader(path.open()) if row.get("kind") == "pcic_r_eval"]
    baseline = sum(float(row.get("baseline_seconds") or 0.0) for row in rows)
    method = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in rows)
    rules = [json.loads(row.get("rescue_rule") or "{}") for row in rows]
    return {
        "avg_delta_ppl": sum(float(row["delta_ppl"]) for row in rows) / len(rows),
        "method_ratio": method / max(baseline, 1e-9),
        "gate_s": sum(float(row.get("gate_seconds") or 0.0) for row in rows),
        "extended": sum(int(rule.get("sentinel_cascade_extended", 0) or 0) for rule in rules),
        "early": sum(int(rule.get("sentinel_cascade_accepted_early", 0) or 0) for rule in rules),
        "anchors": ";".join(str(anchor) for anchor in (rules[0].get("sentinel_cascade_anchor_combos") or [])) if rules else "",
        "combos": ";".join(row["combo"] for row in rows),
    }


print("| margin | dataset | avg_delta_ppl | method/base | gate_s | extended | early | anchors | combos |")
print("|---:|---|---:|---:|---:|---:|---:|---|---|")
for margin in margins:
    tag = margin_tag(margin)
    cases = [
        ("Hard-topic eval128", root / f"server_pcic_hardtopic_b4_horizongate_condautoanchor_{tag}_eval128_seed64_eager/pcic_r_blockwise_results.csv"),
        ("War", root / f"server_pcic_war_b2_horizongate_condautoanchor_{tag}_seed64_eager/pcic_r_blockwise_results.csv"),
        ("Monte", root / f"server_pcic_monte_b2_horizongate_condautoanchor_{tag}_seed64_eager/pcic_r_blockwise_results.csv"),
    ]
    for label, path in cases:
        row = summarize(path)
        print(
            f"| {margin} | {label} | {row['avg_delta_ppl']:.6f} | {row['method_ratio']:.3f} | "
            f"{row['gate_s']:.3f} | {row['extended']} | {row['early']} | "
            f"`{row['anchors']}` | `{row['combos']}` |"
        )
PY
