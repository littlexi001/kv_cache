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

HARD_COMBOS='0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'
RULER_COMBOS='0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'

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
  echo "[start] $name anchors=$anchors"
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
    --sentinel_cascade_skip_anchor_nonpositive_gain true \
    --sentinel_loss_slack 0.0 \
    --sentinel_all_min_margin 0.0 \
    --horizon_gate_min_gain 0.0 \
    --horizon_gate_min_ratio 0.0 \
    --horizon_gate_uncertainty_floor 0.005 \
    > "outputs/logs/${name}.log" 2>&1
  echo "[done] $name"
}

HARD_ANCHORS=$(select_anchor "$HARD_COMBOS" 'server_pcic_hardtopic_static_b4_eval64_{combo_tag}_eager')
RULER_ANCHORS=$(select_anchor "$RULER_COMBOS" 'server_pcic_rulerval_static_b2_eval96_{combo_tag}_eager')
NEEDLE_ANCHORS=$("$PY" scripts/select_pcic_validation_anchor.py \
  --combos "$RULER_COMBOS" \
  --fixed_pattern 'server_pcic_needleval_static_b2_eval128_{combo_tag}_eager' \
  --topk "${ANCHOR_TOPK:-1}" \
  --score "${ANCHOR_SCORE:-avg_delta_ppl}")

run_case server_pcic_hardtopic_b4_horizongate_condautoanchor_skipanchor_gain_eval128_seed64_eager \
  "$HARD" 4 128 128 64 "$HARD_COMBOS" "$HARD_ANCHORS"
run_case server_pcic_needle_b4_horizongate_condautoanchor_skipanchor_gain_eval128_seed64_eager \
  data/pcic_needle_style_eval_2026_06_29.txt 4 128 128 64 "$RULER_COMBOS" "$NEEDLE_ANCHORS"

for task in multineedle variable topicswitch; do
  case "$task" in
    multineedle) text=data/pcic_ruler_style_multineedle_eval_2026_06_29.txt ;;
    variable) text=data/pcic_ruler_style_variable_eval_2026_06_29.txt ;;
    topicswitch) text=data/pcic_ruler_style_topicswitch_eval_2026_06_29.txt ;;
  esac
  run_case "server_pcic_ruler_${task}_b3_horizongate_condautoanchor_skipanchor_gain_eval96_seed64_eager" \
    "$text" 3 96 96 48 "$RULER_COMBOS" "$RULER_ANCHORS"
done

"$PY" - <<'PY'
import csv
import json
import pathlib

doc = pathlib.Path("docs/pcic_skip_gate_anchor_gain_online_2026_06_29.md")
csv_out = pathlib.Path("docs/pcic_skip_gate_anchor_gain_online_2026_06_29.csv")

cases = [
    ("hard_base", "hard", "base", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_condautoanchor_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("hard_skip", "hard", "skip", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_condautoanchor_skipanchor_gain_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("needle_base", "needle", "base", pathlib.Path("outputs/server_pcic_needle_b4_horizongate_condautoanchor_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("needle_skip", "needle", "skip", pathlib.Path("outputs/server_pcic_needle_b4_horizongate_condautoanchor_skipanchor_gain_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
]
for task in ["multineedle", "variable", "topicswitch"]:
    cases.append((f"ruler_{task}_base", f"ruler_{task}", "base", pathlib.Path(f"outputs/server_pcic_ruler_{task}_b3_horizongate_condautoanchor_eval96_seed64_eager/pcic_r_blockwise_results.csv")))
    cases.append((f"ruler_{task}_skip", f"ruler_{task}", "skip", pathlib.Path(f"outputs/server_pcic_ruler_{task}_b3_horizongate_condautoanchor_skipanchor_gain_eval96_seed64_eager/pcic_r_blockwise_results.csv")))


def summarize(case_id, task, mode, path):
    rows = [row for row in csv.DictReader(path.open()) if row.get("kind") == "pcic_r_eval"]
    rules = [json.loads(row.get("rescue_rule") or "{}") for row in rows]
    baseline = sum(float(row.get("baseline_seconds") or 0.0) for row in rows)
    method = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in rows)
    gate = sum(float(row.get("gate_seconds") or 0.0) for row in rows)
    skipped = sum(int(rule.get("sentinel_cascade_skipped_by_anchor_nonpositive_gain", 0) or 0) for rule in rules)
    extended = sum(int(rule.get("sentinel_cascade_extended", 0) or 0) for rule in rules)
    early = sum(int(rule.get("sentinel_cascade_accepted_early", 0) or 0) for rule in rules)
    return {
        "case_id": case_id,
        "task": task,
        "mode": mode,
        "avg_delta_ppl": f"{sum(float(row['delta_ppl']) for row in rows) / max(1, len(rows)):.6f}",
        "method_ratio": f"{method / max(baseline, 1e-9):.3f}",
        "gate_s": f"{gate:.3f}",
        "extended": str(extended),
        "early": str(early),
        "skip_anchor_gain": str(skipped),
        "combos": "/".join(row["combo"] for row in rows),
    }


summaries = [summarize(*case) for case in cases]
with csv_out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
    writer.writeheader()
    writer.writerows(summaries)

by_task = {}
for row in summaries:
    by_task.setdefault(row["task"], {})[row["mode"]] = row

lines = [
    "# Anchor Nonpositive-Gain Skip-Gate Online Eval（2026-06-29）",
    "",
    "## 规则",
    "",
    "```text",
    "if initial_selected_combo in validation_prior_anchors",
    "and sentinel_horizon_gain_ratio <= 0:",
    "    skip extension",
    "```",
    "",
    f"原始 CSV：`{csv_out}`",
    "",
    "## 结果表",
    "",
    "| case | task | mode | avg_delta_ppl | method/base | gate_s | extended | early | skipped | combos |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for row in summaries:
    lines.append(
        f"| {row['case_id']} | {row['task']} | {row['mode']} | {row['avg_delta_ppl']} | "
        f"{row['method_ratio']} | {row['gate_s']} | {row['extended']} | {row['early']} | "
        f"{row['skip_anchor_gain']} | `{row['combos']}` |"
    )

lines += [
    "",
    "## Base vs Skip",
    "",
    "| task | ΔPPL change | gate_s change | method/base change | skipped | same combos |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
]
for task, rows in by_task.items():
    base = rows["base"]
    skip = rows["skip"]
    lines.append(
        f"| {task} | {float(skip['avg_delta_ppl']) - float(base['avg_delta_ppl']):.6f} | "
        f"{float(skip['gate_s']) - float(base['gate_s']):.3f} | "
        f"{float(skip['method_ratio']) - float(base['method_ratio']):.3f} | "
        f"{skip['skip_anchor_gain']} | {str(skip['combos'] == base['combos'])} |"
    )

lines += [
    "",
    "## 解释",
    "",
    "- 该规则是保守 skip-gate 候选，目标是减少 easy-regime 中无效 extension。",
    "- 如果 ΔPPL 不退化且 gate_s 下降，可进入下一轮更大样本验证。",
    "- 如果质量退化或 gate 不降，则说明后验规则没有在线泛化，应退回设计阶段。",
]
doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
