#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
GPU_COUNT=${GPU_COUNT:-8}
ANCHOR_ACCEPT_MARGIN=${ANCHOR_ACCEPT_MARGIN:-0.012}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p data outputs/logs docs

COMBOS='0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'
VAL_TEXT=data/pcic_ruler_style_validation_2026_06_29.txt
MULTI_TEXT=data/pcic_ruler_style_multineedle_eval_2026_06_29.txt
VAR_TEXT=data/pcic_ruler_style_variable_eval_2026_06_29.txt
TOPIC_TEXT=data/pcic_ruler_style_topicswitch_eval_2026_06_29.txt

"$PY" - <<'PY'
from __future__ import annotations

from pathlib import Path


def distractors(prefix: str, count: int) -> list[str]:
    rows = []
    for idx in range(count):
        rows.append(
            f"{prefix} distractor {idx:04d}: the archive discusses calendars, delivery logs, "
            f"museum labels, river maps, project notes, and unrelated identifier "
            f"{(idx * 37) % 997:03d}-{(idx * 71) % 991:03d}-{(idx * 19) % 983:03d}. "
            "These details are intentionally irrelevant to the final answer."
        )
    return rows


def write_multineedle(path: Path, code_a: str, code_b: str, code_c: str, city: str) -> None:
    if path.exists():
        return
    lines: list[str] = [
        "Synthetic RULER-style multi-needle document.",
        f"Needle A: the blue archive code is {code_a}.",
        f"Needle B: the red archive code is {code_b}.",
        f"Needle C: the green archive code is {code_c}.",
    ]
    lines += distractors("multi-needle first", 520)
    lines += [
        f"Bridge statement: the city tied to all three needles is {city}.",
        "Retrieval task: combine the three archive codes in blue-red-green order.",
        f"Retrieval answer: {code_a} then {code_b} then {code_c}, all tied to {city}.",
    ]
    lines += distractors("multi-needle tail", 180)
    lines += [
        f"Final consistency line: the ordered codes remain {code_a}, {code_b}, and {code_c}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_variable(path: Path, alpha: str, beta: str, gamma: str, result: str) -> None:
    if path.exists():
        return
    lines: list[str] = [
        "Synthetic RULER-style variable binding document.",
        f"Set variable ALPHA to {alpha}.",
        f"Set variable BETA to {beta}.",
        "Later instructions may mention many decoy variable names.",
    ]
    lines += distractors("variable middle", 420)
    lines += [
        f"Update variable GAMMA to {gamma}.",
        f"Define RESULT as ALPHA plus BETA plus GAMMA, which equals {result}.",
    ]
    lines += distractors("variable tail", 260)
    lines += [
        "Question: what is RESULT?",
        f"Answer: RESULT is {result}, derived from ALPHA={alpha}, BETA={beta}, GAMMA={gamma}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_topic_switch(path: Path, early_topic: str, late_topic: str, key: str) -> None:
    if path.exists():
        return
    lines: list[str] = [
        "Synthetic RULER-style topic switch document.",
        f"Initial topic marker: {early_topic}.",
        f"The initial marker has key {key}.",
    ]
    lines += distractors("topic early", 360)
    lines += [
        f"Topic switch: the document now discusses {late_topic}.",
        "The final answer must still remember the initial topic marker and key.",
    ]
    lines += distractors("topic late", 340)
    lines += [
        f"Answer line: the initial topic was {early_topic}, the late topic was {late_topic}, and the key was {key}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


write_multineedle(
    Path("data/pcic_ruler_style_validation_2026_06_29.txt"),
    "VAL-A17",
    "VAL-B42",
    "VAL-C93",
    "Ningbo",
)
write_multineedle(
    Path("data/pcic_ruler_style_multineedle_eval_2026_06_29.txt"),
    "MUL-A64",
    "MUL-B28",
    "MUL-C75",
    "Wuxi",
)
write_variable(
    Path("data/pcic_ruler_style_variable_eval_2026_06_29.txt"),
    alpha="orchid",
    beta="harbor",
    gamma="matrix",
    result="orchid-harbor-matrix",
)
write_topic_switch(
    Path("data/pcic_ruler_style_topicswitch_eval_2026_06_29.txt"),
    early_topic="ancient-glass",
    late_topic="satellite-gardens",
    key="TG-418-ZETA",
)
PY

run_fixed_one() {
  local gpu=$1
  local combo=$2
  local tag=${combo//,/_}
  local name="server_pcic_rulerval_static_b2_eval96_${tag}_eager"
  local out="outputs/${name}"
  if [[ -f "$out/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[fixed-start] $name gpu=$gpu combo=$combo"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/run_pcic_rescue_blockwise_local.py \
    --model_name_or_path "$MODEL" \
    --text_path "$VAL_TEXT" \
    --output_dir "$out" \
    --prefill_tokens 2048 \
    --num_blocks 2 \
    --calibration_tokens 16 \
    --eval_tokens_per_block 96 \
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

run_fixed_validation() {
  local gpu=0
  for combo in ${COMBOS//;/ }; do
    run_fixed_one "$gpu" "$combo" &
    gpu=$(((gpu + 1) % GPU_COUNT))
    if [[ "$gpu" -eq 0 ]]; then
      wait
    fi
  done
  wait
}

run_eval_case() {
  local name=$1
  local text=$2
  local sentinel_tokens=$3
  local initial_tokens=$4
  local accept_margin=$5
  local anchors=$6
  local out="outputs/${name}"
  if [[ -f "$out/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[eval-start] $name anchors=$anchors margin=$accept_margin"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" src/run_pcic_rescue_blockwise_local.py \
    --model_name_or_path "$MODEL" \
    --text_path "$text" \
    --output_dir "$out" \
    --prefill_tokens 2048 \
    --num_blocks 3 \
    --calibration_tokens 16 \
    --eval_tokens_per_block 96 \
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
  echo "[eval-done] $name"
}

run_fixed_validation

ANCHORS=$("$PY" scripts/select_pcic_validation_anchor.py \
  --combos "$COMBOS" \
  --fixed_pattern 'server_pcic_rulerval_static_b2_eval96_{combo_tag}_eager' \
  --topk "${ANCHOR_TOPK:-1}" \
  --score "${ANCHOR_SCORE:-avg_delta_ppl}")

"$PY" scripts/select_pcic_validation_anchor.py \
  --combos "$COMBOS" \
  --fixed_pattern 'server_pcic_rulerval_static_b2_eval96_{combo_tag}_eager' \
  --score "${ANCHOR_SCORE:-avg_delta_ppl}" \
  --print_table > docs/pcic_ruler_style_validation_prior_2026_06_29.md

echo "[ruler-style-smoke] anchors=${ANCHORS}"

for task in multineedle variable topicswitch; do
  case "$task" in
    multineedle) text="$MULTI_TEXT" ;;
    variable) text="$VAR_TEXT" ;;
    topicswitch) text="$TOPIC_TEXT" ;;
  esac
  run_eval_case "server_pcic_ruler_${task}_b3_horizongate_top2_eval96_seed64_eager" \
    "$text" 64 32 0.01 ""
  run_eval_case "server_pcic_ruler_${task}_b3_horizongate_condautoanchor_eval96_seed64_eager" \
    "$text" 96 48 "$ANCHOR_ACCEPT_MARGIN" "$ANCHORS"
done

"$PY" - <<'PY'
import csv
import json
import pathlib

csv_path = pathlib.Path("docs/pcic_ruler_style_smoke_2026_06_29.csv")
doc_path = pathlib.Path("docs/pcic_ruler_style_smoke_2026_06_29.md")

cases = []
for task in ["multineedle", "variable", "topicswitch"]:
    cases.append((f"{task}_top2", task, "top2", pathlib.Path(f"outputs/server_pcic_ruler_{task}_b3_horizongate_top2_eval96_seed64_eager/pcic_r_blockwise_results.csv")))
    cases.append((f"{task}_cond", task, "cond_auto_anchor", pathlib.Path(f"outputs/server_pcic_ruler_{task}_b3_horizongate_condautoanchor_eval96_seed64_eager/pcic_r_blockwise_results.csv")))


def summarize(case_id, task, mode, path):
    rows = [row for row in csv.DictReader(path.open()) if row.get("kind") == "pcic_r_eval"]
    rules = [json.loads(row.get("rescue_rule") or "{}") for row in rows]
    baseline = sum(float(row.get("baseline_seconds") or 0.0) for row in rows)
    method = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in rows)
    gate = sum(float(row.get("gate_seconds") or 0.0) for row in rows)
    anchors = ";".join(str(anchor) for anchor in (rules[0].get("sentinel_cascade_anchor_combos") or [])) if rules else ""
    return {
        "case_id": case_id,
        "task": task,
        "mode": mode,
        "blocks": str(len(rows)),
        "avg_delta_ppl": f"{sum(float(row['delta_ppl']) for row in rows) / max(1, len(rows)):.6f}",
        "method_ratio": f"{method / max(baseline, 1e-9):.3f}",
        "gate_s": f"{gate:.3f}",
        "extended": str(sum(int(rule.get("sentinel_cascade_extended", 0) or 0) for rule in rules)),
        "early": str(sum(int(rule.get("sentinel_cascade_accepted_early", 0) or 0) for rule in rules)),
        "anchors": anchors,
        "combos": "/".join(row["combo"] for row in rows),
    }


summaries = [summarize(*case) for case in cases]
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
    writer.writeheader()
    writer.writerows(summaries)

lines = [
    "# RULER-style Offline Smoke（2026-06-29）",
    "",
    "## 目的",
    "",
    "不下载外部数据，生成三类 RULER-style 长上下文文本，检查 paper 主线是否只在 hard-topic/小说文本上成立。",
    "",
    "任务：multi-needle、variable binding、topic switch。",
    "",
    "validation-prior anchor 见：`docs/pcic_ruler_style_validation_prior_2026_06_29.md`",
    f"原始 CSV：`{csv_path}`",
    "",
    "## 结果表",
    "",
    "| case | task | mode | avg_delta_ppl | method/base | gate_s | extended | early | anchors | combos |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
]
for row in summaries:
    lines.append(
        f"| {row['case_id']} | {row['task']} | {row['mode']} | {row['avg_delta_ppl']} | "
        f"{row['method_ratio']} | {row['gate_s']} | {row['extended']} | {row['early']} | "
        f"`{row['anchors']}` | `{row['combos']}` |"
    )
lines += [
    "",
    "## 解释边界",
    "",
    "- 这是无下载 synthetic smoke，不等同正式 RULER / LongBench。",
    "- 若 cond_auto_anchor 在多个任务上保持或改善 PPL drift，说明主线有跨模式迹象。",
    "- 若某些任务退化，应作为 rescue gate 的失败模式，用于后续 adaptive trigger / benchmark 设计。",
]
doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
