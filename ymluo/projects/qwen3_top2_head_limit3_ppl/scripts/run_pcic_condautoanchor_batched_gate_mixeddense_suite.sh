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
export LAYER_BUDGET_MIXED_DENSE=1
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

run_batched_case() {
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
    --sentinel_batched_candidates true \
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

echo "[condautoanchor-batched-mixeddense] hard=${HARD_ANCHORS} war=${WAR_ANCHORS} monte=${MONTE_ANCHORS}"

run_batched_case server_pcic_hardtopic_b4_horizongate_condautoanchor_batched_mixeddense_eval128_seed64_eager \
  "$HARD" 4 128 128 64 "$HARD_COMBOS" "$HARD_ANCHORS"
run_batched_case server_pcic_war_b2_horizongate_condautoanchor_batched_mixeddense_seed64_eager \
  data/war_and_peace_pg2600.txt 2 64 64 32 "$WAR_COMBOS" "$WAR_ANCHORS"
run_batched_case server_pcic_monte_b2_horizongate_condautoanchor_batched_mixeddense_seed64_eager \
  data/count_monte_cristo_pg1184.txt 2 64 64 32 "$MONTE_COMBOS" "$MONTE_ANCHORS"

"$PY" - <<'PY'
import csv
import json
import pathlib

doc = pathlib.Path("docs/pcic_condautoanchor_batched_gate_mixeddense_2026_06_29.md")
csv_path = pathlib.Path("docs/pcic_condautoanchor_batched_gate_mixeddense_2026_06_29.csv")

cases = [
    ("hard_serial", "hard", "serial", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_condautoanchor_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("hard_optdispatch", "hard", "optdispatch", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_condautoanchor_batched_optdispatch_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("hard_mixeddense", "hard", "mixeddense", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_condautoanchor_batched_mixeddense_eval128_seed64_eager/pcic_r_blockwise_results.csv")),
    ("war_serial", "war", "serial", pathlib.Path("outputs/server_pcic_war_b2_horizongate_condautoanchor_seed64_eager/pcic_r_blockwise_results.csv")),
    ("war_optdispatch", "war", "optdispatch", pathlib.Path("outputs/server_pcic_war_b2_horizongate_condautoanchor_batched_optdispatch_seed64_eager/pcic_r_blockwise_results.csv")),
    ("war_mixeddense", "war", "mixeddense", pathlib.Path("outputs/server_pcic_war_b2_horizongate_condautoanchor_batched_mixeddense_seed64_eager/pcic_r_blockwise_results.csv")),
    ("monte_serial", "monte", "serial", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_condautoanchor_seed64_eager/pcic_r_blockwise_results.csv")),
    ("monte_optdispatch", "monte", "optdispatch", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_condautoanchor_batched_optdispatch_seed64_eager/pcic_r_blockwise_results.csv")),
    ("monte_mixeddense", "monte", "mixeddense", pathlib.Path("outputs/server_pcic_monte_b2_horizongate_condautoanchor_batched_mixeddense_seed64_eager/pcic_r_blockwise_results.csv")),
]


def summarize(case_id: str, dataset: str, mode: str, path: pathlib.Path) -> dict[str, str]:
    rows = [row for row in csv.DictReader(path.open()) if row.get("kind") == "pcic_r_eval"]
    rules = [json.loads(row.get("rescue_rule") or "{}") for row in rows]
    baseline = sum(float(row.get("baseline_seconds") or 0.0) for row in rows)
    method = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in rows)
    gate = sum(float(row.get("gate_seconds") or 0.0) for row in rows)
    avg_delta = sum(float(row["delta_ppl"]) for row in rows) / max(1, len(rows))
    combos = "/".join(row["combo"] for row in rows)
    return {
        "case_id": case_id,
        "dataset": dataset,
        "mode": mode,
        "avg_delta_ppl": f"{avg_delta:.6f}",
        "method_ratio": f"{method / max(baseline, 1e-9):.3f}",
        "gate_s": f"{gate:.3f}",
        "extended": str(sum(int(rule.get("sentinel_cascade_extended", 0) or 0) for rule in rules)),
        "early": str(sum(int(rule.get("sentinel_cascade_accepted_early", 0) or 0) for rule in rules)),
        "batched": str(max((int(rule.get("sentinel_batched_candidates", 0) or 0) for rule in rules), default=0)),
        "combos": combos,
    }


summaries = [summarize(*case) for case in cases]
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
    writer.writeheader()
    writer.writerows(summaries)

by_dataset: dict[str, dict[str, dict[str, str]]] = {}
for row in summaries:
    by_dataset.setdefault(row["dataset"], {})[row["mode"]] = row

lines = [
    "# Mixed Full/Landmark Dense Batch-row Gate（2026-06-29）",
    "",
    "## 目的",
    "",
    "验证 Stage 1 mixed-budget tensorized path：在 mixed full/landmark layers 上用整批 dense mask forward 替代 row-wise 分组 dispatch。",
    "",
    f"原始 CSV：`{csv_path}`",
    "",
    "## 结果表",
    "",
    "| run | dataset | mode | avg_delta_ppl | method/base | gate_s | extended | early | batched | combos |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for row in summaries:
    lines.append(
        f"| {row['case_id']} | {row['dataset']} | {row['mode']} | {row['avg_delta_ppl']} | "
        f"{row['method_ratio']} | {row['gate_s']} | {row['extended']} | {row['early']} | "
        f"{row['batched']} | `{row['combos']}` |"
    )

lines += [
    "",
    "## Mixed Dense vs Optimized Dispatch",
    "",
    "| dataset | optdispatch gate_s | mixeddense gate_s | gate_s change | serial gate_s | mixeddense same combos as serial |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
]
for dataset, rows in by_dataset.items():
    serial = rows["serial"]
    opt = rows["optdispatch"]
    dense = rows["mixeddense"]
    lines.append(
        f"| {dataset} | {float(opt['gate_s']):.3f} | {float(dense['gate_s']):.3f} | "
        f"{float(dense['gate_s']) - float(opt['gate_s']):.3f} | {float(serial['gate_s']):.3f} | "
        f"{str(dense['combos'] == serial['combos'])} |"
    )

lines += [
    "",
    "## 解释",
    "",
    "- 如果 combos 保持一致，说明 dense mixed full/landmark path 没有改变 selector 语义。",
    "- 如果 gate 下降，说明 row-wise group dispatch 是主要瓶颈之一。",
    "- 如果 gate 上升，说明 dense full score 代价大于省掉的 dispatch，下一步应直接做 sparse gather/fused kernel。",
]
doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
