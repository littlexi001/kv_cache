#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "docs" / "pcic_minimal_component_ablation_2026_06_29.csv"
OUT_MD = ROOT / "docs" / "pcic_minimal_component_ablation_2026_06_29.md"


CASES = [
    ("hard_noanchor", "hard", "no_validation_anchor_top2", "outputs/server_pcic_ablate_hard_noanchor_top2_eval128_seed64_eager/pcic_r_blockwise_results.csv"),
    ("hard_memoryonly", "hard", "memory_only_no_rescue", "outputs/server_pcic_ablate_hard_memoryonly_eval128_seed64_eager/pcic_r_blockwise_results.csv"),
    ("hard_condanchor", "hard", "main_cond_rescue", "outputs/server_pcic_ablate_hard_condanchor_eval128_seed64_eager/pcic_r_blockwise_results.csv"),
    ("hard_nohistory", "hard", "no_history_memory", "outputs/server_pcic_ablate_hard_nohistory_eval128_seed64_eager/pcic_r_blockwise_results.csv"),
    ("hard_nopairwise", "hard", "no_pairwise_probe", "outputs/server_pcic_ablate_hard_nopairwise_eval128_seed64_eager/pcic_r_blockwise_results.csv"),
    ("hard_minloss", "hard", "min_loss_no_pairwise_proxy", "outputs/server_pcic_ablate_hard_minloss_eval128_seed64_eager/pcic_r_blockwise_results.csv"),
    ("monte_noanchor", "monte", "no_validation_anchor_top2", "outputs/server_pcic_ablate_monte_noanchor_seed64_eager/pcic_r_blockwise_results.csv"),
    ("monte_condanchor", "monte", "main_cond_rescue", "outputs/server_pcic_ablate_monte_condanchor_seed64_eager/pcic_r_blockwise_results.csv"),
    ("monte_memoryonly", "monte", "memory_only_no_rescue", "outputs/server_pcic_ablate_monte_memoryonly_seed64_eager/pcic_r_blockwise_results.csv"),
    ("monte_nohistory", "monte", "no_history_memory", "outputs/server_pcic_ablate_monte_nohistory_seed64_eager/pcic_r_blockwise_results.csv"),
    ("monte_nopairwise", "monte", "no_pairwise_probe", "outputs/server_pcic_ablate_monte_nopairwise_seed64_eager/pcic_r_blockwise_results.csv"),
    ("monte_minloss", "monte", "min_loss_no_pairwise_proxy", "outputs/server_pcic_ablate_monte_minloss_seed64_eager/pcic_r_blockwise_results.csv"),
    ("rulervar_noanchor", "ruler_variable", "no_validation_anchor_top2", "outputs/server_pcic_ablate_rulervar_noanchor_seed64_eager/pcic_r_blockwise_results.csv"),
    ("rulervar_memoryonly", "ruler_variable", "memory_only_no_rescue", "outputs/server_pcic_ablate_rulervar_memoryonly_seed64_eager/pcic_r_blockwise_results.csv"),
    ("rulervar_condanchor", "ruler_variable", "main_cond_rescue", "outputs/server_pcic_ablate_rulervar_condanchor_seed64_eager/pcic_r_blockwise_results.csv"),
    ("rulervar_nopairwise", "ruler_variable", "no_pairwise_probe", "outputs/server_pcic_ablate_rulervar_nopairwise_seed64_eager/pcic_r_blockwise_results.csv"),
]


CANONICAL_FALLBACKS = {
    ("hard", "no_validation_anchor_top2"): ("pcic_corrected_gate_core_results_2026_06_29.csv", "hard_top2"),
    ("hard", "main_cond_rescue"): ("pcic_corrected_gate_core_results_2026_06_29.csv", "hard_cond"),
    ("monte", "no_validation_anchor_top2"): ("pcic_corrected_gate_core_results_2026_06_29.csv", "monte_top2"),
    ("monte", "main_cond_rescue"): ("pcic_corrected_gate_core_results_2026_06_29.csv", "monte_cond"),
    ("ruler_variable", "no_validation_anchor_top2"): ("pcic_corrected_gate_core_results_2026_06_29.csv", "ruler_variable_top2"),
    ("ruler_variable", "main_cond_rescue"): ("pcic_corrected_gate_core_results_2026_06_29.csv", "ruler_variable_cond"),
}


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_corrected_fallback(task: str, ablation: str) -> dict[str, str] | None:
    fallback = CANONICAL_FALLBACKS.get((task, ablation))
    if fallback is None:
        return None
    filename, fallback_case_id = fallback
    path = ROOT / "docs" / filename
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            if item.get("case_id") != fallback_case_id:
                continue
            blocks = int(fnum(item.get("blocks"), 0))
            extended = int(fnum(item.get("extended"), 0))
            return {
                "status": "historical",
                "path": f"docs/{filename}:{fallback_case_id}",
                "blocks": str(blocks),
                "avg_delta_ppl": f"{fnum(item.get('avg_delta_ppl')):.6f}",
                "method_ratio": f"{fnum(item.get('corrected_method_ratio') or item.get('old_method_ratio')):.3f}",
                "gate_s": f"{fnum(item.get('corrected_gate_s') or item.get('old_gate_s')):.3f}",
                "extended": str(extended),
                "early": str(max(0, blocks - extended)),
                "combos": str(item.get("combos", "")),
            }
    return None


def summarize_case(case_id: str, task: str, ablation: str, rel_path: str) -> dict[str, str]:
    path = ROOT / rel_path
    row: dict[str, str] = {
        "case_id": case_id,
        "task": task,
        "ablation": ablation,
        "path": rel_path,
        "status": "missing",
        "blocks": "0",
        "avg_delta_ppl": "",
        "method_ratio": "",
        "gate_s": "",
        "extended": "",
        "early": "",
        "combos": "",
    }
    if not path.exists():
        fallback = load_corrected_fallback(task, ablation)
        if fallback is not None:
            row.update(fallback)
        return row

    eval_rows = [
        item
        for item in csv.DictReader(path.open(newline="", encoding="utf-8"))
        if item.get("kind") == "pcic_r_eval"
    ]
    if not eval_rows:
        row["status"] = "empty"
        return row

    rules = []
    for item in eval_rows:
        try:
            rules.append(json.loads(item.get("rescue_rule") or "{}"))
        except json.JSONDecodeError:
            rules.append({})

    baseline_seconds = sum(fnum(item.get("baseline_seconds")) for item in eval_rows)
    method_seconds = sum(fnum(item.get("method_seconds") or item.get("seconds")) for item in eval_rows)
    gate_seconds = sum(fnum(item.get("gate_seconds")) for item in eval_rows)
    avg_delta = sum(fnum(item.get("delta_ppl")) for item in eval_rows) / len(eval_rows)

    row.update(
        {
            "status": "ok",
            "blocks": str(len(eval_rows)),
            "avg_delta_ppl": f"{avg_delta:.6f}",
            "method_ratio": f"{method_seconds / max(baseline_seconds, 1e-9):.3f}",
            "gate_s": f"{gate_seconds:.3f}",
            "extended": str(sum(int(rule.get("sentinel_cascade_extended", 0) or 0) for rule in rules)),
            "early": str(sum(int(rule.get("sentinel_cascade_accepted_early", 0) or 0) for rule in rules)),
            "combos": "/".join(item.get("combo", "") for item in eval_rows),
        }
    )
    return row


def main() -> None:
    rows = [summarize_case(*case) for case in CASES]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PCIC Minimal Component Ablation（2026-06-29）",
        "",
        "该表由 `scripts/run_pcic_minimal_component_ablation_suite.sh` / `scripts/summarize_pcic_minimal_component_ablation.py` 生成。",
        "默认 suite 不跑模型；设置 `RUN_EXPERIMENTS=1` 才会启动实验。",
        "可用 `ONLY_CASES=\"hard_memoryonly hard_nohistory hard_nopairwise\"` 先跑 P0 小集合。",
        "`historical` 表示该行复用了已有 corrected core CSV；`missing` 表示严格消融尚未实际运行。",
        "",
        "## Results",
        "",
        "| task | ablation | status | avg ΔPPL | method/base | gate_s | extended | early | combos |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {task} | `{ablation}` | `{status}` | {avg} | {ratio} | {gate} | {extended} | {early} | `{combos}` |".format(
                task=row["task"],
                ablation=row["ablation"],
                status=row["status"],
                avg=row["avg_delta_ppl"] or "-",
                ratio=row["method_ratio"] or "-",
                gate=row["gate_s"] or "-",
                extended=row["extended"] or "-",
                early=row["early"] or "-",
                combos=row["combos"] or row["path"],
            )
        )

    missing = [row for row in rows if row["status"] not in {"ok", "historical"}]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `no_validation_anchor_top2`：保留 pairwise/horizon top-k rescue，但去掉 validation-prior anchor。",
            "- `memory_only_no_rescue`：使用 `--combo_select_policy risk_memory`，不运行 sentinel/horizon candidate arbitration，作为严格 no-rescue 对照。",
            "- `main_cond_rescue`：当前主方法，对照 rescue gate 是否修复 failure。",
            "- `no_history_memory`：设置 `--risk_memory_use_history false`，不 seed、不更新跨 block 历史，测试 historical prior 的贡献。",
            "- `no_pairwise_probe`：设置 `--pairwise_candidate_probe false`，关闭候选间 sentinel/horizon 对比，只保留 memory anchor。",
            "- `min_loss_no_pairwise_proxy`：用 calibration min-loss 作为额外负对照；它不是严格 no-pairwise，只用于辅助解释。",
            "",
            f"missing strict cases: {len(missing)}",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
