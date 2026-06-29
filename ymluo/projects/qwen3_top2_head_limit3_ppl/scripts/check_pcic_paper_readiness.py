#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT_MD = DOCS / "pcic_paper_readiness_gate_2026_06_29.md"
OUT_CSV = DOCS / "pcic_paper_readiness_gate_2026_06_29.csv"


def read_text(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_csv(rel: str) -> list[dict[str, str]]:
    path = ROOT / rel
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def status(ok: bool, partial: bool = False) -> str:
    if ok:
        return "pass"
    if partial:
        return "partial"
    return "missing"


def main() -> None:
    method_spec = read_text("docs/horizon_pcic_method_spec_2026_06_29.md")
    mainline = read_text("docs/pcic_mainline_fixed_online_rescue_2026_06_29.md")
    trace = read_text("docs/pcic_blockwise_policy_trace_2026_06_29.md")
    corrected = read_text("docs/horizon_pcic_corrected_key_results_2026_06_29.md")
    component = read_text("docs/pcic_component_evidence_matrix_2026_06_29.md")
    ablation_rows = read_csv("docs/pcic_minimal_component_ablation_2026_06_29.csv")

    strict_ablation_rows = [
        row
        for row in ablation_rows
        if row.get("ablation")
        in {"memory_only_no_rescue", "no_history_memory", "no_pairwise_probe"}
    ]
    strict_ok = [row for row in strict_ablation_rows if row.get("status") == "ok"]
    strict_missing = [row for row in strict_ablation_rows if row.get("status") != "ok"]
    historical_rows = [row for row in ablation_rows if row.get("status") == "historical"]

    hard_main = next(
        (
            row
            for row in ablation_rows
            if row.get("task") == "hard"
            and row.get("ablation") == "main_cond_rescue"
            and row.get("status") in {"ok", "historical"}
        ),
        None,
    )
    hard_noanchor = next(
        (
            row
            for row in ablation_rows
            if row.get("task") == "hard"
            and row.get("ablation") == "no_validation_anchor_top2"
            and row.get("status") in {"ok", "historical"}
        ),
        None,
    )
    hard_rescue_gain = None
    if hard_main and hard_noanchor:
        hard_rescue_gain = fnum(hard_main.get("avg_delta_ppl")) - fnum(hard_noanchor.get("avg_delta_ppl"))

    mainline_has_oracle_evidence = (
        "Hard-topic eval128" in mainline
        and "cond-oracle gap" in mainline
        and "0.000000" in mainline
        and "Monte" in mainline
        and "-0.210210" in mainline
    )
    corrected_has_speed_accounting = (
        "corrected gate" in corrected
        and "corrected method/base" in corrected
        and "paper claim" in corrected
        and "baseline" in corrected
    )

    gates: list[dict[str, str]] = [
        {
            "gate": "method_definition",
            "status": status(
                "Algorithm 1: Horizon-PCIC" in method_spec
                and "Pairwise-CIC" in method_spec
                and "conditional horizon rescue gate" in method_spec
            ),
            "evidence": "Method spec contains formal problem, Pairwise-CIC, rescue gate, and Algorithm 1.",
            "next_action": "Keep as paper Method section seed.",
        },
        {
            "gate": "fixed_online_oracle",
            "status": status(mainline_has_oracle_evidence),
            "evidence": "Mainline table shows conditional rescue reaches oracle on Hard-topic and online beats best fixed on Monte.",
            "next_action": "Replicate on formal LongBench/RULER subset.",
        },
        {
            "gate": "blockwise_dynamic_trace",
            "status": status("Hard-topic b8 conditional rescue" in trace and "RULER-style variable binding" in trace),
            "evidence": "Trace table shows non-trivial combo switches across hard-topic and RULER-style variable/topic cases.",
            "next_action": "Turn trace into paper figure with block text/task positions.",
        },
        {
            "gate": "corrected_speed_accounting",
            "status": status(corrected_has_speed_accounting),
            "evidence": "Corrected gate document prevents overclaiming speed and records conservative method/base ratios.",
            "next_action": "Implement fused/sparse candidate probe before claiming baseline speed.",
        },
        {
            "gate": "component_claim_boundary",
            "status": status("missing_direct_ablation" in component and "不能强写" in component),
            "evidence": "Component matrix separates supported claims from missing direct ablations.",
            "next_action": "Update after strict ablation suite finishes.",
        },
        {
            "gate": "strict_component_ablation",
            "status": status(len(strict_missing) == 0 and bool(strict_ablation_rows), partial=bool(strict_ok)),
            "evidence": f"strict ok={len(strict_ok)}, strict missing={len(strict_missing)}, historical baseline rows={len(historical_rows)}.",
            "next_action": "Run ONLY_CASES P0 first, then P1/P2 if P0 supports claim.",
        },
        {
            "gate": "rescue_quality_case",
            "status": status(hard_rescue_gain is not None and hard_rescue_gain < 0.0),
            "evidence": (
                "Hard main_cond_rescue improves over no_validation_anchor_top2 by "
                f"{hard_rescue_gain:.6f} ΔPPL."
                if hard_rescue_gain is not None
                else "Hard rescue comparison unavailable."
            ),
            "next_action": "Add memory_only_no_rescue comparison to isolate rescue gate itself.",
        },
        {
            "gate": "formal_benchmark",
            "status": "missing",
            "evidence": "Current RULER-style results are synthetic/offline smoke, not formal RULER/LongBench.",
            "next_action": "Run formal or locally cached benchmark subset without external downloads.",
        },
        {
            "gate": "real_speed",
            "status": "missing",
            "evidence": "Corrected gate shows method cost remains above baseline; fused/sparse candidate probe is not done.",
            "next_action": "Implement fused/sparse probe or report speed as limitation.",
        },
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate", "status", "evidence", "next_action"])
        writer.writeheader()
        writer.writerows(gates)

    counts: dict[str, int] = {}
    for gate in gates:
        counts[gate["status"]] = counts.get(gate["status"], 0) + 1

    lines = [
        "# PCIC Paper Readiness Gate（2026-06-29）",
        "",
        "目的：把当前 paper 主线转成可重复检查的 readiness gate。每次补实验后重跑该脚本，即可看到哪些 claim 已经能写，哪些仍缺证据。",
        "",
        "## 总览",
        "",
        f"- pass: {counts.get('pass', 0)}",
        f"- partial: {counts.get('partial', 0)}",
        f"- missing: {counts.get('missing', 0)}",
        "",
        "当前判断：方法创新性主线已经成型，但还不能标记为 ICML-ready；主要缺口是严格组件消融、正式 benchmark、真实速度。",
        "",
        "## Gate Table",
        "",
        "| gate | status | evidence | next action |",
        "| --- | --- | --- | --- |",
    ]
    for gate in gates:
        lines.append(
            f"| `{gate['gate']}` | `{gate['status']}` | {gate['evidence']} | {gate['next_action']} |"
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 可以继续沿 `Pairwise-CIC + online blockwise selection + rescue gate` 主线推进。",
            "- 当前最强、最安全的论文 claim 是：online counterfactual policy selection 能修复固定策略 / short-horizon 的失败。",
            "- 暂时不能强 claim：端到端快于 baseline、Pairwise/memory 在所有任务上不可替代、正式 benchmark 已充分验证。",
            "",
            "CSV：`docs/pcic_paper_readiness_gate_2026_06_29.csv`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
