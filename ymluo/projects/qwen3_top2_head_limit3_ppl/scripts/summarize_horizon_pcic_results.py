from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"


CASES = [
    (
        "hardtopic_eval64_raw_s64",
        "Hard-topic eval64 raw s64",
        "server_pcic_hardtopic_b4_horizongate_s64_eval64_seed64_g0_r0_eager",
    ),
    (
        "hardtopic_eval64_top2",
        "Hard-topic eval64 top2 cascade",
        "server_pcic_hardtopic_b4_horizongate_top2_timingv2_eval64_seed64_eager",
    ),
    (
        "hardtopic_eval128_raw_s64",
        "Hard-topic eval128 raw s64",
        "server_pcic_hardtopic_b4_horizongate_s64_eval128_seed64_g0_r0_eager",
    ),
    (
        "hardtopic_eval128_top2",
        "Hard-topic eval128 top2 cascade",
        "server_pcic_hardtopic_b4_horizongate_top2_timingv2_eval128_seed64_eager",
    ),
    (
        "war_raw_s64",
        "War raw s64",
        "server_pcic_war_b2_horizongate_s64_seed64_g0_r0_eager",
    ),
    (
        "war_top2",
        "War top2 cascade",
        "server_pcic_war_b2_horizongate_top2_timingv2_seed64_eager",
    ),
    (
        "monte_raw_s64",
        "Monte raw s64",
        "server_pcic_monte_b2_horizongate_s64_seed64_g0_r0_eager",
    ),
    (
        "monte_top2",
        "Monte top2 cascade",
        "server_pcic_monte_b2_horizongate_top2_timingv2_seed64_eager",
    ),
]


QUALITY_BASELINES = {
    "Hard-topic eval64": {
        "none": 0.030228,
        "conffast_s8": 0.003316,
        "static_0,6": -0.012719,
    },
    "Hard-topic eval128": {
        "none": 0.009629,
        "conffast_s8": 0.038679,
        "static_0,6": -0.020744,
    },
    "War": {
        "old_no_rescue": -0.687288,
    },
    "Monte": {
        "old_no_rescue": -0.108427,
    },
}


def read_eval_rows(output_name: str) -> list[dict[str, str]]:
    path = OUTPUTS / output_name / "pcic_r_blockwise_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return [row for row in rows if row.get("kind") == "pcic_r_eval"]


def summarize_case(case_id: str, label: str, output_name: str) -> dict[str, object]:
    eval_rows = read_eval_rows(output_name)
    delta_ppls = [float(row["delta_ppl"]) for row in eval_rows]
    seconds = [float(row["seconds"]) for row in eval_rows]
    baseline_seconds = [float(row["baseline_seconds"]) for row in eval_rows]
    gate_seconds = [float(row.get("gate_seconds") or 0.0) for row in eval_rows]
    method_seconds = [float(row.get("method_seconds") or row["seconds"]) for row in eval_rows]
    rules = [json.loads(row.get("rescue_rule") or "{}") for row in eval_rows]
    cascade_extended = sum(int(rule.get("sentinel_cascade_extended", 0) or 0) for rule in rules)
    cascade_early = sum(int(rule.get("sentinel_cascade_accepted_early", 0) or 0) for rule in rules)
    extended_counts = [
        len(rule.get("sentinel_cascade_extended_candidates") or [])
        for rule in rules
        if int(rule.get("sentinel_cascade_extended", 0) or 0)
    ]
    batched_method_seconds = 0.0
    batched_proxy_available = True
    for row, rule in zip(eval_rows, rules):
        initial_seconds = rule.get("sentinel_cascade_initial_seconds")
        if not isinstance(initial_seconds, dict) or not initial_seconds:
            batched_proxy_available = False
            break
        initial_stage_seconds = max(float(value) for value in initial_seconds.values())
        extension_seconds = rule.get("sentinel_cascade_extension_seconds")
        extension_stage_seconds = (
            max(float(value) for value in extension_seconds.values())
            if isinstance(extension_seconds, dict) and extension_seconds
            else 0.0
        )
        selected_prefix_seconds = float(rule.get("sentinel_selected_prefix_seconds", 0.0) or 0.0)
        selected_remainder_seconds = max(0.0, float(row["seconds"]) - selected_prefix_seconds)
        batched_method_seconds += (
            initial_stage_seconds + extension_stage_seconds + selected_remainder_seconds
        )
    batched_total_ratio = (
        batched_method_seconds / sum(baseline_seconds) if batched_proxy_available else None
    )
    return {
        "case_id": case_id,
        "label": label,
        "output": output_name,
        "blocks": len(eval_rows),
        "avg_delta_ppl": sum(delta_ppls) / len(delta_ppls),
        "selected_ratio": sum(seconds) / sum(baseline_seconds),
        "serial_total_ratio": sum(method_seconds) / sum(baseline_seconds),
        "batched_proxy_ratio": batched_total_ratio,
        "gate_s": sum(gate_seconds),
        "cascade_extended": cascade_extended,
        "cascade_early": cascade_early,
        "avg_extended_candidates": (
            sum(extended_counts) / len(extended_counts) if extended_counts else 0.0
        ),
        "combos": ";".join(row["combo"] for row in eval_rows),
    }


def pct_reduction(old: float, new: float) -> float:
    return 100.0 * (old - new) / old


def format_optional_float(value: object, digits: int) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "case_id",
        "label",
        "output",
        "blocks",
        "avg_delta_ppl",
        "selected_ratio",
        "serial_total_ratio",
        "batched_proxy_ratio",
        "gate_s",
        "cascade_extended",
        "cascade_early",
        "avg_extended_candidates",
        "combos",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    by_id = {str(row["case_id"]): row for row in rows}

    def row_line(row: dict[str, object]) -> str:
        return (
            f"| {row['label']} | {int(row['blocks'])} | "
            f"{float(row['avg_delta_ppl']):.6f} | "
            f"{float(row['selected_ratio']):.3f} | "
            f"{float(row['serial_total_ratio']):.3f} | "
            f"{format_optional_float(row['batched_proxy_ratio'], 3)} | "
            f"{float(row['gate_s']):.3f} | "
            f"{int(row['cascade_extended'])} | {int(row['cascade_early'])} | "
            f"{float(row['avg_extended_candidates']):.2f} | `{row['combos']}` |"
        )

    lines: list[str] = []
    lines.append("# Horizon-PCIC 关键结果汇总（2026-06-29）")
    lines.append("")
    lines.append("本文件由 `scripts/summarize_horizon_pcic_results.py` 从远端 `outputs/*/pcic_r_blockwise_results.csv` 自动生成。")
    lines.append("")
    lines.append("## 主结果表")
    lines.append("")
    lines.append(
        "| run | blocks | avg_delta_ppl | selected/base | serial_total/base | batched_proxy/base | gate_s | extended | early | avg_ext_cands | combos |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(row_line(row))
    lines.append("")

    lines.append("## Gate 成本下降")
    lines.append("")
    lines.append("| dataset | raw_s64_gate_s | top2_gate_s | reduction | quality_same |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    pairs = [
        ("Hard-topic eval64", "hardtopic_eval64_raw_s64", "hardtopic_eval64_top2"),
        ("Hard-topic eval128", "hardtopic_eval128_raw_s64", "hardtopic_eval128_top2"),
        ("War", "war_raw_s64", "war_top2"),
        ("Monte", "monte_raw_s64", "monte_top2"),
    ]
    for label, raw_id, top2_id in pairs:
        raw = by_id[raw_id]
        top2 = by_id[top2_id]
        quality_same = abs(float(raw["avg_delta_ppl"]) - float(top2["avg_delta_ppl"])) < 1e-9
        lines.append(
            f"| {label} | {float(raw['gate_s']):.3f} | {float(top2['gate_s']):.3f} | "
            f"{pct_reduction(float(raw['gate_s']), float(top2['gate_s'])):.1f}% | {quality_same} |"
        )
    lines.append("")

    lines.append("## 质量参考 baseline")
    lines.append("")
    lines.append("| dataset | baseline | avg_delta_ppl |")
    lines.append("| --- | --- | ---: |")
    for dataset, baselines in QUALITY_BASELINES.items():
        for name, value in baselines.items():
            lines.append(f"| {dataset} | {name} | {value:.6f} |")
    lines.append("")

    lines.append("## 当前可写进 paper 的结论")
    lines.append("")
    lines.append("1. `top2 cascade` 在 Hard-topic、War、Monte 上保持 `raw_s64` 的质量。")
    lines.append("2. `top2 cascade` 将 gate 串行成本降低约 50%–59%，但仍未解决全部端到端速度问题。")
    lines.append("3. 方法主线应表述为：`Pairwise-CIC + online blockwise policy selection + horizon-aware top-k cascade rescue gate`。")
    lines.append("4. 创新点不是固定稀疏注意力规则，而是在线反事实候选评估预算分配。")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    rows = [summarize_case(*case) for case in CASES]
    write_csv(rows, DOCS / "horizon_pcic_key_results_2026_06_29.csv")
    write_markdown(rows, DOCS / "horizon_pcic_key_results_2026_06_29.md")
    print(DOCS / "horizon_pcic_key_results_2026_06_29.md")


if __name__ == "__main__":
    main()
