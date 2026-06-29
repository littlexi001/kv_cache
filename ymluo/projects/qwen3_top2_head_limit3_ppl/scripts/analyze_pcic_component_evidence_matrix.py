#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT_CSV = DOCS / "pcic_component_evidence_matrix_2026_06_29.csv"
OUT_MD = DOCS / "pcic_component_evidence_matrix_2026_06_29.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    data = path.read_bytes()
    encoding = "utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    return list(csv.DictReader(io.StringIO(data.decode(encoding))))


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def fmt(value: float) -> str:
    if abs(value) < 0.5e-6:
        value = 0.0
    return f"{value:.6f}"


def main() -> None:
    fixed = by_key(read_csv(DOCS / "pcic_mainline_fixed_online_rescue_2026_06_29.csv"), "case_id")
    corrected = by_key(read_csv(DOCS / "pcic_corrected_gate_core_results_2026_06_29.csv"), "case_id")
    delayed = by_key(read_csv(DOCS / "pcic_delayed_rescue_eval128_2026_06_29.csv"), "case_id")
    ruler = by_key(read_csv(DOCS / "pcic_ruler_style_smoke_2026_06_29.csv"), "case_id")
    margin_rows = read_csv(DOCS / "pcic_conditional_auto_anchor_margin_grid_2026_06_29.csv")
    skip = by_key(read_csv(DOCS / "pcic_corrected_gate_skipanchor_gain_2026_06_29.csv"), "case_id")

    hard = fixed["hardtopic_eval128"]
    monte = fixed["monte"]
    hard_top2 = corrected["hard_top2"]
    hard_cond = corrected["hard_cond"]
    variable_top2 = ruler["variable_top2"]
    variable_cond = ruler["variable_cond"]

    margin_0012 = [
        row for row in margin_rows if row.get("margin") == "0.012"
    ]
    hard_m0012 = next(row for row in margin_0012 if row["dataset"] == "Hard-topic eval128")
    war_m0012 = next(row for row in margin_0012 if row["dataset"] == "War")
    monte_m0012 = next(row for row in margin_0012 if row["dataset"] == "Monte")

    rows: list[dict[str, str]] = [
        {
            "component": "Online blockwise selection",
            "claim": "不是 best fixed combo 可替代；policy 会随 block/任务变化。",
            "positive_evidence": (
                "Monte cond/online vs best fixed ΔPPL "
                f"{fmt(fnum(monte['cond_minus_fixed']))}；"
                "Hard-topic eval128 cond vs best fixed ΔPPL "
                f"{fmt(fnum(hard['cond_minus_fixed']))}。"
            ),
            "negative_or_boundary": "War 是 easy regime，fixed=online=oracle，说明动态选择收益依赖非平稳文本。",
            "status": "supported_but_needs_standard_benchmark",
            "source": "docs/pcic_mainline_fixed_online_rescue_2026_06_29.md",
        },
        {
            "component": "Conditional horizon rescue gate",
            "claim": "短 horizon/top2 会短视；rescue gate 能修复 delayed-win failure。",
            "positive_evidence": (
                "Hard-topic eval128 top2 ΔPPL "
                f"{fmt(fnum(hard_top2['avg_delta_ppl']))} -> cond "
                f"{fmt(fnum(hard_cond['avg_delta_ppl']))}；"
                "cond-oracle gap "
                f"{fmt(fnum(hard['cond_oracle_gap']))}。"
            ),
            "negative_or_boundary": (
                "corrected gate_s 从 top2 "
                f"{fnum(hard_top2['corrected_gate_s']):.3f} 增至 cond "
                f"{fnum(hard_cond['corrected_gate_s']):.3f}，速度仍是瓶颈。"
            ),
            "status": "quality_supported_speed_not_solved",
            "source": "docs/horizon_pcic_corrected_key_results_2026_06_29.md",
        },
        {
            "component": "Validation-prior anchor",
            "claim": "anchor 不是手工固定，而是 validation prior；conditional gate 避免 easy case 无意义扩展。",
            "positive_evidence": (
                "margin=0.012: Hard-topic ΔPPL "
                f"{fmt(fnum(hard_m0012['avg_delta_ppl']))}；War ΔPPL "
                f"{fmt(fnum(war_m0012['avg_delta_ppl']))}；Monte ΔPPL "
                f"{fmt(fnum(monte_m0012['avg_delta_ppl']))}。"
            ),
            "negative_or_boundary": "阈值 0.012 仍是经验选择，需要标准验证集或自适应 margin 规则。",
            "status": "supported_with_threshold_risk",
            "source": "docs/pcic_conditional_auto_anchor_margin_grid_2026_06_29.md",
        },
        {
            "component": "RULER-style variable rescue",
            "claim": "rescue/anchor 对 variable binding 类长上下文失败有效，不只是在 hard-topic 文本有效。",
            "positive_evidence": (
                "RULER-style variable top2 ΔPPL "
                f"{fmt(fnum(variable_top2['avg_delta_ppl']))} -> cond "
                f"{fmt(fnum(variable_cond['avg_delta_ppl']))}。"
            ),
            "negative_or_boundary": "这是 synthetic/offline smoke，不是正式 RULER；只能作为机制证据。",
            "status": "mechanism_supported_not_formal_benchmark",
            "source": "docs/pcic_ruler_style_smoke_2026_06_29.md",
        },
        {
            "component": "Skip/early-exit heuristics",
            "claim": "简单 early-exit 不能替代主 rescue gate；需要 calibrated skip-gate。",
            "positive_evidence": (
                "skip rule 在 corrected gate 上能省成本，如 needle "
                f"{fnum(skip['needle_base']['corrected_gate_s']):.3f} -> "
                f"{fnum(skip['needle_skip']['corrected_gate_s']):.3f}。"
            ),
            "negative_or_boundary": (
                "needle ΔPPL 退化 "
                f"{fmt(fnum(skip['needle_skip']['avg_delta_ppl']) - fnum(skip['needle_base']['avg_delta_ppl']))}；"
                "因此默认关闭，不作为主方法。"
            ),
            "status": "negative_ablation_guides_future_speed",
            "source": "docs/pcic_corrected_gate_skipanchor_gain_2026_06_29.md",
        },
        {
            "component": "Pairwise-CIC / risk memory ablation",
            "claim": "论文需证明 pairwise calibration 与 memory prior 均不可或缺。",
            "positive_evidence": "当前 Method spec 已定义；已有 fixed/online/oracle 与 trace 间接支持 policy selection。",
            "negative_or_boundary": "缺少直接 no-pairwise、no-memory 消融；这是 ICML 证据链的 P0 缺口。",
            "status": "missing_direct_ablation",
            "source": "docs/horizon_pcic_method_spec_2026_06_29.md",
        },
    ]

    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines: list[str] = []
    lines.append("# PCIC Component Evidence Matrix（2026-06-29）")
    lines.append("")
    lines.append("目的：把 paper 的三个核心贡献拆成可审稿的 evidence / boundary / missing ablation，避免把尚未证明的内容写成强 claim。")
    lines.append("")
    lines.append("## 总表")
    lines.append("")
    lines.append("| component | status | positive evidence | boundary / missing | source |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['component']} | `{row['status']}` | {row['positive_evidence']} | {row['negative_or_boundary']} | `{row['source']}` |"
        )

    lines.append("")
    lines.append("## 论文写法建议")
    lines.append("")
    lines.append("- 可以强写：`online blockwise policy selection`、`conditional horizon rescue gate`、`fixed policy 不足以覆盖非平稳文本`。")
    lines.append("- 可以作为机制证据写：RULER-style variable binding smoke、blockwise policy trace、delayed-win case study。")
    lines.append("- 必须保守写：速度。corrected gate 后 conditional rescue 仍明显慢，当前不能声称端到端快于 baseline。")
    lines.append("- 不能强写：Pairwise-CIC/risk memory 的直接必要性，直到补齐 no-pairwise/no-memory 消融。")
    lines.append("")
    lines.append("## 下一步最小实验矩阵")
    lines.append("")
    lines.append("| ablation | 目的 | 最小数据 | 成功标准 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| no-rescue | 证明 rescue gate 必要 | Hard-topic eval128 + RULER variable | top2/no-rescue 明显差于 conditional rescue |")
    lines.append("| no-memory | 证明 historical prior 必要 | Monte + hard-topic b8 | 去掉 memory 后 block trace 更不稳或 PPL drift 变差 |")
    lines.append("| no-pairwise | 证明 Pairwise-CIC 不是普通 ranking | Monte + RULER variable | 非 pairwise scorer 更接近 fixed 或 short-horizon failure |")
    lines.append("| fixed-best | 证明不是固定 combo | 正式 LongBench/RULER subset | online 接近 oracle 且优于 best fixed |")
    lines.append("| fused-probe | 证明系统可行 | hard/war/monte | corrected gate 或 tokens/s 接近 baseline |")
    lines.append("")
    lines.append("## 当前结论")
    lines.append("")
    lines.append("Horizon-PCIC 的 paper 主线已经具备方法创新性雏形：它把 KV compression 从固定稀疏规则提升为在线反事实策略选择。")
    lines.append("但 ICML 级投稿还缺两个硬证据：")
    lines.append("")
    lines.append("1. `no-pairwise / no-memory` 直接消融；")
    lines.append("2. 正式 benchmark 与真实速度/fused probe。")
    lines.append("")
    lines.append(f"CSV：`docs/{OUT_CSV.name}`")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
