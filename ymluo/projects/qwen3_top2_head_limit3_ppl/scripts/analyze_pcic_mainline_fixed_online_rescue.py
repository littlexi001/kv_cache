#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT_CSV = DOCS / "pcic_mainline_fixed_online_rescue_2026_06_29.csv"
OUT_MD = DOCS / "pcic_mainline_fixed_online_rescue_2026_06_29.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def split_combos(value: str) -> list[str]:
    if not value:
        return []
    sep = "/" if "/" in value else ";"
    return [item.strip() for item in value.split(sep) if item.strip()]


def fmt(value: float) -> str:
    if abs(value) < 0.5e-6:
        value = 0.0
    return f"{value:.6f}"


def main() -> None:
    fixed_rows = {
        row["case_id"]: row
        for row in read_csv(DOCS / "pcic_fixed_online_oracle_2026_06_29.csv")
    }
    corrected_rows = {
        row["case_id"]: row
        for row in read_csv(DOCS / "pcic_corrected_gate_core_results_2026_06_29.csv")
    }

    cases = [
        {
            "case_id": "hardtopic_eval128",
            "label": "Hard-topic eval128",
            "top2_id": "hard_top2",
            "cond_id": "hard_cond",
            "interpretation": "conditional rescue 修复 top2 delayed-win failure，并达到 blockwise oracle。",
        },
        {
            "case_id": "war",
            "label": "War and Peace",
            "top2_id": "war_top2",
            "cond_id": "war_cond",
            "interpretation": "easy regime；固定策略、top2 和 conditional rescue 质量相同，主要用于证明 rescue gate 不破坏质量。",
        },
        {
            "case_id": "monte",
            "label": "Count of Monte Cristo",
            "top2_id": "monte_top2",
            "cond_id": "monte_cond",
            "interpretation": "online blockwise selection 明显优于 best fixed，说明不是离线固定 combo 可替代。",
        },
    ]

    out_rows: list[dict[str, Any]] = []
    for case in cases:
        fixed = fixed_rows[case["case_id"]]
        top2 = corrected_rows[case["top2_id"]]
        cond = corrected_rows[case["cond_id"]]

        cond_combos = split_combos(cond["combos"])
        top2_combos = split_combos(top2["combos"])
        fixed_delta = fnum(fixed["best_fixed_delta_ppl"])
        oracle_delta = fnum(fixed["oracle_delta_ppl"])
        top2_delta = fnum(top2["avg_delta_ppl"])
        cond_delta = fnum(cond["avg_delta_ppl"])

        out_rows.append(
            {
                "case_id": case["case_id"],
                "label": case["label"],
                "blocks": cond["blocks"],
                "best_fixed_combo": fixed["best_fixed_combo"],
                "best_fixed_delta_ppl": fixed_delta,
                "top2_delta_ppl": top2_delta,
                "cond_rescue_delta_ppl": cond_delta,
                "oracle_delta_ppl": oracle_delta,
                "cond_minus_fixed": cond_delta - fixed_delta,
                "cond_oracle_gap": cond_delta - oracle_delta,
                "top2_oracle_gap": top2_delta - oracle_delta,
                "cond_corrected_gate_s": fnum(cond["corrected_gate_s"]),
                "cond_corrected_method_ratio": fnum(cond["corrected_method_ratio"]),
                "cond_extended": cond["extended"],
                "cond_unique_combos": len(set(cond_combos)),
                "cond_combo_switches": sum(
                    1 for prev, cur in zip(cond_combos, cond_combos[1:]) if prev != cur
                ),
                "top2_combos": "/".join(top2_combos),
                "cond_combos": "/".join(cond_combos),
                "oracle_combos": fixed["oracle_combos"].replace(";", "/"),
                "interpretation": case["interpretation"],
            }
        )

    fieldnames = list(out_rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    lines: list[str] = []
    lines.append("# PCIC 主线：Fixed / Online / Rescue / Oracle 证据表（2026-06-29）")
    lines.append("")
    lines.append("目的：把 paper 主线收敛到一个可辩护的核心命题：")
    lines.append("")
    lines.append("> Pairwise-CIC + online blockwise selection + conditional rescue gate 不是固定 sparse attention combo 的小改，而是一个在线策略选择器；rescue gate 用额外 horizon probe 修复短视选择。")
    lines.append("")
    lines.append("本表只使用已有本地 CSV 合成，不重新跑模型，不访问网络。speed/gate 统一采用 corrected gate 口径。")
    lines.append("")
    lines.append("## 主结果")
    lines.append("")
    lines.append("| dataset | best fixed | fixed ΔPPL | top2 ΔPPL | conditional rescue ΔPPL | oracle ΔPPL | cond-fixed | cond-oracle gap | corrected gate_s | unique combos | switches |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in out_rows:
        lines.append(
            "| {label} | `{best_fixed_combo}` | {fixed} | {top2} | {cond} | {oracle} | {cond_fixed} | {gap} | {gate:.3f} | {uniq} | {switches} |".format(
                label=row["label"],
                best_fixed_combo=row["best_fixed_combo"],
                fixed=fmt(row["best_fixed_delta_ppl"]),
                top2=fmt(row["top2_delta_ppl"]),
                cond=fmt(row["cond_rescue_delta_ppl"]),
                oracle=fmt(row["oracle_delta_ppl"]),
                cond_fixed=fmt(row["cond_minus_fixed"]),
                gap=fmt(row["cond_oracle_gap"]),
                gate=row["cond_corrected_gate_s"],
                uniq=row["cond_unique_combos"],
                switches=row["cond_combo_switches"],
            )
        )

    lines.append("")
    lines.append("## 逐项解释")
    lines.append("")
    for row in out_rows:
        lines.append(f"### {row['label']}")
        lines.append("")
        lines.append(f"- 结论：{row['interpretation']}")
        lines.append(f"- top2 combos：`{row['top2_combos']}`")
        lines.append(f"- conditional rescue combos：`{row['cond_combos']}`")
        lines.append(f"- oracle combos：`{row['oracle_combos']}`")
        lines.append("")

    lines.append("## 对论文创新性的含义")
    lines.append("")
    lines.append("- **不是 SparQ/qabs 固定算子的复述**：主贡献应写成 online policy selection，而不是某个固定 candidate。")
    lines.append("- **不是 best fixed combo 可替代**：Monte 上 conditional/online 比 best fixed 改善 `-0.210210` ΔPPL；Hard-topic eval128 上 conditional rescue 比 best fixed 改善 `-0.028889` ΔPPL。")
    lines.append("- **rescue gate 有必要性**：Hard-topic eval128 中 top2 与 oracle gap 为 `0.054004`，conditional rescue 将 gap 降到 `0.000000`。")
    lines.append("- **动态性证据仍需加强**：War 是 easy regime，固定策略已经足够；后续需要更多非平稳任务证明 blockwise switch 普遍有效。")
    lines.append("- **速度 claim 必须保守**：conditional rescue corrected gate 成本仍高，paper 现在只能声称方法学和质量证据，不能声称端到端速度已超过 baseline。")
    lines.append("")
    lines.append("## 下一步最关键实验")
    lines.append("")
    lines.append("1. 在标准或准标准长上下文任务上补 `best fixed / online / oracle` 三者对比。")
    lines.append("2. 输出 blockwise trace 图，证明策略切换与文本结构、horizon risk 有对应关系。")
    lines.append("3. 做 fused/sparse candidate probe，把 corrected gate 从方法瓶颈变成可控 overhead。")
    lines.append("")
    lines.append(f"CSV：`docs/{OUT_CSV.name}`")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
