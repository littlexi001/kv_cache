#!/usr/bin/env python
"""Generate paired query-shrinkage sensitivity tables for the appendix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = (
    ROOT.parents[1] / "projects" / "qwen3_top2_head_limit3_ppl" / "src"
)
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from verify_qksieve_robust_paper_evidence_20260810 import (  # noqa: E402
    validate_shrinkage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "data" / "qksieve_shrinkage_sensitivity_summary.json",
    )
    parser.add_argument(
        "--output_en",
        type=Path,
        default=ROOT / "data" / "generated" / "qksieve_shrinkage_tables.tex",
    )
    parser.add_argument(
        "--output_zh",
        type=Path,
        default=(
            ROOT / "data" / "generated" / "qksieve_shrinkage_tables_zh.tex"
        ),
    )
    return parser.parse_args()


def percent(value: Any, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}\\%"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def render(summary: dict[str, Any], *, chinese: bool, provenance: str) -> str:
    aggregate = sorted(
        summary["aggregate"],
        key=lambda row: (float(row["selected_fraction"]), float(row["shrinkage"])),
    )
    production = {
        float(row["selected_fraction"]): row
        for row in aggregate
        if abs(float(row["shrinkage"]) - 0.75) < 1.0e-12
    }
    checks = sorted(
        summary["acceptance"]["checks"],
        key=lambda row: float(row["selected_fraction"]),
    )

    grid_rows = []
    for row in aggregate:
        shrinkage = float(row["shrinkage"])
        cells = [
            percent(row["selected_fraction"]),
            f"{shrinkage:.2f}",
            percent(row["top2_recall"]),
            percent(row["selected_attention_mass"]),
            percent(row["top2_attention_mass_recall"]),
            f"{float(row['score_pearson']):.4f}",
            f"{float(row['score_rmse']):.4f}",
        ]
        if abs(shrinkage - 0.75) < 1.0e-12:
            cells = [f"\\textbf{{{cell}}}" for cell in cells]
        grid_rows.append(" & ".join(cells) + r" \\")

    check_rows = []
    for row in checks:
        fraction = float(row["selected_fraction"])
        prod = production[fraction]
        check_rows.append(
            "{} & {} & {} & {} & {} & {:.3f}$\\times$ \\\\".format(
                percent(fraction),
                percent(prod["top2_recall"]),
                percent(row["production_recall_regret"]),
                percent(prod["selected_attention_mass"]),
                percent(row["production_mass_regret"]),
                float(row["production_rmse_ratio_to_best"]),
            )
        )

    passed = bool(summary["acceptance"]["passed"])
    failures = [str(value) for value in summary["acceptance"].get("failures", [])]
    if chinese:
        grid_caption = (
            "固定 Query shrinkage 的严格配对敏感性。四条 32K 真实文本轨迹覆盖"
            " Llama/Qwen 与体育/医学；每个 $\\lambda$ 使用相同层、解码步、Query head"
            " 和 KV head。粗体为冻结的 $\\lambda=0.75$。"
        )
        check_caption = (
            "冻结 $\\lambda=0.75$ 相对五个候选 shrinkage 中最优值的 regret。"
        )
        status = "通过" if passed else "未通过"
        failure_text = "；失败项：" + "；".join(failures) if failures else ""
        conclusion = (
            f"预注册稳定性检查{status}{failure_text}。该实验只检验 selector 的数值"
            "敏感性，不能替代 LongBench、RULER 或 PPL 的下游质量证据。选取比例"
            "为 4% 时，Top-2 质量召回可以超过 100%，因为分子使用选中 4% "
            "token 的 attention mass，分母仍是 oracle Top-2% mass。"
        )
        subsection = "固定 Query shrinkage 敏感性"
        fraction_label = "选取比例"
        recall_label = "Top-2 召回"
        mass_label = "选中质量"
        top2_mass_label = "Top-2 质量召回"
        recall_regret = "召回 regret"
        mass_regret = "质量 regret"
    else:
        grid_caption = (
            "Strictly paired sensitivity to fixed Query shrinkage on four real "
            "32K traces spanning Llama/Qwen and sports/medicine. Every $\\lambda$ "
            "uses the same layer, decode step, Query head, and KV head. Bold marks "
            "the frozen $\\lambda=0.75$."
        )
        check_caption = (
            "Regret of frozen $\\lambda=0.75$ relative to the best of five "
            "shrinkage candidates."
        )
        status = "passed" if passed else "failed"
        failure_text = "; failures: " + "; ".join(failures) if failures else ""
        conclusion = (
            f"The preregistered stability check {status}{failure_text}. This is a "
            "selector-level numerical sensitivity test; it does not replace "
            "downstream LongBench, RULER, or PPL evidence. At 4% selected, "
            "Top-2 mass recall can exceed 100% because its numerator covers "
            "4% of tokens while the denominator remains oracle Top-2% mass."
        )
        subsection = "Fixed Query-shrinkage sensitivity"
        fraction_label = "Selected"
        recall_label = "Top-2 recall"
        mass_label = "Selected mass"
        top2_mass_label = "Top-2 mass recall"
        recall_regret = "Recall regret"
        mass_regret = "Mass regret"

    return "\n".join(
        [
            f"% Generated from paired shrinkage evidence: {provenance}",
            f"\\subsection{{{subsection}}}",
            "\\label{app:shrinkage-sensitivity}",
            "\\begin{table*}[t]",
            f"\\caption{{{grid_caption}}}",
            "\\label{tab:shrinkage-grid}",
            "\\centering\\small",
            "\\begin{tabular}{rrrrrrr}",
            "\\toprule",
            f"{fraction_label} & $\\lambda$ & {recall_label} & {mass_label} & "
            f"{top2_mass_label} & Pearson & RMSE \\\\",
            "\\midrule",
            *grid_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "\\begin{table}[t]",
            f"\\caption{{{check_caption}}}",
            "\\label{tab:shrinkage-regret}",
            "\\centering\\small",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{rrrrrr}",
            "\\toprule",
            f"{fraction_label} & {recall_label} & {recall_regret} & "
            f"{mass_label} & {mass_regret} & RMSE/best \\\\",
            "\\midrule",
            *check_rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
            "",
            latex_escape(conclusion),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    validate_shrinkage(summary)
    provenance = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    for output, chinese in ((args.output_en, False), (args.output_zh, True)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render(summary, chinese=chinese, provenance=provenance),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
