#!/usr/bin/env python
"""Generate auditable LaTeX tables from frozen quality summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import make_qksieve_quality_generalization_figure as quality_figure


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = (
    ROOT.parents[1]
    / "projects"
    / "qwen3_top2_head_limit3_ppl"
    / "src"
)
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import verify_qksieve_robust_paper_evidence_20260810 as evidence_verify


MODEL_LABELS = {
    "llama31_8b": "Llama-3.1-8B",
    "qwen3_4b": "Qwen3-4B",
    "mistral_7b": "Mistral-7B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--longbench",
        type=Path,
        default=ROOT / "data" / "qksieve_robust_longbench_summary.json",
    )
    parser.add_argument(
        "--ruler",
        type=Path,
        default=ROOT / "data" / "qksieve_robust_ruler_summary.json",
    )
    parser.add_argument(
        "--multimodel",
        type=Path,
        default=ROOT / "data" / "qksieve_robust_multimodel_summary.json",
    )
    parser.add_argument(
        "--output_en",
        type=Path,
        default=ROOT / "data" / "generated" / "qksieve_quality_tables.tex",
    )
    parser.add_argument(
        "--output_zh",
        type=Path,
        default=(
            ROOT / "data" / "generated" / "qksieve_quality_tables_zh.tex"
        ),
    )
    parser.add_argument(
        "--appendix_en",
        type=Path,
        default=(
            ROOT / "data" / "generated" / "qksieve_quality_appendix.tex"
        ),
    )
    parser.add_argument(
        "--appendix_zh",
        type=Path,
        default=(
            ROOT
            / "data"
            / "generated"
            / "qksieve_quality_appendix_zh.tex"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value: Any, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}\\%"


def interval(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "--"
    return f"[{percent(value[0])}, {percent(value[1])}]"


def longbench_rows(longbench: dict[str, Any]) -> list[str]:
    methods = longbench["methods"]
    full = methods["full_kv"]
    qksieve_names = [name for name in methods if name != "full_kv"]
    if len(qksieve_names) != 1:
        raise ValueError("LongBench summary must contain one frozen QKSieve method")
    ours = methods[qksieve_names[0]]
    return [
        "Llama-3.1-8B & {:.4f} & {:.4f} & {} & {} \\\\".format(
            float(full["macro_score"]),
            float(ours["macro_score"]),
            percent(ours["quality_retention"]),
            interval(longbench["bootstrap"]["quality_retention_95ci"]),
        )
    ]


def ruler_rows(ruler: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for length in sorted(int(item) for item in ruler["per_length"]):
        result = ruler["per_length"][str(length)]
        rows.append(
            "{}K & {:.4f} & {:.4f} & {} & {} \\\\".format(
                length // 1024,
                float(result["full_macro"]),
                float(result["qksieve_macro"]),
                percent(result["quality_retention"]),
                interval(result["bootstrap"]["quality_retention_95ci"]),
            )
        )
    overall = ruler["overall"]
    rows.append("\\midrule")
    rows.append(
        "Overall & {:.4f} & {:.4f} & {} & {} \\\\".format(
            float(overall["full_macro"]),
            float(overall["qksieve_macro"]),
            percent(overall["quality_retention"]),
            interval(ruler["bootstrap"]["quality_retention_95ci"]),
        )
    )
    return rows


def model_rows(multimodel: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for model in quality_figure.MODEL_ORDER:
        result = multimodel["models"][model]
        rows.append(
            "{} & {:.4f} & {:.4f} & {} & {} \\\\".format(
                MODEL_LABELS[model],
                float(result["full_macro"]),
                float(result["qksieve_macro"]),
                percent(result["quality_retention"]),
                interval(result["quality_retention_95ci"]),
            )
        )
    return rows


def compact_rows(
    longbench: dict[str, Any],
    ruler: dict[str, Any],
    multimodel: dict[str, Any],
    *,
    chinese: bool,
) -> list[str]:
    methods = longbench["methods"]
    full = methods["full_kv"]
    qksieve_names = [name for name in methods if name != "full_kv"]
    if len(qksieve_names) != 1:
        raise ValueError("LongBench summary must contain one frozen QKSieve method")
    ours = methods[qksieve_names[0]]
    complete_label = (
        "完整 LB / Llama-3.1-8B"
        if chinese
        else "Full LB / Llama-3.1-8B"
    )
    ruler_label = "RULER / Llama-3.1-8B"
    screen_prefix = "LB 筛查" if chinese else "LB screen"
    rows = [
        "{} & {:.4f} & {:.4f} & {} & {} \\\\".format(
            complete_label,
            float(full["macro_score"]),
            float(ours["macro_score"]),
            percent(ours["quality_retention"]),
            interval(longbench["bootstrap"]["quality_retention_95ci"]),
        )
    ]
    ruler_overall = ruler["overall"]
    rows.append(
        "{} & {:.4f} & {:.4f} & {} & {} \\\\".format(
            ruler_label,
            float(ruler_overall["full_macro"]),
            float(ruler_overall["qksieve_macro"]),
            percent(ruler_overall["quality_retention"]),
            interval(ruler["bootstrap"]["quality_retention_95ci"]),
        )
    )
    rows.append("\\midrule")
    for model in quality_figure.MODEL_ORDER:
        result = multimodel["models"][model]
        rows.append(
            "{} / {} & {:.4f} & {:.4f} & {} & {} \\\\".format(
                screen_prefix,
                MODEL_LABELS[model],
                float(result["full_macro"]),
                float(result["qksieve_macro"]),
                percent(result["quality_retention"]),
                interval(result["quality_retention_95ci"]),
            )
        )
    return rows


def render_main(
    longbench: dict[str, Any],
    ruler: dict[str, Any],
    multimodel: dict[str, Any],
    *,
    chinese: bool,
    provenance: str,
) -> str:
    if chinese:
        caption = (
            "冻结 QKSieve-Robust 的任务质量。完整 LongBench、RULER 和跨模型筛查"
            "分别包含 3,750、650 和每模型 160 个严格配对；区间按任务或"
            "任务--长度单元 bootstrap。"
        )
        evaluation = "评测 / 模型"
        retention = "保持率"
    else:
        caption = (
            "Task quality for frozen QKSieve-Robust. Complete LongBench, "
            "RULER, and each cross-model screen contain 3,750, 650, and 160 "
            "strict pairs; intervals bootstrap tasks or task--length cells."
        )
        evaluation = "Evaluation / model"
        retention = "Retention"
    lines = [
        f"% Generated from frozen evidence: {provenance}",
        "\\begin{table}[t]",
        f"\\caption{{{caption}}}",
        "\\label{tab:quality-main}",
        "\\centering",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        f"{evaluation} & Full & QKSieve & {retention} & Paired 95\\% CI \\\\",
        "\\midrule",
        *compact_rows(longbench, ruler, multimodel, chinese=chinese),
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def render(
    longbench: dict[str, Any],
    ruler: dict[str, Any],
    multimodel: dict[str, Any],
    *,
    chinese: bool,
    provenance: str,
) -> str:
    if chinese:
        longbench_caption = (
            "冻结 QKSieve-Robust 的完整 LongBench 结果，覆盖 16 个任务和 "
            "3,750 个严格配对。区间为 paired-bootstrap 95\\% CI。"
        )
        ruler_caption = (
            "冻结 QKSieve-Robust 的正式 RULER 结果。区间为 paired-bootstrap "
            "95\\% CI。"
        )
        model_caption = (
            "同一冻结配置的跨模型 LongBench screen，每个模型 160 个严格配对。"
        )
        history = "长度"
        model_header = "模型"
        retention = "保持率"
    else:
        longbench_caption = (
            "Complete LongBench results for frozen QKSieve-Robust over 16 "
            "tasks and 3,750 strict pairs. The interval is a paired-bootstrap "
            "95\\% CI."
        )
        ruler_caption = (
            "Formal RULER results for frozen QKSieve-Robust. Intervals are "
            "paired-bootstrap 95\\% CIs."
        )
        model_caption = (
            "Cross-model LongBench screen under one frozen configuration, "
            "with 160 strict pairs per model."
        )
        history = "Length"
        model_header = "Model"
        retention = "Retention"

    lines = [
        f"% Generated from frozen evidence: {provenance}",
        "\\begin{table}[t]",
        f"\\caption{{{longbench_caption}}}",
        "\\label{tab:longbench-main}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        f"{model_header} & Full & QKSieve & {retention} & Paired 95\\% CI \\\\",
        "\\midrule",
        *longbench_rows(longbench),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        f"\\caption{{{ruler_caption}}}",
        "\\label{tab:ruler-formal}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        f"{history} & Full & QKSieve & {retention} & Paired 95\\% CI \\\\",
        "\\midrule",
        *ruler_rows(ruler),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        f"\\caption{{{model_caption}}}",
        "\\label{tab:multimodel-formal}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        f"{model_header} & Full & QKSieve & {retention} & Paired 95\\% CI \\\\",
        "\\midrule",
        *model_rows(multimodel),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    longbench = read_json(args.longbench)
    ruler = read_json(args.ruler)
    multimodel = read_json(args.multimodel)
    evidence_verify.validate_longbench(longbench)
    evidence_verify.validate_ruler(ruler)
    evidence_verify.validate_multimodel(multimodel)
    quality_figure.validate(ruler, multimodel)
    provenance = (
        f"longbench={sha256(args.longbench)}; ruler={sha256(args.ruler)}; "
        f"multimodel={sha256(args.multimodel)}"
    )
    for output, chinese in ((args.output_en, False), (args.output_zh, True)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render_main(
                longbench,
                ruler,
                multimodel,
                chinese=chinese,
                provenance=provenance,
            ),
            encoding="utf-8",
        )
    for output, chinese in (
        (args.appendix_en, False),
        (args.appendix_zh, True),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render(
                longbench,
                ruler,
                multimodel,
                chinese=chinese,
                provenance=provenance,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
