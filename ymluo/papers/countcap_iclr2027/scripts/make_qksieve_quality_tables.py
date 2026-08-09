#!/usr/bin/env python
"""Generate auditable LaTeX tables from frozen quality summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import make_qksieve_quality_generalization_figure as quality_figure


ROOT = Path(__file__).resolve().parents[1]
MODEL_LABELS = {
    "llama31_8b": "Llama-3.1-8B",
    "qwen3_4b": "Qwen3-4B",
    "mistral_7b": "Mistral-7B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def render(
    ruler: dict[str, Any],
    multimodel: dict[str, Any],
    *,
    chinese: bool,
    provenance: str,
) -> str:
    if chinese:
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
    ruler = read_json(args.ruler)
    multimodel = read_json(args.multimodel)
    quality_figure.validate(ruler, multimodel)
    provenance = f"ruler={sha256(args.ruler)}; multimodel={sha256(args.multimodel)}"
    for output, chinese in ((args.output_en, False), (args.output_zh, True)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render(
                ruler,
                multimodel,
                chinese=chinese,
                provenance=provenance,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
