#!/usr/bin/env python
"""Generate auditable H100 system tables from matched frozen evidence."""

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
    validate_h100,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "data" / "qksieve_h100_matched_summary.json",
    )
    parser.add_argument(
        "--output_en",
        type=Path,
        default=ROOT / "data" / "generated" / "qksieve_h100_tables.tex",
    )
    parser.add_argument(
        "--output_zh",
        type=Path,
        default=ROOT / "data" / "generated" / "qksieve_h100_tables_zh.tex",
    )
    return parser.parse_args()


def percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}\\%"


def speed(value: Any) -> str:
    return f"{float(value):.2f}$\\times$"


def memory_ratio(value: Any) -> str:
    return f"{float(value):.3f}$\\times$"


def render(summary: dict[str, Any], *, chinese: bool, provenance: str) -> str:
    if chinese:
        attention_caption = (
            "匹配 H100 上直接计时的原生 MHA attention；所有路径使用相同常驻 K/V。"
        )
        decode_caption = "匹配 H100 上的整模型稳态 decode 与 CUDA 峰值显存。"
        request_caption = (
            "匹配 H100 上的请求生命周期。Cold E2E 从 dense prefill 前开始计时。"
        )
        history = "长度"
        prebuild = "构建 (s)"
        peak = "峰值显存比"
        cold = "Cold 构建+解码"
        cold_e2e = "Cold E2E"
        warm = "Warm"
        shared = "共享前缀"
        append = "追加"
        auxiliary = "辅助索引"
    else:
        attention_caption = (
            "Direct native-MHA attention timing on matched H100s; all paths "
            "use the same resident K/V."
        )
        decode_caption = (
            "Whole-model steady decode and CUDA peak memory on matched H100s."
        )
        request_caption = (
            "Matched-H100 request lifecycle. Cold E2E starts before dense prefill."
        )
        history = "History"
        prebuild = "Build (s)"
        peak = "Peak mem. ratio"
        cold = "Cold build+decode"
        cold_e2e = "Cold E2E"
        warm = "Warm"
        shared = "Shared prefix"
        append = "Append"
        auxiliary = "Aux. index"

    attention_rows = []
    for row in summary["attention"]:
        attention_rows.append(
            "{}K & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {} & {} & {} & {} \\\\".format(
                int(row["history_tokens"]) // 1024,
                float(row["full_mha_ms"]),
                float(row["qksieve_fast_ms"]),
                float(row["qksieve_robust_ms"]),
                float(row["fier_ms"]),
                speed(row["fast_speedup"]),
                speed(row["robust_speedup"]),
                speed(row["robust_vs_fier"]),
                percent(row["qksieve_total_auxiliary_ratio_of_full_kv"]),
            )
        )

    decode_rows = []
    for row in summary["steady_decode"]:
        decode_rows.append(
            "{}K & {:.3f} & {:.3f} & {} & {:.3f} & {} \\\\".format(
                int(row["history_tokens"]) // 1024,
                float(row["full_steady_ms_per_token"]),
                float(row["qksieve_steady_ms_per_token"]),
                speed(row["steady_decode_speedup"]),
                float(row["qksieve_prebuild_seconds_median"]),
                memory_ratio(row["qksieve_to_full_peak_allocated_ratio"]),
            )
        )

    request_rows = []
    for row in summary["persistent_requests"]:
        request_rows.append(
            "{}K & {} & {} & {} & {} & {} & {} \\\\".format(
                int(row["history_tokens"]) // 1024,
                speed(row["cold_speedup"]),
                speed(row["cold_end_to_end_speedup"]),
                speed(row["warm_speedup"]),
                speed(row["shared_prefix_amortized_speedup"]),
                speed(row["append_only_speedup"]),
                memory_ratio(row["qksieve_to_full_cold_peak_allocated_ratio"]),
            )
        )

    return "\n".join(
        [
            f"% Generated from frozen H100 evidence: {provenance}",
            "\\begin{table*}[t]",
            f"\\caption{{{attention_caption}}}",
            "\\label{tab:h100-attention}",
            "\\centering\\small",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            f"{history} & Full & Fast & Robust & FIER & Fast sp. & Robust sp. & Robust/FIER & {auxiliary} \\\\",
            "\\midrule",
            *attention_rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table*}",
            "",
            "\\begin{table}[t]",
            f"\\caption{{{decode_caption}}}",
            "\\label{tab:h100-decode}",
            "\\centering\\small",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            f"{history} & Full & Robust & Speedup & {prebuild} & {peak} \\\\",
            "\\midrule",
            *decode_rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
            "",
            "\\begin{table*}[t]",
            f"\\caption{{{request_caption}}}",
            "\\label{tab:h100-requests}",
            "\\centering\\small",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            f"{history} & {cold} & {cold_e2e} & {warm} & {shared} & {append} & {peak} \\\\",
            "\\midrule",
            *request_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    validate_h100(summary)
    provenance = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    for output, chinese in ((args.output_en, False), (args.output_zh, True)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render(summary, chinese=chinese, provenance=provenance),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
