#!/usr/bin/env python
"""Summarize paired Full, rank-16, and block-conditional PPL probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FULL = "full_attention"
REFERENCE = "qksieve_qmse_oas_requestlocal_valuesketch16_k1280"
CANDIDATE = "qksieve_qmse_oas_requestlocal_blockcondres8_r8_m8_k1120"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--output_markdown", type=Path)
    return parser.parse_args()


def load_cases(run_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    cases: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for directory in sorted(run_root.glob("n*_seed*")):
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            incomplete.append(directory.name)
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        indexed = {str(row.get("variant")): row for row in payload["rows"]}
        missing = sorted({FULL, REFERENCE, CANDIDATE} - set(indexed))
        if missing or not (directory / "ALL_COMPLETE").exists():
            incomplete.append(f"{directory.name}:missing={','.join(missing)}")
            continue
        full = indexed[FULL]
        reference = indexed[REFERENCE]
        candidate = indexed[CANDIDATE]
        full_ppl = float(full["ppl"])
        reference_ppl = float(reference["ppl"])
        candidate_ppl = float(candidate["ppl"])
        cases.append(
            {
                "case": directory.name,
                "history_tokens": int(payload["history_tokens"]),
                "eval_tokens": int(payload["eval_tokens"]),
                "full_ppl": full_ppl,
                "reference_ppl": reference_ppl,
                "candidate_ppl": candidate_ppl,
                "reference_quality_vs_full": full_ppl / reference_ppl,
                "candidate_quality_vs_full": full_ppl / candidate_ppl,
                "candidate_quality_vs_reference": (
                    reference_ppl / candidate_ppl
                ),
                "reference_top1_agreement": float(
                    reference.get("top1_agreement", 0.0)
                ),
                "candidate_top1_agreement": float(
                    candidate.get("top1_agreement", 0.0)
                ),
                "reference_kl": float(
                    reference.get("kl_full_to_sparse_mean", math.nan)
                ),
                "candidate_kl": float(
                    candidate.get("kl_full_to_sparse_mean", math.nan)
                ),
                "reference_tokens_per_head": float(
                    reference["actual_attention_tokens_mean"]
                ),
                "candidate_tokens_per_head": float(
                    candidate["actual_attention_tokens_mean"]
                ),
                "reference_auxiliary_ratio": float(
                    reference.get("packed_total_auxiliary_ratio_of_full_kv", 0.0)
                ),
                "candidate_auxiliary_ratio": float(
                    candidate.get("packed_total_auxiliary_ratio_of_full_kv", 0.0)
                ),
                "candidate_overflow_rate": float(
                    candidate.get("candidate_overflow_rate_mean", 0.0)
                ),
                "quality_reference_runtime_seconds_per_step": float(
                    reference.get("sparse_seconds_per_step", math.nan)
                ),
                "quality_candidate_runtime_seconds_per_step": float(
                    candidate.get("sparse_seconds_per_step", math.nan)
                ),
            }
        )
    return cases, incomplete


def summarize(cases: list[dict[str, Any]], incomplete: list[str]) -> dict[str, Any]:
    quality_passes = [
        row["candidate_quality_vs_full"] >= 0.995
        and row["candidate_quality_vs_reference"] >= 0.995
        and row["candidate_top1_agreement"] >= 0.99
        and row["candidate_overflow_rate"] == 0.0
        for row in cases
    ]
    return {
        "schema": "qksieve_blockcondres_model_ppl_summary_v1",
        "reference": REFERENCE,
        "candidate": CANDIDATE,
        "complete": bool(cases) and not incomplete,
        "quality_gate": {
            "candidate_quality_vs_full_minimum": 0.995,
            "candidate_quality_vs_reference_minimum": 0.995,
            "top1_agreement_minimum": 0.99,
            "candidate_overflow_rate_maximum": 0.0,
            "passed_cases": sum(quality_passes),
            "total_cases": len(cases),
            "all_passed": bool(cases) and all(quality_passes),
        },
        "macro": {
            "reference_quality_vs_full": (
                sum(row["reference_quality_vs_full"] for row in cases)
                / len(cases)
                if cases
                else math.nan
            ),
            "candidate_quality_vs_full": (
                sum(row["candidate_quality_vs_full"] for row in cases)
                / len(cases)
                if cases
                else math.nan
            ),
            "candidate_quality_vs_reference": (
                sum(row["candidate_quality_vs_reference"] for row in cases)
                / len(cases)
                if cases
                else math.nan
            ),
        },
        "cases": sorted(cases, key=lambda row: row["history_tokens"]),
        "incomplete": incomplete,
        "claim_boundary": (
            "This is a same-stream model-level PPL gate using the materialized "
            "full proxy top-k quality path. Runtime includes Python reference "
            "operations and is not a CUDA speed result."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Block 条件残差模型级 PPL 验证",
        "",
        "同一自然代码流、共享 prefill。参照为 rank-16/top-1280，候选为 "
        "rank-8/top-1120 + block-256 d8 INT8 条件残差。PPL 越低越好；质量"
        "保持率定义为参照 PPL 除以待测 PPL。",
        "",
        "| 长度 | Full PPL | rank16 PPL | 新方法 PPL | 新方法/Full | 新方法/rank16 | Top-1 | token/head |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["cases"]:
        lines.append(
            f"| {row['history_tokens'] // 1024}K | {row['full_ppl']:.6f} | "
            f"{row['reference_ppl']:.6f} | {row['candidate_ppl']:.6f} | "
            f"{100.0 * row['candidate_quality_vs_full']:.3f}% | "
            f"{100.0 * row['candidate_quality_vs_reference']:.3f}% | "
            f"{100.0 * row['candidate_top1_agreement']:.2f}% | "
            f"{row['candidate_tokens_per_head']:.1f} |"
        )
    gate = payload["quality_gate"]
    lines.extend(
        [
            "",
            f"质量门通过：**{gate['passed_cases']}/{gate['total_cases']}**；"
            f"全部通过：**{gate['all_passed']}**。",
            "",
            "注意：该路径物化完整 proxy score，并使用 Python/PyTorch block "
            "统计，因此这里只能支持质量结论，不能用于报告速度。",
        ]
    )
    if payload["incomplete"]:
        lines.extend(["", f"未完成：`{payload['incomplete']}`"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cases, incomplete = load_cases(args.run_root)
    payload = summarize(cases, incomplete)
    output_json = args.output_json or args.run_root / "paired_summary.json"
    output_markdown = (
        args.output_markdown or args.run_root / "paired_summary_zh.md"
    )
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
