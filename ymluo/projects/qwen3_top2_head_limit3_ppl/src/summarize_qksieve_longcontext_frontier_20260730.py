#!/usr/bin/env python
"""Summarize paired long-context QKSieve quality/speed frontiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_md", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("rows"), list):
        raise ValueError(f"{path} does not contain result rows")
    return payload


def finite_speedup(numerator: float, denominator: float) -> float | None:
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return numerator / denominator


def summarize_payload(
    payload: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    rows = payload["rows"]
    full = next(
        (row for row in rows if row.get("variant") == "full_attention"),
        None,
    )
    if full is None:
        raise ValueError(f"{source} has no Full-attention row")
    full_steady = float(full["steady_sparse_seconds_per_step"])
    full_online = float(full["sparse_decode_seconds"])
    history = int(payload["history_tokens"])
    output_rows = []
    for row in rows:
        if row is full:
            continue
        actual_tokens = float(row["actual_attention_tokens_mean"])
        steady = float(row["steady_sparse_seconds_per_step"])
        online = float(row["sparse_decode_seconds"])
        output_rows.append(
            {
                "variant": row["variant"],
                "max_exact_tokens_per_head": int(
                    row.get("max_exact_tokens_per_head", 1280)
                ),
                "actual_exact_tokens_per_head": actual_tokens,
                "actual_exact_fraction": actual_tokens / history,
                "quantile_sample_count": int(
                    row.get("packed_quantile_sample_count", 0)
                ),
                "ppl": float(row["ppl"]),
                "quality_retention": float(row["quality_retention"]),
                "top1_agreement": float(row["top1_agreement"]),
                "kl_full_to_sparse": float(
                    row["kl_full_to_sparse_mean"]
                ),
                "steady_ms_per_token": 1000.0 * steady,
                "steady_speedup": finite_speedup(full_steady, steady),
                "online_ms_per_token": (
                    1000.0 * online / max(1, int(row["tokens"]) - 1)
                ),
                "online_speedup": finite_speedup(full_online, online),
                "fixed_overhead_seconds": float(
                    row["fixed_sparse_overhead_seconds"]
                ),
                "candidate_overflow_rate": float(
                    row["candidate_overflow_rate_mean"]
                ),
            }
        )
    return {
        "source": str(source),
        "history_tokens": history,
        "eval_tokens": int(payload["eval_tokens"]),
        "native_context_tokens": int(payload["native_context_tokens"]),
        "context_extrapolation": bool(payload["context_extrapolation"]),
        "full": {
            "ppl": float(full["ppl"]),
            "nll": float(full["nll"]),
            "steady_ms_per_token": 1000.0 * full_steady,
            "online_ms_per_token": (
                1000.0
                * full_online
                / max(1, int(full["tokens"]) - 1)
            ),
        },
        "rows": output_rows,
    }


def markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# QKSieve long-context quality-speed frontier",
        "",
    ]
    for summary in summaries:
        status = (
            "extrapolation stress"
            if summary["context_extrapolation"]
            else "native context"
        )
        lines.extend(
            [
                (
                    f"## {summary['history_tokens']:,} history tokens "
                    f"({status})"
                ),
                "",
                (
                    f"Full PPL: {summary['full']['ppl']:.6f}; "
                    f"steady: {summary['full']['steady_ms_per_token']:.3f} "
                    "ms/token."
                ),
                "",
                (
                    "| Variant | Exact/head | Active | Quantile samples | "
                    "PPL retention | Top-1 | KL | Steady | Online |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["rows"]:
            steady = row["steady_speedup"]
            online = row["online_speedup"]
            lines.append(
                "| {variant} | {tokens:.1f} | {active:.3%} | "
                "{samples:,} | {retention:.3%} | {top1:.3%} | "
                "{kl:.5f} | {steady} | {online} |".format(
                    variant=row["variant"],
                    tokens=row["actual_exact_tokens_per_head"],
                    active=row["actual_exact_fraction"],
                    samples=row["quantile_sample_count"],
                    retention=row["quality_retention"],
                    top1=row["top1_agreement"],
                    kl=row["kl_full_to_sparse"],
                    steady=(
                        f"{steady:.3f}x" if steady is not None else "n/a"
                    ),
                    online=(
                        f"{online:.3f}x" if online is not None else "n/a"
                    ),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = [args.input]
    if args.baseline is not None:
        paths.insert(0, args.baseline)
    summaries = [summarize_payload(load_payload(path), path) for path in paths]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"runs": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        markdown_table(summaries) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
