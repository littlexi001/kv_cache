#!/usr/bin/env python
"""Summarize repeated QKSieve cold/warm speed measurements by context length."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


OUTPUT_TOKENS = (1, 4, 8, 16, 32, 64, 128, 256, 512)
OURS_VARIANT = (
    "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280"
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else math.inf


def find_row(payload: dict[str, Any], method: str) -> dict[str, Any]:
    matches = [row for row in payload["rows"] if row["method"] == method]
    if len(matches) != 1:
        raise ValueError(f"expected one {method!r} row, found {len(matches)}")
    return matches[0]


def find_ours_row(payload: dict[str, Any]) -> dict[str, Any]:
    variants = payload.get("requested_variants", [])
    if variants != [OURS_VARIANT]:
        raise ValueError(f"unexpected QKSieve variants: {variants!r}")
    matches = [
        row for row in payload["rows"] if row["method"] != "full_attention"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one sparse row, found {len(matches)}")
    return matches[0]


def load_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    full = find_row(payload, "full_attention")
    ours = find_ours_row(payload)
    full_step = float(full["steady_sparse_seconds_per_step"])
    sparse_step = float(ours["steady_sparse_seconds_per_step"])
    warm_fixed = float(ours["fixed_sparse_overhead_seconds"])
    resident_build = float(
        payload.get("resident_value_sketch_precompute", {}).get(
            "total_seconds", 0.0
        )
    )
    resident_key_build = float(
        payload.get("resident_key_factor_precompute", {}).get(
            "total_seconds", 0.0
        )
    )
    cold_fixed = warm_fixed + resident_build + resident_key_build
    prefill = float(payload.get("shared_prefill_seconds", 0.0))
    saving = full_step - sparse_step
    row: dict[str, Any] = {
        "summary_path": str(path),
        "context_tokens": int(payload["history_tokens"]),
        "eval_tokens": int(payload["eval_tokens"]),
        "full_step_ms": full_step * 1000.0,
        "sparse_step_ms": sparse_step * 1000.0,
        "steady_speedup": safe_ratio(full_step, sparse_step),
        "warm_fixed_s": warm_fixed,
        "resident_build_s": resident_build,
        "resident_key_build_s": resident_key_build,
        "cold_fixed_s": cold_fixed,
        "prefill_s": prefill,
        "warm_break_even_tokens": safe_ratio(warm_fixed, saving),
        "cold_break_even_tokens": safe_ratio(cold_fixed, saving),
        "quality_retention": math.exp(
            float(full["nll"]) - float(ours["nll"])
        ),
        "auxiliary_ratio": float(
            ours.get("packed_total_auxiliary_ratio_of_full_kv", 0.0)
        ),
    }
    for generated in OUTPUT_TOKENS:
        dense_decode = generated * full_step
        sparse_decode = generated * sparse_step
        row[f"warm_g{generated}"] = safe_ratio(
            dense_decode, warm_fixed + sparse_decode
        )
        row[f"cold_g{generated}"] = safe_ratio(
            dense_decode, cold_fixed + sparse_decode
        )
        row[f"raw_g{generated}"] = safe_ratio(
            prefill + dense_decode,
            prefill + cold_fixed + sparse_decode,
        )
    return row


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["context_tokens"])].append(row)
    output = []
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key not in {"context_tokens"}
    ]
    for context_tokens, group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "context_tokens": context_tokens,
            "repeats": len(group),
        }
        for key in numeric_keys:
            values = [float(item[key]) for item in group]
            summary[f"{key}_median"] = statistics.median(values)
            summary[f"{key}_p05"] = percentile(values, 0.05)
            summary[f"{key}_p95"] = percentile(values, 0.95)
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# QKSieve 长度与输出长度速度曲面",
        "",
        "所有结果使用相同的候选、bit allocation、精确稀疏 Attention 和 ValueSketch 补偿。",
        "Warm 包含请求级 QK/Key 索引但复用缓存常驻 ValueSketch；Cold 还包含 ValueSketch 构建；Raw 再计入完整 prefill。",
        "",
        "| 上下文 | 重复 | Full ms/token | QKSieve ms/token | 稳态 | Warm fixed | Cold fixed | Warm回本 | Cold回本 | 质量保持 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {context_tokens:,} | {repeats} | {full_step_ms_median:.2f} | "
            "{sparse_step_ms_median:.2f} | {steady_speedup_median:.3f}x | "
            "{warm_fixed_s_median:.3f}s | {cold_fixed_s_median:.3f}s | "
            "{warm_break_even_tokens_median:.1f} | "
            "{cold_break_even_tokens_median:.1f} | "
            "{quality_retention_median:.3%} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Cached-prefix Warm 总速度",
            "",
            "| 上下文 | G=8 | G=16 | G=32 | G=64 | G=128 | G=256 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {context_tokens:,} | {warm_g8_median:.3f}x | "
            "{warm_g16_median:.3f}x | {warm_g32_median:.3f}x | "
            "{warm_g64_median:.3f}x | {warm_g128_median:.3f}x | "
            "{warm_g256_median:.3f}x |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Cold 总速度",
            "",
            "| 上下文 | G=8 | G=16 | G=32 | G=64 | G=128 | G=256 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {context_tokens:,} | {cold_g8_median:.3f}x | "
            "{cold_g16_median:.3f}x | {cold_g32_median:.3f}x | "
            "{cold_g64_median:.3f}x | {cold_g128_median:.3f}x | "
            "{cold_g256_median:.3f}x |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    candidate_paths = sorted(
        args.run_root.glob("n*/r*/legacy/quality/summary.json")
    )
    paths = [
        path
        for path in candidate_paths
        if (path.parents[2] / "ALL_COMPLETE").is_file()
    ]
    if not paths:
        raise FileNotFoundError(
            f"no completed summaries under {args.run_root}; "
            f"found {len(candidate_paths)} partial summary files"
        )
    rows = [load_run(path) for path in paths]
    summaries = aggregate(rows)
    (args.run_root / "length_speed_surface.json").write_text(
        json.dumps(
            {"runs": rows, "summaries": summaries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(args.run_root / "length_speed_surface.csv", summaries)
    write_markdown(args.run_root / "length_speed_surface.md", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
