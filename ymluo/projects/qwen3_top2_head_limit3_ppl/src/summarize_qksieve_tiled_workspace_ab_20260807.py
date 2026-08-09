#!/usr/bin/env python
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


MODES = (
    "legacy_allocating",
    "resident_strided_scalar",
    "resident_strided_tiled",
)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty measurement list")
    return float(statistics.median(values))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_qksieve_tiled_workspace_ab.py RUN_ROOT")
    root = Path(sys.argv[1])
    repeats: dict[int, dict[str, dict[str, float]]] = {}
    for summary_path in sorted(root.glob("r*/*/summary.json")):
        repeat = int(summary_path.parents[1].name.removeprefix("r"))
        mode = summary_path.parent.name
        if mode not in MODES:
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        full_rows = [
            row for row in payload["rows"] if row["variant"] == "full_attention"
        ]
        sparse_rows = [
            row for row in payload["rows"] if row["variant"] != "full_attention"
        ]
        if len(full_rows) != 1 or len(sparse_rows) != 1:
            raise RuntimeError(f"invalid row cardinality in {summary_path}")
        full, sparse = full_rows[0], sparse_rows[0]
        repeats.setdefault(repeat, {})[mode] = {
            "full_ppl": float(full["ppl"]),
            "sparse_ppl": float(sparse["ppl"]),
            "quality_retention": float(full["ppl"]) / float(sparse["ppl"]),
            "top1_agreement": float(sparse.get("top1_agreement", 0.0)),
            "full_step_s": float(full["steady_sparse_seconds_per_step"]),
            "sparse_step_s": float(sparse["steady_sparse_seconds_per_step"]),
            "fixed_s": float(sparse["fixed_sparse_overhead_seconds"]),
            "attention_tokens": float(sparse["actual_attention_tokens_mean"]),
        }
    complete = {
        repeat: rows
        for repeat, rows in repeats.items()
        if set(rows) == set(MODES)
    }
    if not complete:
        raise RuntimeError(f"no complete A/B repeats under {root}")

    aggregates: dict[str, dict[str, float]] = {}
    for mode in MODES:
        rows = [complete[repeat][mode] for repeat in sorted(complete)]
        aggregates[mode] = {
            key: median([row[key] for row in rows]) for key in rows[0]
        }
        aggregates[mode]["decode_speedup_vs_full"] = (
            aggregates[mode]["full_step_s"]
            / aggregates[mode]["sparse_step_s"]
        )
    legacy_step = aggregates["legacy_allocating"]["sparse_step_s"]
    legacy_fixed = aggregates["legacy_allocating"]["fixed_s"]
    for mode in MODES:
        aggregates[mode]["step_speedup_vs_legacy"] = (
            legacy_step / aggregates[mode]["sparse_step_s"]
        )
        aggregates[mode]["fixed_speedup_vs_legacy"] = (
            legacy_fixed / aggregates[mode]["fixed_s"]
            if aggregates[mode]["fixed_s"] > 0.0
            else 0.0
        )

    result = {
        "schema": "qksieve_tiled_workspace_ab_v1",
        "complete_repeats": len(complete),
        "modes": aggregates,
        "per_repeat": complete,
        "quality_contract": (
            "identical frozen selector and ValueSketch formula; only workspace "
            "lifetime, strided KV addressing, and exact-QK thread mapping differ"
        ),
    }
    (root / "summary_ab.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# QKSieve tiled/workspace A/B",
        "",
        "| 模式 | PPL保持 | Top-1 | 稳态ms/token | 对Full加速 | 对旧版加速 | 固定开销s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = aggregates[mode]
        lines.append(
            f"| {mode} | {100*row['quality_retention']:.4f}% | "
            f"{100*row['top1_agreement']:.2f}% | "
            f"{1000*row['sparse_step_s']:.3f} | "
            f"{row['decode_speedup_vs_full']:.3f}x | "
            f"{row['step_speedup_vs_legacy']:.3f}x | {row['fixed_s']:.4f} |"
        )
    lines.extend(
        [
            "",
            "质量边界：三种模式共享同一冻结 selector、候选集合、ValueSketch "
            "公式和原始 K/V；只改变 workspace 生命周期、K/V stride 寻址与精确 "
            "QK 的线程映射。",
        ]
    )
    (root / "summary_ab_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
