#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FOCUS_EXPERIMENTS = [
    "qwen8b_longbench_m4_conformal_auto",
    "qwen8b_longbench_m8_conformal_auto",
    "qwen8b_mixed13_m1_conformal_auto",
    "qwen8b_mixed13_m2_conformal_auto",
    "qwen8b_ruler4k_m3_conformal_auto",
    "qwen8b_ruler8k_m3_conformal_auto",
    "qwen8b_ruler4k_m5_conformal_auto",
    "qwen8b_ruler8k_m5_conformal_auto",
    "qwen8b_ruler4k_m5_conformal_floor2",
    "qwen8b_ruler8k_m5_conformal_floor2",
    "qwen8b_ruler16k_m2_conformal_auto_sharded",
    "qwen8b_ruler16k_m3_conformal_auto_sharded",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["experiment"], row["method"]): row for row in rows}


def report_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    idx = by_key(rows)
    out: list[dict[str, object]] = []
    for experiment in FOCUS_EXPERIMENTS:
        full = idx.get((experiment, "full_kv_cache"))
        risk = idx.get((experiment, "variable_budget_kv_planner"))
        if full is None or risk is None:
            continue
        full_query = fnum(full, "avg_query_ms")
        full_decode = fnum(full, "avg_decode_ms")
        risk_query = fnum(risk, "avg_query_ms")
        risk_decode = fnum(risk, "avg_decode_ms")
        planner = fnum(risk, "avg_planner_ms")
        repack = fnum(risk, "avg_repack_ms")
        query_saved = full_query - risk_query
        decode_saved = full_decode - risk_decode
        fixed_overhead = planner + repack
        net_component_gain = query_saved + decode_saved - fixed_overhead
        out.append(
            {
                "experiment": experiment,
                "samples": int(float(risk.get("samples", 0) or 0)),
                "score_pct": fnum(risk, "score_pct"),
                "kv_ratio_pct": fnum(risk, "kv_ratio_pct"),
                "online_speedup": fnum(risk, "online_speedup_sum"),
                "e2e_speedup": fnum(risk, "e2e_speedup_sum"),
                "full_query_ms": full_query,
                "risk_query_ms": risk_query,
                "query_saved_ms": query_saved,
                "full_decode_ms": full_decode,
                "risk_decode_ms": risk_decode,
                "decode_saved_ms": decode_saved,
                "planner_ms": planner,
                "repack_ms": repack,
                "fixed_overhead_ms": fixed_overhead,
                "net_component_gain_ms": net_component_gain,
            }
        )
    return out


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# RiskKV Overhead Report",
        "",
        "| Experiment | N | Score | KV | Online | Query saved | Decode saved | Planner | Repack | Net component gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {samples} | {score_pct:.2f}% | {kv_ratio_pct:.2f}% | {online_speedup:.3f}x | "
            "{query_saved_ms:.2f} ms | {decode_saved_ms:.2f} ms | {planner_ms:.2f} ms | {repack_ms:.2f} ms | "
            "{net_component_gain_ms:.2f} ms |".format(**row)
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- Query savings grow with context length because compact KV reduces attention over the active cache.",
        "- Planner and repack are fixed online overheads; at 4k they can cancel most query/decode savings.",
        "- Long-context settings cross the break-even point because query/decode savings dominate fixed overhead.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("outputs/runtime_scaling_summary_20260707/runtime_scaling_summary.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/runtime_scaling_summary_20260707/icml_overhead"),
    )
    args = parser.parse_args()
    rows = report_rows(load_rows(args.summary_csv))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "experiment",
        "samples",
        "score_pct",
        "kv_ratio_pct",
        "online_speedup",
        "e2e_speedup",
        "full_query_ms",
        "risk_query_ms",
        "query_saved_ms",
        "full_decode_ms",
        "risk_decode_ms",
        "decode_saved_ms",
        "planner_ms",
        "repack_ms",
        "fixed_overhead_ms",
        "net_component_gain_ms",
    ]
    with (args.output_dir / "icml_overhead_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.output_dir / "icml_overhead_report.md", rows)
    (args.output_dir / "icml_overhead_report.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
