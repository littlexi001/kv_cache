#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(fnum(row, key) for row in rows) / max(1, len(rows))


def accuracy(rows: list[dict[str, str]]) -> float:
    return mean(rows, "score")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def method_summary(
    label: str,
    rows: list[dict[str, str]],
    full_rows: list[dict[str, str]],
) -> dict[str, Any]:
    full_score = accuracy(full_rows)
    method_score = accuracy(rows)
    full_online = sum(fnum(row, "online_seconds") for row in full_rows)
    full_total = sum(fnum(row, "total_seconds") for row in full_rows)
    method_online = sum(fnum(row, "online_seconds") for row in rows)
    method_total = sum(fnum(row, "total_seconds") for row in rows)
    return {
        "method": label,
        "samples": len(rows),
        "accuracy": method_score,
        "accuracy_over_full": method_score / full_score if full_score > 0 else None,
        "mean_kv_ratio": mean(rows, "keep_fraction"),
        "online_speed_vs_full": full_online / max(1e-12, method_online),
        "total_speed_vs_full": full_total / max(1e-12, method_total),
        "mean_prefill_seconds": mean(rows, "prefill_seconds"),
        "mean_gather_seconds": mean(rows, "kv_gather_seconds"),
        "mean_query_seconds": mean(rows, "query_seconds"),
        "mean_decode_seconds": mean(rows, "decode_seconds"),
        "valid_answer_format_rate": sum(bool(row.get("longbench_v2_pred", "")) for row in rows)
        / max(1, len(rows)),
        "direct_rate": sum(int(fnum(row, "ours_direct_structured_answer_used")) for row in rows)
        / max(1, len(rows)),
        "operator_counts": dict(Counter(row.get("ours_operator_mode", "") for row in rows)),
        "route_errors": sum(
            row.get("ours_operator_fallback_reason", "").startswith("operator_router_error") for row in rows
        ),
    }


def breakdown_rows(
    methods: dict[str, list[dict[str, str]]],
    key: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for method, rows in methods.items():
        for row in rows:
            grouped[(method, row.get(key, "") or "unknown")].append(row)
    output: list[dict[str, Any]] = []
    for (method, group), rows in sorted(grouped.items()):
        output.append(
            {
                "breakdown": key,
                "group": group,
                "method": method,
                "samples": len(rows),
                "accuracy": accuracy(rows),
                "mean_kv_ratio": mean(rows, "keep_fraction"),
                "direct_rate": sum(int(fnum(row, "ours_direct_structured_answer_used")) for row in rows)
                / max(1, len(rows)),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--v466", required=True)
    parser.add_argument("--direct-off", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_rows = {
        "full": read_csv(Path(args.full)),
        "v466": read_csv(Path(args.v466)),
        "v466_direct_off": read_csv(Path(args.direct_off)),
    }
    indices = {
        method: {row["sample_id"]: row for row in rows}
        for method, rows in source_rows.items()
    }
    common_ids = set.intersection(*(set(index) for index in indices.values()))
    if not common_ids:
        raise RuntimeError("No sample IDs are shared by Full, v466, and direct-off results.")
    ordered_ids = [row["sample_id"] for row in source_rows["full"] if row["sample_id"] in common_ids]
    methods = {
        method: [index[sample_id] for sample_id in ordered_ids]
        for method, index in indices.items()
    }
    if any(len(rows) != len(ordered_ids) for rows in methods.values()):
        raise RuntimeError("Matched LongBench v2 rows are inconsistent across methods.")

    full_rows = methods["full"]
    overall = [method_summary(method, rows, full_rows) for method, rows in methods.items()]
    breakdowns: list[dict[str, Any]] = []
    for key in ("difficulty", "length_category", "domain"):
        breakdowns.extend(breakdown_rows(methods, key))

    paired: list[dict[str, Any]] = []
    for sample_id in ordered_ids:
        full = indices["full"][sample_id]
        v466 = indices["v466"][sample_id]
        direct_off = indices["v466_direct_off"][sample_id]
        paired.append(
            {
                "sample_id": sample_id,
                "domain": full.get("domain", ""),
                "sub_domain": full.get("sub_domain", ""),
                "difficulty": full.get("difficulty", ""),
                "length_category": full.get("length_category", ""),
                "answer": full.get("answers", ""),
                "full_pred": full.get("longbench_v2_pred", ""),
                "full_score": fnum(full, "score"),
                "v466_pred": v466.get("longbench_v2_pred", ""),
                "v466_score": fnum(v466, "score"),
                "v466_operator": v466.get("ours_operator_mode", ""),
                "v466_direct": int(fnum(v466, "ours_direct_structured_answer_used")),
                "direct_off_pred": direct_off.get("longbench_v2_pred", ""),
                "direct_off_score": fnum(direct_off, "score"),
                "direct_off_operator": direct_off.get("ours_operator_mode", ""),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "overall.csv", overall)
    write_csv(output_dir / "official_breakdowns.csv", breakdowns)
    write_csv(output_dir / "paired_predictions.csv", paired)
    payload = {
        "matched_samples": len(ordered_ids),
        "source_samples": {method: len(rows) for method, rows in source_rows.items()},
        "overall": overall,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
