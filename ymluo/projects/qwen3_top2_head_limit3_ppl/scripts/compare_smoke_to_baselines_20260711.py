#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def baseline_map(directory: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(directory / "task_results.csv")
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        task = row.get("task", "")
        sample_id = row.get("sample_id", "")
        if task and sample_id:
            out[(task, sample_id)] = row
    return out


def label_for_dir(path: Path) -> str:
    name = path.name
    if name.startswith("riskkv_v19_"):
        name = name[len("riskkv_v19_") :]
    suffixes = [
        "_20260711_v323_safe_certificate_smoke_m20_bDyn_pDyn",
        "_20260711_v322_sparse_nofull_qa_smoke_m20_bDyn_pDyn",
        "_20260711_b16_compressed_mid_smoke_m20_bDyn_pDyn",
        "_20260711_b16_manyblocks_xl_smoke_m20_bDyn_pDyn",
        "_20260711_bm25_bridge_smoke_m20_bDyn_pDyn",
        "_20260711_b16_windowvote_sweep_m100_bDyn_pDyn",
        "_20260711_b16_microspan_sweep_m100_bDyn_pDyn",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def summarize_variant(
    label: str,
    rows: list[dict[str, str]],
    full_by_key: dict[tuple[str, str], dict[str, str]],
    v300_by_key: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("task", "")].append(row)
    grouped["ALL"] = rows

    out: list[dict[str, Any]] = []
    for task in sorted(grouped, key=lambda item: (item != "ALL", item)):
        subset = grouped[task]
        if not subset:
            continue
        full_rows: list[dict[str, str]] = []
        v300_rows: list[dict[str, str]] = []
        for row in subset:
            key = (row.get("task", ""), row.get("sample_id", ""))
            if key in full_by_key:
                full_rows.append(full_by_key[key])
            if key in v300_by_key:
                v300_rows.append(v300_by_key[key])
        score = mean([fnum(row, "score") for row in subset])
        kv = mean([fnum(row, "keep_fraction") for row in subset])
        online = mean([fnum(row, "online_seconds") for row in subset])
        total = mean([fnum(row, "total_seconds") for row in subset])
        full_score = mean([fnum(row, "score") for row in full_rows])
        v300_score = mean([fnum(row, "score") for row in v300_rows])
        full_online = mean([fnum(row, "online_seconds") for row in full_rows])
        v300_online = mean([fnum(row, "online_seconds") for row in v300_rows])
        out.append(
            {
                "method": label,
                "label": label,
                "task": task,
                "samples": len(subset),
                "score": score,
                "full_score_same_samples": full_score,
                "v300_score_same_samples": v300_score,
                "score_vs_full": score / full_score if full_score > 0 else "",
                "score_vs_v300": score / v300_score if v300_score > 0 else "",
                "kv_keep": kv,
                "online_seconds": online,
                "total_seconds": total,
                "full_online_speed": full_online / online if online > 0 and full_online > 0 else "",
                "v300_online_speed": v300_online / online if online > 0 and v300_online > 0 else "",
                "matched_full": len(full_rows),
                "matched_v300": len(v300_rows),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "label",
        "task",
        "samples",
        "score",
        "full_score_same_samples",
        "v300_score_same_samples",
        "score_vs_full",
        "score_vs_v300",
        "kv_keep",
        "online_seconds",
        "total_seconds",
        "full_online_speed",
        "v300_online_speed",
        "matched_full",
        "matched_v300",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dir", required=True)
    parser.add_argument("--v300_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("variant_dirs", nargs="+")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_by_key = baseline_map(Path(args.full_dir))
    v300_by_key = baseline_map(Path(args.v300_dir))

    rows: list[dict[str, Any]] = []
    for item in args.variant_dirs:
        directory = Path(item)
        label = label_for_dir(directory)
        variant_rows = read_csv(directory / "task_results.csv")
        if not variant_rows:
            continue
        rows.extend(summarize_variant(label, variant_rows, full_by_key, v300_by_key))

    rows.sort(key=lambda row: (row["label"], row["task"] != "ALL", row["task"]))
    summary_rows = [row for row in rows if row.get("task") == "ALL"]
    write_csv(output_dir / "summary_table.csv", summary_rows)
    write_csv(output_dir / "detail_table.csv", rows)
    (output_dir / "summary_table.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "detail_table.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_rows": len(summary_rows), "detail_rows": len(rows), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
