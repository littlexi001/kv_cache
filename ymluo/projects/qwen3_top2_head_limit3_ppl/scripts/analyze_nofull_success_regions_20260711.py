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


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("benchmark", ""), row.get("task", ""), row.get("sample_id", ""))


def fnum(value: Any) -> float:
    try:
        return float(str(value or 0.0))
    except Exception:
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


FEATURES = [
    "raw_prefix_tokens",
    "keep_fraction",
    "ours_score_max",
    "ours_score_mean",
    "ours_score_gap2",
    "ours_score_gap3",
    "ours_score_entropy",
    "ours_score_risk_linear_value",
    "ours_score_risk_raw_prefix_at_most",
    "ours_score_risk_triggered",
    "ours_score_risk_raw_triggered",
    "ours_score_safe_certified",
]


def summarize_group(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"group": label, "samples": len(rows)}
    out["nofull_score"] = mean([fnum(row.get("nofull_score")) for row in rows])
    out["v300_score"] = mean([fnum(row.get("v300_score")) for row in rows])
    out["full_score"] = mean([fnum(row.get("full_score")) for row in rows])
    out["score_vs_v300"] = out["nofull_score"] / out["v300_score"] if out["v300_score"] > 0 else ""
    out["score_vs_full"] = out["nofull_score"] / out["full_score"] if out["full_score"] > 0 else ""
    out["nofull_kv"] = mean([fnum(row.get("nofull_kv")) for row in rows])
    out["v300_kv"] = mean([fnum(row.get("v300_kv")) for row in rows])
    for feature in FEATURES:
        values = [fnum(row.get(feature)) for row in rows if str(row.get(feature, "")) != ""]
        if values:
            out[f"mean_{feature}"] = mean(values)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nofull_dirs", nargs="+", required=True)
    parser.add_argument("--v300_dir", required=True)
    parser.add_argument("--full_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    v300 = {key(row): row for row in read_csv(Path(args.v300_dir) / "task_results.csv")}
    full = {key(row): row for row in read_csv(Path(args.full_dir) / "task_results.csv")}

    joined: list[dict[str, Any]] = []
    for directory_text in args.nofull_dirs:
        directory = Path(directory_text)
        for row in read_csv(directory / "task_results.csv"):
            k = key(row)
            v300_row = v300.get(k)
            full_row = full.get(k)
            if v300_row is None or full_row is None:
                continue
            out: dict[str, Any] = {
                "benchmark": k[0],
                "task": k[1],
                "sample_id": k[2],
                "nofull_score": fnum(row.get("score")),
                "v300_score": fnum(v300_row.get("score")),
                "full_score": fnum(full_row.get("score")),
                "nofull_kv": fnum(row.get("keep_fraction")),
                "v300_kv": fnum(v300_row.get("keep_fraction")),
                "full_online": fnum(full_row.get("online_seconds")),
                "nofull_online": fnum(row.get("online_seconds")),
                "v300_online": fnum(v300_row.get("online_seconds")),
                "nofull_keeps_v300": int(fnum(row.get("score")) + 1e-9 >= fnum(v300_row.get("score"))),
                "nofull_keeps_95_full": int(fnum(row.get("score")) + 1e-9 >= 0.95 * fnum(full_row.get("score"))),
            }
            for feature in FEATURES:
                out[feature] = v300_row.get(feature, "")
            joined.append(out)

    write_csv(output_dir / "joined_nofull_regions.csv", joined)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[f"task={row['task']}"].append(row)
        grouped[f"task={row['task']},keeps_v300={row['nofull_keeps_v300']}"].append(row)
        grouped[f"task={row['task']},keeps_95full={row['nofull_keeps_95_full']}"].append(row)
    summary = [summarize_group(label, rows) for label, rows in sorted(grouped.items())]
    write_csv(output_dir / "nofull_region_summary.csv", summary)
    (output_dir / "nofull_region_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"joined": len(joined), "summary_rows": len(summary), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
