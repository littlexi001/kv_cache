#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with (path / "task_results.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row.get("task", ""), row.get("sample_id", ""))] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    base = read_rows(Path(args.base))
    candidate = read_rows(Path(args.candidate))
    deltas = []
    for key, row in candidate.items():
        if key not in base:
            continue
        delta = fnum(row.get("score")) - fnum(base[key].get("score"))
        deltas.append((delta, key, row, base[key]))

    losses = [item for item in deltas if item[0] < -1e-9]
    wins = [item for item in deltas if item[0] > 1e-9]
    print(
        f"candidate={Path(args.candidate).name} base={Path(args.base).name} "
        f"matched={len(deltas)} wins={len(wins)} losses={len(losses)} net_delta={sum(item[0] for item in deltas):.6f}"
    )

    for title, subset, reverse in (("LOSSES", losses, False), ("WINS", wins, True)):
        print(title)
        for delta, (task, sample_id), row, base_row in sorted(subset, key=lambda item: item[0], reverse=reverse)[
            : args.limit
        ]:
            print(
                f"{task},{sample_id[:12]},delta={delta:.6f},"
                f"cand={fnum(row.get('score')):.6f},base={fnum(base_row.get('score')):.6f},"
                f"keep={row.get('kept_prefix_tokens')}/{row.get('raw_prefix_tokens')},"
                f"cov={row.get('ours_query_coverage_recall','')},"
                f"retry={row.get('ours_retry_fallback_active','')},"
                f"retry_full={row.get('ours_retry_full_fallback_active','')},"
                f"out_fb={row.get('ours_output_fallback_active','')},"
                f"pred={row.get('prediction','')[:160]}"
            )


if __name__ == "__main__":
    main()
