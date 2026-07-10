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


def read_all_summary(path: Path) -> dict[str, Any] | None:
    summary = path / "summary.csv"
    if not summary.exists():
        return None
    with summary.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("task") == "ALL":
                return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        help="Row spec label=path. Can be repeated.",
    )
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    rows = []
    full_score = None
    for spec in args.row:
        if "=" not in spec:
            raise ValueError(f"Expected label=path, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        row = read_all_summary(Path(raw_path))
        if row is None:
            rows.append((label, None))
            continue
        score = fnum(row.get("score"))
        if full_score is None and ("full" in label.lower() or "full" in raw_path.lower()):
            full_score = score
        rows.append((label, row))

    if args.latex:
        print(r"Policy & Score & Full-score ratio & KV ratio & Online \\")
        print(r"\midrule")
    else:
        print("policy,score,full_score_ratio,kv_ratio,online_seconds")

    for label, row in rows:
        if row is None:
            if args.latex:
                print(f"{label} & -- & -- & -- & -- \\\\")
            else:
                print(f"{label},MISSING,MISSING,MISSING,MISSING")
            continue
        score = fnum(row.get("score"))
        keep = fnum(row.get("mean_keep_fraction"))
        online = fnum(row.get("mean_online_seconds"))
        ratio = score / full_score if full_score else 0.0
        if args.latex:
            print(f"{label} & {score:.6f} & {ratio * 100.0:.1f}\\% & {keep * 100.0:.2f}\\% & {online:.3f}s \\\\")
        else:
            print(f"{label},{score:.6f},{ratio:.6f},{keep:.6f},{online:.6f}")


if __name__ == "__main__":
    main()
