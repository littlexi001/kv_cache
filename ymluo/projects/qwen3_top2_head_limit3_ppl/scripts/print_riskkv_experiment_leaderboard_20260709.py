#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")


def fnum(value: str, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def main() -> None:
    rows = []
    for summary in sorted((ROOT / "outputs").glob("riskkv_*20260709*/summary.csv")):
        with summary.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("benchmark") != "longbench" or row.get("task") != "ALL":
                    continue
                rows.append(
                    {
                        "experiment": summary.parent.name,
                        "samples": row.get("samples", ""),
                        "score": fnum(row.get("score", "")),
                        "keep": fnum(row.get("mean_keep_fraction", "")),
                        "online": fnum(row.get("mean_online_seconds", "")),
                        "total": fnum(row.get("mean_total_seconds", "")),
                    }
                )
    rows.sort(key=lambda row: (row["score"], -row["keep"], -row["online"]), reverse=True)
    print("experiment,samples,score,keep_fraction,online_seconds,total_seconds")
    for row in rows:
        print(
            f"{row['experiment']},{row['samples']},{row['score']:.6f},"
            f"{row['keep']:.6f},{row['online']:.6f},{row['total']:.6f}"
        )


if __name__ == "__main__":
    main()

