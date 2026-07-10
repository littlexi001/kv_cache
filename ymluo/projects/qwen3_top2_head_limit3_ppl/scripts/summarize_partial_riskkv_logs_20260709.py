#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


RESULT_RE = re.compile(
    r"(?P<method>ours_page_gather|full_kv|streamingllm_sink_recent|h2o_observe|snapkv_observe): "
    r"score=(?P<score>[0-9.]+) kept=(?P<kept>\d+)/(?P<raw>\d+) online=(?P<online>[0-9.]+)s"
)
PROGRESS_RE = re.compile(r"\[(?P<done>\d+)/(?P<total>\d+)\]\s+(?P<benchmark>[^/]+)/(?P<task>[^/]+)/")


def summarize_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    rows: list[tuple[float, int, int, float]] = []
    for match in RESULT_RE.finditer(text):
        rows.append(
            (
                float(match.group("score")),
                int(match.group("kept")),
                int(match.group("raw")),
                float(match.group("online")),
            )
        )
    progress = None
    for match in PROGRESS_RE.finditer(text):
        progress = (
            int(match.group("done")),
            int(match.group("total")),
            match.group("benchmark"),
            match.group("task"),
        )
    if not rows:
        return {
            "samples": 0,
            "score": 0.0,
            "keep": 0.0,
            "online": 0.0,
            "progress": progress,
        }
    n = len(rows)
    return {
        "samples": n,
        "score": sum(row[0] for row in rows) / n,
        "keep": sum(row[1] / max(1, row[2]) for row in rows) / n,
        "online": sum(row[3] for row in rows) / n,
        "progress": progress,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="NAME=PATH entries or raw log paths.")
    args = parser.parse_args()
    print("name,samples,score,keep,online,progress")
    for item in args.logs:
        if "=" in item:
            name, path_s = item.split("=", 1)
        else:
            path_s = item
            name = Path(path_s).stem
        summary = summarize_log(Path(path_s))
        progress = summary["progress"]
        progress_text = ""
        if progress:
            done, total, benchmark, task = progress
            progress_text = f"{done}/{total}:{benchmark}/{task}"
        print(
            f"{name},{summary['samples']},{summary['score']:.6f},"
            f"{summary['keep']:.6f},{summary['online']:.3f},{progress_text}"
        )


if __name__ == "__main__":
    main()
