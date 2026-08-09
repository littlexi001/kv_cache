from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sharded PG19 causal echo PPL runs.")
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_json", type=Path)
    return parser.parse_args()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    full_nll = sum(r["nll"] * r["tokens"] for r in by_method["full_kv"]) / sum(
        r["tokens"] for r in by_method["full_kv"]
    )
    summary: dict[str, Any] = {}
    for method, method_rows in sorted(by_method.items()):
        tokens = sum(row["tokens"] for row in method_rows)
        nll = sum(row["nll"] * row["tokens"] for row in method_rows) / tokens
        summary[method] = {
            "samples": len(method_rows),
            "tokens": tokens,
            "nll": nll,
            "ppl": math.exp(nll),
            "ppl_ratio_to_full": math.exp(nll - full_nll),
            "mean_kv_ratio": sum(row["kv_ratio"] for row in method_rows) / len(method_rows),
            "mean_seconds": sum(row["seconds"] for row in method_rows) / len(method_rows),
            "echo_matches": sum(len(row.get("echo_matches", [])) for row in method_rows),
            "cache_rebuilds": sum(
                int(row.get("timing", {}).get("cache_rebuilds", 0)) for row in method_rows
            ),
        }
    if "causal_echo" in summary and "tokenwise_static" in summary:
        summary["causal_echo"]["overhead_vs_tokenwise_static"] = (
            summary["causal_echo"]["mean_seconds"]
            / summary["tokenwise_static"]["mean_seconds"]
            - 1.0
        )
    return summary


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.input_glob))
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"No result rows found under {args.input_glob}")
    output = {
        "files": files,
        "summary": aggregate(rows),
        "per_sample": [
            {
                "book_index": row["book_index"],
                "book_title": row["book_title"],
                "window": row["window"],
                "method": row["method"],
                "ppl": row["ppl"],
                "seconds": row["seconds"],
                "echo_matches": len(row.get("echo_matches", [])),
                "cache_rebuilds": int(row.get("timing", {}).get("cache_rebuilds", 0)),
            }
            for row in rows
        ],
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
