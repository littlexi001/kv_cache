from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the rotated periodic-reuse oracle benchmark."
    )
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def load_rows(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_root.glob("length*/rotation*/*/sample_results.csv")):
        relative = path.relative_to(run_root).parts
        length = int(relative[0].removeprefix("length"))
        rotation = int(relative[1].removeprefix("rotation"))
        method = relative[2]
        with path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        if len(records) != 1:
            raise RuntimeError(f"expected one sample in {path}, got {len(records)}")
        record = records[0]
        rows.append(
            {
                "length": length,
                "rotation": rotation,
                "method": method,
                "score": float(record["score"]),
                "prediction": record["prediction"],
                "prompt_tokens": int(record["prompt_tokens"]),
                "generated_tokens": int(record["generated_tokens"]),
                "query_seconds": float(record["query_seconds"]),
                "decode_seconds": float(record["decode_seconds"]),
                "online_seconds": float(record["online_seconds"]),
                "total_seconds": float(record["total_seconds"]),
            }
        )
    if not rows:
        raise RuntimeError(f"no sample results found below {run_root}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["length"], row["method"])].append(row)

    lengths = []
    for length in sorted({row["length"] for row in rows}):
        base = grouped[(length, "base")]
        base_online = mean(row["online_seconds"] for row in base)
        base_score = mean(row["score"] for row in base)
        base_prediction = base[0]["prediction"]
        methods = []
        for method in ("base", "reuse2", "reuse4", "reuse8"):
            subset = grouped[(length, method)]
            online = mean(row["online_seconds"] for row in subset)
            score = mean(row["score"] for row in subset)
            methods.append(
                {
                    "method": method,
                    "rotations": len(subset),
                    "score": score,
                    "quality_retention": score / base_score,
                    "exact_base_prediction": all(
                        row["prediction"] == base_prediction for row in subset
                    ),
                    "query_seconds": mean(
                        row["query_seconds"] for row in subset
                    ),
                    "decode_seconds": mean(
                        row["decode_seconds"] for row in subset
                    ),
                    "online_seconds": online,
                    "online_speedup_vs_base": base_online / online,
                    "total_seconds": mean(
                        row["total_seconds"] for row in subset
                    ),
                }
            )
        lengths.append(
            {
                "prompt_tokens": length,
                "generated_tokens": base[0]["generated_tokens"],
                "methods": methods,
            }
        )
    return {
        "samples": len(rows),
        "note": (
            "Periodic reuse rates are approximately 50%, 75%, and 87.5% for "
            "reuse2, reuse4, and reuse8 after the first refresh."
        ),
        "lengths": lengths,
    }


def main() -> None:
    args = parse_args()
    output = summarize(load_rows(args.run_root))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
