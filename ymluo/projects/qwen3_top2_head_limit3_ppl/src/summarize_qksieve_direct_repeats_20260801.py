#!/usr/bin/env python
"""Aggregate repeated direct-CUDA QKSieve measurements by median and IQR."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    return cuts[0], cuts[2]


def main() -> None:
    args = parse_args()
    paths = [Path(item) for item in sorted(glob.glob(args.input_glob))]
    if len(paths) < 2:
        raise ValueError("at least two direct-timing files are required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    contracts = {
        json.dumps(payload["timing_contract"], sort_keys=True)
        for payload in payloads
    }
    if len(contracts) != 1:
        raise ValueError("timing contracts differ across repeats")
    metadata_fields = ("device", "dtype", "allocation_profile")
    metadata: dict[str, Any] = {}
    for field in metadata_fields:
        values = {payload.get(field) for payload in payloads}
        if len(values) != 1:
            raise ValueError(f"{field} differs across repeats: {values}")
        metadata[field] = values.pop()

    by_length: dict[int, list[dict[str, Any]]] = {}
    for payload in payloads:
        seen: set[int] = set()
        for row in payload["rows"]:
            history = int(row["history_tokens"])
            if history in seen:
                raise ValueError(f"duplicate length {history} in one repeat")
            seen.add(history)
            by_length.setdefault(history, []).append(row)
    if any(len(rows) != len(payloads) for rows in by_length.values()):
        raise ValueError("repeats do not contain the same context lengths")

    output_rows: list[dict[str, Any]] = []
    for history, rows in sorted(by_length.items()):
        output: dict[str, Any] = {"history_tokens": history}
        iqr: dict[str, list[float]] = {}
        common_fields = set.intersection(*(set(row) for row in rows))
        for field in sorted(common_fields):
            if field == "history_tokens":
                continue
            values = [row[field] for row in rows]
            if all(isinstance(value, bool) for value in values):
                if len(set(values)) != 1:
                    raise ValueError(f"boolean field {field} differs at {history}")
                output[field] = values[0]
            elif all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                numeric = [float(value) for value in values]
                output[field] = statistics.median(numeric)
                q1, q3 = quartiles(numeric)
                iqr[field] = [q1, q3]
            else:
                if len(set(values)) != 1:
                    raise ValueError(f"field {field} differs at {history}")
                output[field] = values[0]
        output["iqr"] = iqr
        output_rows.append(output)

    result = {
        "schema": "qksieve_direct_cuda_repeat_summary_v1",
        **metadata,
        "repeat_count": len(payloads),
        "source_files": [str(path) for path in paths],
        "timing_contract": payloads[0]["timing_contract"],
        "aggregation": "median with inclusive Q1/Q3",
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
