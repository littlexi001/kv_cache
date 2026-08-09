#!/usr/bin/env python
"""Aggregate repeated QKSieve/FIER direct-CUDA stage matrices."""

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


def quartiles(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [values[0], values[0]]
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    return [cuts[0], cuts[2]]


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    common = set.intersection(*(set(record) for record in records))
    if any(set(record) != common for record in records):
        raise ValueError("repeat records expose different fields")
    result: dict[str, Any] = {}
    iqr: dict[str, list[float]] = {}
    for field in sorted(common):
        values = [record[field] for record in records]
        if all(isinstance(value, bool) for value in values):
            if len(set(values)) != 1:
                raise ValueError(f"boolean field {field} differs")
            result[field] = values[0]
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            numeric = [float(value) for value in values]
            result[field] = statistics.median(numeric)
            iqr[field] = quartiles(numeric)
        elif all(isinstance(value, dict) for value in values):
            result[field] = aggregate_records(values)
        else:
            canonical = [json.dumps(value, sort_keys=True) for value in values]
            if len(set(canonical)) != 1:
                raise ValueError(f"field {field} differs")
            result[field] = values[0]
    if iqr:
        result["iqr"] = iqr
    return result


def main() -> None:
    args = parse_args()
    paths = [Path(item) for item in sorted(glob.glob(args.input_glob))]
    if len(paths) < 2:
        raise ValueError("at least two stage-matrix files are required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(
        payload.get("schema") != "qksieve_direct_cuda_stage_matrix_v1"
        for payload in payloads
    ):
        raise ValueError("unexpected input schema")
    for field in ("hardware", "contract"):
        values = {json.dumps(payload[field], sort_keys=True) for payload in payloads}
        if len(values) != 1:
            raise ValueError(f"{field} differs across repeats")

    length_maps = [
        {int(row["history_tokens"]): row for row in payload["lengths"]}
        for payload in payloads
    ]
    length_sets = {tuple(sorted(mapping)) for mapping in length_maps}
    if len(length_sets) != 1:
        raise ValueError("context lengths differ across repeats")

    output_lengths: list[dict[str, Any]] = []
    for history in sorted(length_maps[0]):
        rows = [mapping[history] for mapping in length_maps]
        method_maps = [
            {method["method"]: method for method in row["methods"]}
            for row in rows
        ]
        method_sets = {tuple(sorted(mapping)) for mapping in method_maps}
        if len(method_sets) != 1:
            raise ValueError(f"methods differ at history={history}")
        output_lengths.append(
            {
                "history_tokens": history,
                "selected_tokens_per_query_head": rows[0][
                    "selected_tokens_per_query_head"
                ],
                "full_attention": aggregate_records(
                    [row["full_attention"] for row in rows]
                ),
                "methods": [
                    aggregate_records([mapping[name] for mapping in method_maps])
                    for name in sorted(method_maps[0])
                ],
            }
        )

    output = {
        "schema": "qksieve_public_direct_repeat_summary_v1",
        "hardware": payloads[0]["hardware"],
        "contract": payloads[0]["contract"],
        "repeat_count": len(payloads),
        "aggregation": "median with inclusive Q1/Q3",
        "source_files": [str(path) for path in paths],
        "lengths": output_lengths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
