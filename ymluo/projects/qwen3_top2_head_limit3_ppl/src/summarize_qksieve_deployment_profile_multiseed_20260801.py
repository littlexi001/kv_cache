"""Aggregate repeated direct-CUDA QKSieve profile measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "qksieve_deployment_direct_cuda_stages_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--history_tokens", default="32768,131072")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile_range(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def aggregate_documents(
    documents: list[tuple[str, dict[str, Any]]],
    history_tokens: set[int],
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    hardware: set[str] = set()
    for source, document in documents:
        if document.get("schema") != SCHEMA:
            raise ValueError(f"unexpected schema in {source}: {document.get('schema')}")
        hardware_field = document.get("hardware", "unknown")
        if isinstance(hardware_field, dict):
            hardware_name = hardware_field.get("device_name", "unknown")
        else:
            hardware_name = hardware_field
        hardware.add(str(hardware_name))
        for row in document.get("rows", []):
            length = int(row["history_tokens"])
            if length not in history_tokens:
                continue
            copied = dict(row)
            copied["source"] = source
            grouped[(str(row["profile"]), length)].append(copied)

    rows: list[dict[str, Any]] = []
    metric_fields = (
        "attention_complete_direct_ms",
        "attention_speedup_vs_full_preexpanded_sdpa",
        "fused_sampled_retrieval_direct_ms",
        "exact_sparse_attention_direct_ms",
        "query_plus_retrieval_direct_ms",
        "historical_index_build_direct_ms",
        "per_token_index_append_direct_ms",
        "mean_selected_tokens_per_query_head",
    )
    for (profile, length), repeats in sorted(grouped.items()):
        rows.append(
            {
                "profile": profile,
                "history_tokens": length,
                "logical_index_bits_per_token_per_kv_head": int(
                    repeats[0]["logical_index_bits_per_token_per_kv_head"]
                ),
                "repeats": len(repeats),
                "sources": [row["source"] for row in repeats],
                **{
                    field: percentile_range(
                        [float(row[field]) for row in repeats]
                    )
                    for field in metric_fields
                },
            }
        )

    expected = {
        (profile, length)
        for profile, _ in grouped
        for length in history_tokens
    }
    missing = sorted(expected - set(grouped))
    return {
        "schema": "qksieve_deployment_profile_multiseed_summary_v1",
        "contract": {
            "timing": "direct complete CUDA paths; no stage sums",
            "aggregation": "median with observed min/max",
            "hardware": sorted(hardware),
            "requested_history_tokens": sorted(history_tokens),
        },
        "rows": rows,
        "missing_profile_length_pairs": [list(item) for item in missing],
    }


def main() -> None:
    args = parse_args()
    lengths = {int(item) for item in args.history_tokens.split(",") if item}
    documents = [
        (str(path), json.loads(path.read_text(encoding="utf-8")))
        for path in args.input
    ]
    result = aggregate_documents(documents, lengths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
