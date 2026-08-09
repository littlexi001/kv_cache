from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("record_candidate_overlap", False):
        raise ValueError("candidate-overlap tracing was not enabled")
    if payload.get("candidate_selection_mode") != "per_head_stream":
        raise ValueError("candidate overlap requires per_head_stream")
    mean_union = payload.get("mean_candidate_union_fraction")
    maximum_union = payload.get("max_candidate_union_fraction")
    if mean_union is None or maximum_union is None:
        raise ValueError("candidate union statistics are missing")
    raw_fraction = float(payload["candidate_fraction"]) * int(
        payload["stream_group_size"]
    )
    history_tokens = int(payload["history_tokens"])
    stream_group_size = int(payload["stream_group_size"])
    # Each stream selects ceil(fraction * history), so the realized union can
    # exceed the continuous fraction by at most one token per stream.
    discrete_upper_bound = (
        raw_fraction + stream_group_size / max(1, history_tokens)
    )
    mean_union = float(mean_union)
    maximum_union = float(maximum_union)
    if not 0 < mean_union <= maximum_union <= discrete_upper_bound + 1.0e-6:
        raise ValueError("candidate union fractions are outside the physical bound")
    return {
        "topic": payload["topic"],
        "history_tokens": history_tokens,
        "eval_tokens": int(payload["eval_tokens"]),
        "candidate_fraction_per_head": float(payload["candidate_fraction"]),
        "stream_group_size": int(payload["stream_group_size"]),
        "raw_concatenated_fraction": raw_fraction,
        "discrete_union_upper_bound": discrete_upper_bound,
        "mean_unique_union_fraction": mean_union,
        "max_unique_union_fraction": maximum_union,
        "mean_duplicate_candidate_rate": 1.0 - mean_union / raw_fraction,
        "ppl": float(payload["ppl"]),
        "online_seconds_with_trace_overhead": float(payload["online_seconds"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        summarize_payload(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(args.input_dir.glob("*.json"))
        if path.name != "summary.json"
    ]
    if len(rows) != 2:
        raise ValueError(f"expected two candidate-overlap cases, found {len(rows)}")
    summary = {
        "status": "diagnostic_only_trace_overhead_not_a_speed_result",
        "cases": len(rows),
        "rows": rows,
        "mean_duplicate_candidate_rate": sum(
            float(row["mean_duplicate_candidate_rate"]) for row in rows
        )
        / len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
