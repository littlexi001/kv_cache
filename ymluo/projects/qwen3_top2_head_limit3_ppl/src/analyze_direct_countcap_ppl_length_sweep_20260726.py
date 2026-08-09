from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


LENGTHS = (2048, 4096, 8192, 16000, 24000, 32000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric_fields = (
        "tokens",
        "nll",
        "ppl",
        "dense_prompt_seconds",
        "sparse_decode_seconds",
        "configured_attention_tokens_mean",
        "configured_attention_ratio_mean",
        "actual_attention_tokens_mean",
        "actual_attention_tokens_min",
        "actual_attention_tokens_max",
    )
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])
    return rows


def aggregate(
    rows: list[dict[str, Any]],
    method: str,
    topic: str | None = None,
) -> dict[str, float]:
    selected = [
        row
        for row in rows
        if row["method"] == method
        and (topic is None or row["topic"] == topic)
    ]
    if not selected:
        raise ValueError(f"missing rows for method={method}, topic={topic}")
    tokens = sum(row["tokens"] for row in selected)
    steps = sum(row["tokens"] - 1 for row in selected)
    nll = sum(row["nll"] * row["tokens"] for row in selected) / tokens
    return {
        "cases": float(len(selected)),
        "tokens": tokens,
        "nll": nll,
        "ppl": math.exp(nll),
        "prefill_seconds": sum(
            row["dense_prompt_seconds"] for row in selected
        ),
        "decode_seconds": sum(
            row["sparse_decode_seconds"] for row in selected
        ),
        "milliseconds_per_step": (
            1000
            * sum(row["sparse_decode_seconds"] for row in selected)
            / steps
        ),
        "configured_tokens": (
            sum(
                row["configured_attention_tokens_mean"]
                * (row["tokens"] - 1)
                for row in selected
            )
            / steps
        ),
        "actual_tokens": (
            sum(
                row["actual_attention_tokens_mean"]
                * (row["tokens"] - 1)
                for row in selected
            )
            / steps
        ),
        "actual_min": min(
            row["actual_attention_tokens_min"] for row in selected
        ),
        "actual_max": max(
            row["actual_attention_tokens_max"] for row in selected
        ),
    }


def target_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        max(256, min(math.ceil(0.06 * history_tokens), 1280)),
    )


def candidate_capacity(history_tokens: int) -> int:
    selected_fraction = target_count(history_tokens) / history_tokens
    capacity_fraction = max(
        2.0 * selected_fraction,
        selected_fraction + 0.04,
    )
    return min(
        history_tokens,
        math.ceil(capacity_fraction * history_tokens),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detailed_rows = []
    for length in LENGTHS:
        rows = read_rows(
            args.input_dir / f"length{length}" / "case_summary.csv"
        )
        if len(rows) != 12:
            raise RuntimeError(
                f"length {length} has {len(rows)} rows, expected 12"
            )
        full = aggregate(rows, "full_attention")
        direct = aggregate(rows, "direct_countcap")
        count = target_count(length)
        summary = {
            "history_tokens": length,
            "target_attention_tokens": count,
            "target_ratio": count / length,
            "candidate_capacity": candidate_capacity(length),
            "configured_tokens_mean": direct["configured_tokens"],
            "raw_threshold_hits_mean": direct["actual_tokens"],
            "raw_threshold_hits_min": direct["actual_min"],
            "raw_threshold_hits_max": direct["actual_max"],
            "full_nll": full["nll"],
            "direct_nll": direct["nll"],
            "delta_nll": direct["nll"] - full["nll"],
            "full_ppl": full["ppl"],
            "direct_ppl": direct["ppl"],
            "ppl_change_percent": (
                100 * (direct["ppl"] / full["ppl"] - 1)
            ),
            "ppl_retention_percent": (
                100 * full["ppl"] / direct["ppl"]
            ),
            "full_milliseconds_per_step": full["milliseconds_per_step"],
            "direct_milliseconds_per_step": direct[
                "milliseconds_per_step"
            ],
            "decode_speedup": (
                full["decode_seconds"] / direct["decode_seconds"]
            ),
            "protocol_speedup": (
                (full["prefill_seconds"] + full["decode_seconds"])
                / (direct["prefill_seconds"] + direct["decode_seconds"])
            ),
        }
        summary_rows.append(summary)
        detailed_rows.append(
            {
                **summary,
                "topics": {
                    topic: {
                        "full": aggregate(rows, "full_attention", topic),
                        "direct": aggregate(rows, "direct_countcap", topic),
                    }
                    for topic in ("mixed_a", "mixed_b")
                },
            }
        )
    write_csv(output_dir / "length_summary.csv", summary_rows)
    (output_dir / "length_summary.json").write_text(
        json.dumps(detailed_rows, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
