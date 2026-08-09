from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


CASES = (
    ("mixed_a", 0),
    ("mixed_a", 1),
    ("mixed_b", 0),
    ("mixed_b", 1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict paired 128K comparison of Full KV, old CountCap, and the "
            "deployable packed hierarchical qMSE method."
        )
    )
    parser.add_argument("--packed_root", type=Path, required=True)
    parser.add_argument("--old_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"expected list in {path}")
    return [dict(row) for row in payload]


def select_record(
    records: list[dict[str, Any]],
    *,
    topic: str,
    window: int,
    method: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in records
        if str(row.get("topic")) == topic
        and int(row.get("window", -1)) == window
        and str(row.get("method")) == method
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {method} record for {topic}/w{window}, "
            f"found {len(matches)}"
        )
    return matches[0]


def geometric_ppl(records: list[dict[str, Any]]) -> float:
    tokens = sum(int(row["tokens"]) for row in records)
    if tokens <= 0:
        raise ValueError("no evaluated tokens")
    mean_nll = sum(
        int(row["tokens"]) * float(row["nll"]) for row in records
    ) / tokens
    return math.exp(min(20.0, mean_nll))


def arithmetic_mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    old_by_topic = {
        topic: read_records(
            args.old_root
            / f"length128000_{topic}"
            / "case_summary.json"
        )
        for topic, _ in CASES
    }

    full_records = []
    old_records = []
    packed_records = []
    per_case = []
    for topic, window in CASES:
        old_payload = old_by_topic[topic]
        full = select_record(
            old_payload,
            topic=topic,
            window=window,
            method="full_attention",
        )
        old = select_record(
            old_payload,
            topic=topic,
            window=window,
            method="direct_countcap",
        )
        packed = select_record(
            read_records(
                args.packed_root / f"{topic}_w{window}" / "case_summary.json"
            ),
            topic=topic,
            window=window,
            method="direct_countcap",
        )
        if not (
            int(full["tokens"])
            == int(old["tokens"])
            == int(packed["tokens"])
        ):
            raise ValueError(f"token mismatch for {topic}/w{window}")
        if not (
            int(full["target_start"])
            == int(old["target_start"])
            == int(packed["target_start"])
        ):
            raise ValueError(f"target mismatch for {topic}/w{window}")

        full_records.append(full)
        old_records.append(old)
        packed_records.append(packed)

        full_step = float(full["sparse_seconds_per_step"])
        packed_step = float(packed["steady_sparse_seconds_per_step"])
        fixed = float(packed["fixed_sparse_overhead_seconds"])
        tokens = int(packed["tokens"])
        break_even = (
            fixed / (full_step - packed_step)
            if full_step > packed_step
            else math.inf
        )
        per_case.append(
            {
                "topic": topic,
                "window": window,
                "tokens": tokens,
                "target_start": int(packed["target_start"]),
                "full_ppl": float(full["ppl"]),
                "old_countcap_ppl": float(old["ppl"]),
                "packed_qmse_ppl": float(packed["ppl"]),
                "old_quality_retention": math.exp(
                    float(full["nll"]) - float(old["nll"])
                ),
                "packed_quality_retention": math.exp(
                    float(full["nll"]) - float(packed["nll"])
                ),
                "full_seconds_per_step": full_step,
                "old_countcap_seconds_per_step": float(
                    old["sparse_seconds_per_step"]
                ),
                "packed_steady_seconds_per_step": packed_step,
                "old_countcap_steady_speedup": full_step
                / float(old["sparse_seconds_per_step"]),
                "packed_steady_speedup": full_step / packed_step,
                "packed_fixed_overhead_seconds": fixed,
                "packed_measured_speedup_including_fixed": float(
                    full["sparse_decode_seconds"]
                )
                / float(packed["sparse_decode_seconds"]),
                "packed_break_even_generated_steps": break_even,
                "packed_projected_1024_speedup": (
                    1024.0 * full_step
                    / (fixed + 1024.0 * packed_step)
                ),
                "actual_attention_tokens_mean": float(
                    packed["actual_attention_tokens_mean"]
                ),
                "actual_attention_ratio": float(
                    packed["actual_attention_tokens_mean"]
                )
                / 128000.0,
                "packed_index_ratio_of_full_kv": float(
                    packed["packed_index_ratio_of_full_kv"]
                ),
                "candidate_overflow_rate_mean": float(
                    packed["candidate_overflow_rate_mean"]
                ),
            }
        )

    full_ppl = geometric_ppl(full_records)
    old_ppl = geometric_ppl(old_records)
    packed_ppl = geometric_ppl(packed_records)
    aggregate = {
        "cases": len(CASES),
        "tokens": sum(int(row["tokens"]) for row in packed_records),
        "full_geometric_ppl": full_ppl,
        "old_countcap_geometric_ppl": old_ppl,
        "packed_qmse_geometric_ppl": packed_ppl,
        "old_countcap_quality_retention": full_ppl / old_ppl,
        "packed_qmse_quality_retention": full_ppl / packed_ppl,
        "old_countcap_worst_case_retention": min(
            float(row["old_quality_retention"]) for row in per_case
        ),
        "packed_qmse_worst_case_retention": min(
            float(row["packed_quality_retention"]) for row in per_case
        ),
        "packed_actual_attention_tokens_mean": arithmetic_mean(
            per_case, "actual_attention_tokens_mean"
        ),
        "packed_actual_attention_ratio_mean": arithmetic_mean(
            per_case, "actual_attention_ratio"
        ),
        "packed_index_ratio_of_full_kv_mean": arithmetic_mean(
            per_case, "packed_index_ratio_of_full_kv"
        ),
        "packed_steady_speedup_mean": arithmetic_mean(
            per_case, "packed_steady_speedup"
        ),
        "packed_measured_256_speedup_mean": arithmetic_mean(
            per_case, "packed_measured_speedup_including_fixed"
        ),
        "packed_break_even_generated_steps_mean": arithmetic_mean(
            per_case, "packed_break_even_generated_steps"
        ),
        "packed_projected_1024_speedup_mean": arithmetic_mean(
            per_case, "packed_projected_1024_speedup"
        ),
        "candidate_overflow_rate_mean": arithmetic_mean(
            per_case, "candidate_overflow_rate_mean"
        ),
    }

    output = {
        "packed_root": str(args.packed_root),
        "old_root": str(args.old_root),
        "per_case": per_case,
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case.csv", per_case)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
