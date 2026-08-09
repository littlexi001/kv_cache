from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


TOPICS = (
    "sports",
    "medicine",
    "computer",
    "space",
    "politics",
    "religion",
)


def read_list_record(path: Path, topic: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    exact = [
        row
        for row in payload
        if str(row.get("topic")) == topic
    ]
    if exact:
        return dict(exact[0])
    if len(payload) == 1:
        return dict(payload[0])
    raise ValueError(f"no topic={topic} record in {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geometric_ppl(records: list[dict[str, Any]]) -> float:
    tokens = sum(int(record["tokens"]) for record in records)
    nll = sum(
        float(record["nll"]) * int(record["tokens"])
        for record in records
    ) / tokens
    return math.exp(min(20.0, nll))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize deployable packed qMSE prefill-calibrated PPL."
    )
    parser.add_argument("--packed_root", type=Path, required=True)
    parser.add_argument("--quality_reference_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_topic = []
    full_records = []
    packed_records = []
    old_b12_records = []
    old_total15_records = []
    for topic in TOPICS:
        full = read_list_record(
            args.quality_reference_root
            / f"full_l96000_{topic}"
            / "summary.json",
            topic,
        )
        packed = read_list_record(
            args.packed_root
            / f"packed_l96000_{topic}"
            / "case_summary.json",
            topic,
        )
        old_b12 = read_list_record(
            args.quality_reference_root
            / f"autoqmse12z_l96000_{topic}"
            / "summary.json",
            topic,
        )
        old_total15 = read_list_record(
            args.quality_reference_root
            / f"autoqmsetotal15z_l96000_{topic}"
            / "summary.json",
            topic,
        )
        full_records.append(full)
        packed_records.append(packed)
        old_b12_records.append(old_b12)
        old_total15_records.append(old_total15)
        per_topic.append(
            {
                "topic": topic,
                "full_ppl": float(full["ppl"]),
                "packed_prefillcal_ppl": float(packed["ppl"]),
                "quality_retention": math.exp(
                    float(full["nll"]) - float(packed["nll"])
                ),
                "old_qmse_b12_ppl": float(old_b12["ppl"]),
                "old_qmse_total_b15_ppl": float(old_total15["ppl"]),
                "actual_attention_tokens_mean": float(
                    packed["actual_attention_tokens_mean"]
                ),
                "actual_attention_ratio_mean": float(
                    packed["actual_attention_tokens_mean"]
                )
                / 96000.0,
                "steady_seconds_per_step": float(
                    packed["steady_sparse_seconds_per_step"]
                ),
                "fixed_overhead_seconds": float(
                    packed["fixed_sparse_overhead_seconds"]
                ),
                "index_ratio_of_full_kv": float(
                    packed["packed_index_ratio_of_full_kv"]
                ),
                "overflow_rate_mean": float(
                    packed["candidate_overflow_rate_mean"]
                ),
            }
        )

    full_ppl = geometric_ppl(full_records)
    packed_ppl = geometric_ppl(packed_records)
    old_b12_ppl = geometric_ppl(old_b12_records)
    old_total15_ppl = geometric_ppl(old_total15_records)
    aggregate = {
        "topics": len(TOPICS),
        "tokens": sum(int(record["tokens"]) for record in packed_records),
        "full_geometric_ppl": full_ppl,
        "packed_prefillcal_geometric_ppl": packed_ppl,
        "packed_prefillcal_quality_retention": full_ppl / packed_ppl,
        "packed_prefillcal_worst_topic_retention": min(
            float(row["quality_retention"]) for row in per_topic
        ),
        "old_qmse_b12_geometric_ppl": old_b12_ppl,
        "old_qmse_b12_quality_retention": full_ppl / old_b12_ppl,
        "old_qmse_total_b15_geometric_ppl": old_total15_ppl,
        "old_qmse_total_b15_quality_retention": full_ppl / old_total15_ppl,
        "actual_attention_tokens_mean": sum(
            float(row["actual_attention_tokens_mean"])
            for row in per_topic
        )
        / len(per_topic),
        "actual_attention_ratio_mean": sum(
            float(row["actual_attention_ratio_mean"])
            for row in per_topic
        )
        / len(per_topic),
        "index_ratio_of_full_kv_mean": sum(
            float(row["index_ratio_of_full_kv"]) for row in per_topic
        )
        / len(per_topic),
        "steady_seconds_per_step_mean": sum(
            float(row["steady_seconds_per_step"]) for row in per_topic
        )
        / len(per_topic),
        "fixed_overhead_seconds_mean": sum(
            float(row["fixed_overhead_seconds"]) for row in per_topic
        )
        / len(per_topic),
        "overflow_rate_mean": sum(
            float(row["overflow_rate_mean"]) for row in per_topic
        )
        / len(per_topic),
    }

    same_harness_path = (
        args.packed_root
        / "full_l96000_politics_sameharness"
        / "case_summary.json"
    )
    if same_harness_path.exists():
        full_speed = read_list_record(same_harness_path, "politics")
        packed_speed = read_list_record(
            args.packed_root
            / "packed_l96000_politics"
            / "case_summary.json",
            "politics",
        )
        full_step = float(full_speed["steady_sparse_seconds_per_step"])
        packed_step = float(packed_speed["steady_sparse_seconds_per_step"])
        fixed = float(packed_speed["fixed_sparse_overhead_seconds"])
        speed = {
            "topic": "politics",
            "full_seconds_per_step": full_step,
            "packed_steady_seconds_per_step": packed_step,
            "steady_speedup": full_step / packed_step,
            "measured_63_step_speedup_including_fixed": float(
                full_speed["sparse_decode_seconds"]
            )
            / float(packed_speed["sparse_decode_seconds"]),
            "break_even_generated_steps": (
                fixed / (full_step - packed_step)
                if full_step > packed_step
                else math.inf
            ),
            "projected_1024_step_speedup_including_fixed": (
                1024.0 * full_step
                / (fixed + 1024.0 * packed_step)
            ),
        }
    else:
        speed = {}

    output = {
        "packed_root": str(args.packed_root),
        "quality_reference_root": str(args.quality_reference_root),
        "per_topic": per_topic,
        "aggregate": aggregate,
        "same_harness_speed": speed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_topic.csv", per_topic)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
