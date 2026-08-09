from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METHOD_PREFIXES = {
    "full": "full_",
    "qmse_b12": "autoqmse12z_",
    "qmse_total_b15": "autoqmsetotal15z_",
}


def read_record(path: Path, topic: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a list in {path}")
    matching = [
        row
        for row in payload
        if str(row.get("topic")) in {topic, "all"}
    ]
    if not matching:
        raise ValueError(f"no topic={topic} record in {path}")
    exact = [row for row in matching if str(row.get("topic")) == topic]
    return dict(exact[0] if exact else matching[0])


def parse_label(label: str, prefix: str) -> tuple[int, str]:
    remainder = label[len(prefix) :]
    length_label, topic = remainder.split("_", 1)
    if not length_label.startswith("l"):
        raise ValueError(f"invalid length label: {label}")
    return int(length_label[1:]), topic


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["history_tokens"]), str(row["method"])), []
        ).append(row)
    output = []
    full_by_length: dict[int, dict[str, Any]] = {}
    for (history_tokens, method), items in sorted(grouped.items()):
        token_count = sum(int(item["tokens"]) for item in items)
        nll = sum(
            float(item["nll"]) * int(item["tokens"]) for item in items
        ) / token_count
        record = {
            "history_tokens": history_tokens,
            "method": method,
            "topics": len(items),
            "tokens": token_count,
            "nll": nll,
            "geometric_ppl": math.exp(min(20.0, nll)),
            "mean_online_seconds": sum(
                float(item["online_seconds"]) for item in items
            )
            / len(items),
        }
        if method != "full":
            fractions = [
                float(item["candidate_fraction"]) for item in items
                if item.get("candidate_fraction") is not None
            ]
            record["candidate_fraction_mean"] = (
                sum(fractions) / len(fractions) if fractions else None
            )
        output.append(record)
        if method == "full":
            full_by_length[history_tokens] = record
    for record in output:
        if record["method"] == "full":
            record["quality_retention"] = 1.0
            continue
        full = full_by_length[int(record["history_tokens"])]
        record["quality_retention"] = (
            float(full["geometric_ppl"]) / float(record["geometric_ppl"])
        )
        paired = [
            row
            for row in rows
            if int(row["history_tokens"]) == int(record["history_tokens"])
            and str(row["method"]) == str(record["method"])
        ]
        full_topic = {
            str(row["topic"]): row
            for row in rows
            if int(row["history_tokens"]) == int(record["history_tokens"])
            and str(row["method"]) == "full"
        }
        topic_ratios = [
            math.exp(
                float(full_topic[str(row["topic"])]["nll"])
                - float(row["nll"])
            )
            for row in paired
            if str(row["topic"]) in full_topic
        ]
        record["quality_retention_worst_topic"] = min(topic_ratios)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize hierarchical qMSE PPL runs by length and topic."
    )
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for method, prefix in METHOD_PREFIXES.items():
        for directory in sorted(args.input_root.glob(f"{prefix}*")):
            summary_path = directory / "summary.json"
            if not summary_path.exists():
                continue
            history_tokens, topic = parse_label(directory.name, prefix)
            record = read_record(summary_path, topic)
            rows.append(
                {
                    "history_tokens": history_tokens,
                    "topic": topic,
                    "method": method,
                    "tokens": int(record["tokens"]),
                    "nll": float(record["nll"]),
                    "ppl": float(record["ppl"]),
                    "online_seconds": float(
                        record.get(
                            "online_seconds",
                            record.get("mean_online_seconds_per_case", 0.0),
                        )
                    ),
                    "candidate_fraction": record.get(
                        "candidate_fraction_mean",
                        record.get("candidate_fraction"),
                    ),
                }
            )
    output = {
        "input_root": str(args.input_root),
        "per_topic": rows,
        "aggregate": aggregate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
