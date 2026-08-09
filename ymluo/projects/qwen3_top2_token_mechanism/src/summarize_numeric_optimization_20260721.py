from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


TOPICS = ("medicine", "politics", "computer", "space")


def read_sparse(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))[0]


def read_full(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return next(row for row in rows if row["topic"] != "all")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    nll = sum(float(row["nll"]) for row in rows) / len(rows)
    online = sum(float(row["online_seconds"]) for row in rows) / len(rows)
    return {
        "macro_nll": nll,
        "geometric_ppl": math.exp(nll),
        "mean_online_seconds": online,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_root", type=Path, required=True)
    parser.add_argument("--candidate8_root", type=Path, required=True)
    parser.add_argument("--candidate_frontier_root", type=Path, required=True)
    parser.add_argument("--subsystem_speed", type=Path, required=True)
    parser.add_argument("--retrieval_frontier", type=Path, required=True)
    parser.add_argument("--basis_frontier", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full_rows = []
    candidate8_rows = []
    candidate_rows: dict[str, list[dict[str, Any]]] = {"0.05": [], "0.06": []}
    per_topic: dict[str, dict[str, Any]] = {}
    for topic in TOPICS:
        full = read_full(args.full_root / topic / "ppl_summary.json")
        full = full | {
            "online_seconds": float(full["mean_online_seconds"]),
        }
        candidate8 = read_sparse(args.candidate8_root / topic / "summary.json")
        full_rows.append(full)
        candidate8_rows.append(candidate8)
        topic_rows: dict[str, Any] = {"full": full, "candidate0.08": candidate8}
        for fraction, tag in (("0.05", "005"), ("0.06", "006")):
            matches = list(
                args.candidate_frontier_root.glob(
                    f"candidate{tag}_{topic}_*/summary.json"
                )
            )
            if matches:
                row = read_sparse(matches[0])
                candidate_rows[fraction].append(row)
                topic_rows[f"candidate{fraction}"] = row
        per_topic[topic] = topic_rows

    aggregates = {
        "full": aggregate(full_rows),
        "candidate0.08": aggregate(candidate8_rows),
    }
    full_online = aggregates["full"]["mean_online_seconds"]
    for name, row in aggregates.items():
        row["quality_retention_percent"] = (
            aggregates["full"]["geometric_ppl"] / row["geometric_ppl"] * 100.0
        )
        row["speedup_vs_full"] = full_online / row["mean_online_seconds"]
    for fraction, rows in candidate_rows.items():
        if len(rows) != len(TOPICS):
            continue
        name = f"candidate{fraction}"
        aggregates[name] = aggregate(rows)
        aggregates[name]["quality_retention_percent"] = (
            aggregates["full"]["geometric_ppl"]
            / aggregates[name]["geometric_ppl"]
            * 100.0
        )
        aggregates[name]["speedup_vs_full"] = (
            full_online / aggregates[name]["mean_online_seconds"]
        )

    payload: dict[str, Any] = {
        "topics": TOPICS,
        "per_topic": per_topic,
        "aggregate": aggregates,
        "subsystem_speed": json.loads(
            args.subsystem_speed.read_text(encoding="utf-8")
        ),
        "retrieval_frontier": json.loads(
            args.retrieval_frontier.read_text(encoding="utf-8")
        ),
    }
    if args.basis_frontier and args.basis_frontier.exists():
        payload["basis_frontier"] = json.loads(
            args.basis_frontier.read_text(encoding="utf-8")
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

