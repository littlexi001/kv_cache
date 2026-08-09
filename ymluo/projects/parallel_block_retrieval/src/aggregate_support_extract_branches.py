from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_stepwise_set_utility import mcnemar_exact_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-block SUPPORTED/NOT_SUPPORTED extractions without gold labels."
    )
    parser.add_argument("--rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_rows_path", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def grounded(candidate: str, memory: str) -> bool:
    value = normalize(candidate)
    return bool(value) and value in normalize(memory)


def selected_indices(row: dict[str, Any]) -> dict[str, int]:
    valid = [
        index
        for index, branch in enumerate(row["branches"])
        if str(branch.get("state_text", "")).strip()
    ]
    if not valid:
        return {"first_supported": -1, "first_grounded": -1, "consensus": -1}
    first_grounded = next(
        (
            index
            for index in valid
            if grounded(
                str(row["branches"][index]["state_text"]),
                str(row["branches"][index]["memory_text"]),
            )
        ),
        -1,
    )
    groups = Counter(
        normalize(str(row["branches"][index]["state_text"])) for index in valid
    )
    consensus_value = min(
        groups,
        key=lambda value: (
            -groups[value],
            min(
                int(row["branches"][index]["block_rank"])
                for index in valid
                if normalize(str(row["branches"][index]["state_text"])) == value
            ),
            value,
        ),
    )
    consensus = next(
        index
        for index in valid
        if normalize(str(row["branches"][index]["state_text"])) == consensus_value
    )
    return {
        "first_supported": valid[0],
        "first_grounded": first_grounded if first_grounded >= 0 else valid[0],
        "consensus": consensus,
    }


def method_hits(rows: list[dict[str, Any]], field: str) -> list[bool]:
    output = []
    for row in rows:
        index = selected_indices(row)[field]
        output.append(index >= 0 and bool(row["branches"][index]["target_hit"]))
    return output


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.rows_path))
    selections = []
    methods = {}
    for method in ("first_supported", "first_grounded", "consensus"):
        hits = method_hits(rows, method)
        methods[method] = {
            "correct": sum(hits),
            "accuracy": statistics.fmean(hits),
        }
    oracle = [
        any(bool(branch["target_hit"]) for branch in row["branches"]) for row in rows
    ]
    valid_counts = [
        sum(bool(str(branch.get("state_text", "")).strip()) for branch in row["branches"])
        for row in rows
    ]
    for row in rows:
        indices = selected_indices(row)
        selections.append(
            {
                "query_id": int(row["query_id"]),
                "indices": indices,
                "answers": {
                    method: (
                        str(row["branches"][index]["state_text"])
                        if index >= 0
                        else ""
                    )
                    for method, index in indices.items()
                },
            }
        )
    payload: dict[str, Any] = {
        "source": "deterministic aggregation of structured per-block extractions",
        "selection_uses_gold": False,
        "queries": len(rows),
        "branches_per_query": len(rows[0]["branches"]) if rows else 0,
        "methods": methods,
        "oracle_any_branch": {
            "correct": sum(oracle),
            "accuracy": statistics.fmean(oracle),
        },
        "mean_supported_branches": statistics.fmean(valid_counts),
        "queries_without_supported_branch": sum(count == 0 for count in valid_counts),
    }
    if args.baseline_rows_path:
        baseline_rows = read_jsonl(Path(args.baseline_rows_path))
        baseline = [bool(row["branches"][0]["target_hit"]) for row in baseline_rows]
        if len(baseline) != len(rows):
            raise ValueError("baseline and extraction rows do not align")
        for method in methods:
            current = method_hits(rows, method)
            wins = sum(new and not old for old, new in zip(baseline, current, strict=True))
            losses = sum(old and not new for old, new in zip(baseline, current, strict=True))
            methods[method]["vs_baseline"] = {
                "wins": wins,
                "losses": losses,
                "mcnemar_p": mcnemar_exact_p(wins, losses),
            }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "selections.jsonl").open("w", encoding="utf-8") as handle:
        for row in selections:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
