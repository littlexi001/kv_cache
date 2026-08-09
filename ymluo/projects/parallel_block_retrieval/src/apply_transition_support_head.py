from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_branch_transition_verifier import choose_branch
from train_transition_support_head import (
    runtime_transition_scores,
    transition_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a frozen transition head without reading evaluation labels."
    )
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--generation_rows_path", required=True)
    parser.add_argument("--head_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--method", default="heuristic_structured")
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--step_type",
        choices=["resolve_bridge", "resolve_answer_from_bridge"],
        required=True,
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    steps = {
        (int(row["query_id"]), int(row["step_index"])): row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) == args.split and row["step_type"] == args.step_type
    }
    generations = [
        row
        for row in read_jsonl(Path(args.generation_rows_path))
        if str(row["split"]) == args.split and row["step_type"] == args.step_type
    ]
    generations.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    head = json.loads(Path(args.head_path).read_text(encoding="utf-8"))
    method = head["methods"][args.method]
    feature_indices = [int(item) for item in method["feature_indices"]]
    parameters = method["runtime_parameters"]

    output_rows = []
    for generation in generations:
        key = (int(generation["query_id"]), int(generation["step_index"]))
        step = steps[key]
        heuristic_index, traces = choose_branch(step, generation["branches"])
        feature_row = {
            "heuristic_trace": traces,
            "yes_no_scores": [0.0] * len(generation["branches"]),
            "answer_logprob_scores": [0.0] * len(generation["branches"]),
            "branch_retrieval_ranks": [
                int(branch["rank"]) for branch in generation["branches"]
            ],
            "branch_generated_tokens": [
                int(branch["generated_tokens"]) for branch in generation["branches"]
            ],
        }
        features = transition_features(feature_row)[:, feature_indices]
        scores = runtime_transition_scores(features, parameters)
        selected_index = int(np.argmax(scores))
        output_rows.append(
            {
                "query_id": key[0],
                "step_index": key[1],
                "step_type": args.step_type,
                "selection_uses_gold": False,
                "heuristic_index": heuristic_index,
                f"{args.method}_index": selected_index,
                f"{args.method}_scores": [float(item) for item in scores],
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "source": "frozen transition head deployment selections",
                "selection_uses_gold": False,
                "method": args.method,
                "rows": len(output_rows),
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
