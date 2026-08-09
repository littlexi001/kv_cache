from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare monotonic QK expansion with equal-budget coarse expansion."
    )
    parser.add_argument("--rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--base_blocks", type=int, default=3)
    parser.add_argument("--max_extra", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    rows: list[dict[str, Any]], method: str, base_blocks: int, extras: int
) -> dict[str, Any]:
    selected_rows = []
    for row in rows:
        selected = [int(item) for item in row["candidate_candidates"][:base_blocks]]
        for block_id in row[f"{method}_candidates"][:extras]:
            block_id = int(block_id)
            if block_id not in selected:
                selected.append(block_id)
        target = int(row["target_block_id"])
        lexical_budget = len(selected)
        selected_rows.append(
            {
                "hit": target in selected,
                "blocks": len(selected),
                "lexical_equal_budget_hit": target
                in row["candidate_candidates"][:lexical_budget],
            }
        )
    return {
        "extra_qk_candidates": extras,
        "mean_blocks": statistics.fmean(item["blocks"] for item in selected_rows),
        "qk_expansion_recall": statistics.fmean(item["hit"] for item in selected_rows),
        "equal_budget_lexical_recall": statistics.fmean(
            item["lexical_equal_budget_hit"] for item in selected_rows
        ),
    }


def main() -> None:
    args = parse_args()
    if args.base_blocks <= 0 or args.max_extra < 0:
        raise ValueError("base_blocks must be positive and max_extra non-negative")
    rows = read_jsonl(Path(args.rows_path))
    payload: dict[str, Any] = {
        "source": "monotonic QK candidate expansion with per-row equal block budget",
        "selection_uses_gold": False,
        "base_blocks": args.base_blocks,
        "methods": {},
    }
    for method in ("full128", "svd"):
        groups = []
        for split, step_type in sorted(
            {(str(row["split"]), str(row["step_type"])) for row in rows}
        ):
            group = [
                row
                for row in rows
                if str(row["split"]) == split
                and str(row["step_type"]) == step_type
            ]
            groups.append(
                {
                    "split": split,
                    "step_type": step_type,
                    "steps": len(group),
                    "expansions": [
                        evaluate(group, method, args.base_blocks, extras)
                        for extras in range(args.max_extra + 1)
                    ],
                }
            )
        payload["methods"][method] = groups
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
