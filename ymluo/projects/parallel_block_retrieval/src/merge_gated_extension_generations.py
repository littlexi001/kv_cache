from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge low-confidence extension generations with frozen Top3 rows."
    )
    parser.add_argument("--base_rows_path", required=True)
    parser.add_argument("--extension_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_generation_row(
    base: dict[str, Any], extension: dict[str, Any] | None
) -> dict[str, Any]:
    branches = [dict(branch) for branch in base["branches"]]
    if extension is not None:
        branches.extend(dict(branch) for branch in extension["branches"])
    branches.sort(key=lambda branch: (int(branch["rank"]), int(branch["selected_block"])))
    top = branches[0]
    merged = dict(base)
    merged["branches"] = branches
    merged["retrieval_target_span_hit"] = bool(top["retrieval_target_span_hit"])
    merged["retrieval_target_span_hit_at_k"] = any(
        bool(branch["retrieval_target_span_hit"]) for branch in branches
    )
    merged["target_hit"] = bool(top["target_hit"])
    merged["target_f1"] = float(top["target_f1"])
    merged["any_branch_target_hit"] = any(
        bool(branch["target_hit"]) for branch in branches
    )
    merged["total_branch_generation_seconds"] = sum(
        float(branch["generation_seconds"]) for branch in branches
    )
    merged["parallel_branch_critical_seconds"] = max(
        float(branch["generation_seconds"]) for branch in branches
    )
    merged["confidence_gated_extension"] = extension is not None
    return merged


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(Path(args.base_rows_path))
    extension = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.extension_rows_path))
    }
    base_ids = {int(row["query_id"]) for row in base_rows}
    if not set(extension) <= base_ids:
        raise ValueError("extension contains unknown queries")
    rows = [
        merge_generation_row(row, extension.get(int(row["query_id"])))
        for row in base_rows
    ]
    rows.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "Top3 generation plus confidence-gated deeper branches",
        "selection_uses_gold": False,
        "queries": len(rows),
        "expanded_queries": len(extension),
        "mean_blocks": statistics.fmean(len(row["branches"]) for row in rows),
        "top1_accuracy": statistics.fmean(bool(row["target_hit"]) for row in rows),
        "any_branch_accuracy": statistics.fmean(
            bool(row["any_branch_target_hit"]) for row in rows
        ),
        "mean_total_branch_generation_seconds": statistics.fmean(
            float(row["total_branch_generation_seconds"]) for row in rows
        ),
        "mean_parallel_branch_critical_seconds": statistics.fmean(
            float(row["parallel_branch_critical_seconds"]) for row in rows
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
