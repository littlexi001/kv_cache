from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print local rule failure-boundary summary.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    summary = read_rows(output_dir / "summary_by_condition.csv")
    boundary = read_rows(output_dir / "failure_boundary.csv")
    if not summary:
        raise SystemExit(f"no summary_by_condition.csv found under {output_dir}")

    print("Lowest candidate-accuracy conditions")
    print("| model | length | depth | dist | sim | gap | chain | comp | cases | cand acc | gen acc | selectivity |")
    print("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(summary, key=lambda item: float(item["candidate_accuracy"]))[: args.limit]:
        print(
            f"| {row['model_label']} | {row['target_context_tokens']} | {row['depth_percent']} | "
            f"{row['distractor_count']} | {row['distractor_similarity']} | {row['rule_gap_tokens']} | "
            f"{row['chain_length']} | {row['competitor_count']} | {row['cases']} | "
            f"{row['candidate_accuracy']} | {row['generation_accuracy']} | {row['mean_rule_attention_selectivity']} |"
        )

    failed = [row for row in boundary if row.get("first_fail_context_tokens")]
    if failed:
        print()
        print("Earliest failure boundaries")
        print("| model | sim | chain | comp | dist | depth | gap | first fail | acc | last pass | acc |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted(failed, key=lambda item: int(item["first_fail_context_tokens"]))[: args.limit]:
            print(
                f"| {row['model_label']} | {row['distractor_similarity']} | {row['chain_length']} | "
                f"{row['competitor_count']} | {row['distractor_count']} | {row['depth_percent']} | "
                f"{row['rule_gap_tokens']} | {row['first_fail_context_tokens']} | {row['first_fail_accuracy']} | "
                f"{row['last_pass_context_tokens']} | {row['last_pass_accuracy']} |"
            )


if __name__ == "__main__":
    main()
