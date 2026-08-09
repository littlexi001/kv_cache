from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize packed-index attention-kernel and whole-model timing "
            "without conflating steady-state and index-amortized speedups."
        )
    )
    parser.add_argument("--operator_json", required=True, type=Path)
    parser.add_argument("--length_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--projection_steps",
        default="64,256,1024",
        help="Comma-separated generation lengths for amortized projections.",
    )
    return parser.parse_args()


def read_case(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_method = {str(row["method"]): row for row in rows}
    return by_method["full_attention"], by_method["direct_countcap"]


def projected_speedup(
    full_seconds_per_step: float,
    sparse_seconds_per_step: float,
    fixed_seconds: float,
    steps: int,
) -> float:
    return (
        full_seconds_per_step * steps
        / (fixed_seconds + sparse_seconds_per_step * steps)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    projection_steps = [
        int(item) for item in args.projection_steps.split(",") if item.strip()
    ]
    operator = json.loads(args.operator_json.read_text(encoding="utf-8"))
    operator_by_length = {
        int(row["history_tokens"]): row for row in operator["rows"]
    }

    rows: list[dict[str, Any]] = []
    for case_path in sorted(
        args.length_root.glob("*/case_summary.json"),
        key=lambda path: int(path.parent.name),
    ):
        full, sparse = read_case(case_path)
        history_tokens = int(full["history_tokens"])
        full_step = float(full["steady_sparse_seconds_per_step"])
        sparse_step = float(sparse["steady_sparse_seconds_per_step"])
        fixed = float(sparse["fixed_sparse_overhead_seconds"])
        saving = full_step - sparse_step
        operator_row = operator_by_length.get(history_tokens, {})
        row: dict[str, Any] = {
            "history_tokens": history_tokens,
            "evaluation_tokens": int(sparse["tokens"]),
            "quality_retention": float(sparse["quality_retention"]),
            "actual_attention_fraction": float(
                sparse["actual_attention_tokens_mean"] / history_tokens
            ),
            "index_ratio_of_full_kv": float(
                sparse["packed_index_ratio_of_full_kv"]
            ),
            "full_model_ms_per_token": full_step * 1000.0,
            "sparse_model_steady_ms_per_token": sparse_step * 1000.0,
            "whole_model_steady_speedup": full_step / sparse_step,
            "fixed_index_seconds": fixed,
            "measured_finite_run_speedup": float(
                full["sparse_decode_seconds"]
                / sparse["sparse_decode_seconds"]
            ),
            "break_even_generated_tokens": (
                fixed / saving if saving > 0.0 else None
            ),
            "attention_subsystem_speedup": operator_row.get(
                "full_sdpa_over_complete_plus_query_prepare"
            ),
            "sharedtail_attention_subsystem_speedup": operator_row.get(
                "full_sdpa_over_sharedtail_complete_plus_query_prepare"
            ),
        }
        for steps in projection_steps:
            row[f"projected_speedup_{steps}_tokens"] = projected_speedup(
                full_step,
                sparse_step,
                fixed,
                steps,
            )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "length_speed_summary.csv", rows)
    output = {
        "scope": {
            "attention_subsystem": operator.get("scope", ""),
            "whole_model": (
                "One teacher-forced decode stream through the complete "
                "HuggingFace Qwen3-4B model. Steady speed excludes one-time "
                "index construction; finite-run and projected speed include it."
            ),
            "quality_warning": (
                "Each length uses only one 64-token mixed_a window. These PPL "
                "numbers are smoke tests, not independent paper-quality "
                "estimates."
            ),
        },
        "projection_steps": projection_steps,
        "rows": rows,
    }
    (args.output_dir / "length_speed_summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
