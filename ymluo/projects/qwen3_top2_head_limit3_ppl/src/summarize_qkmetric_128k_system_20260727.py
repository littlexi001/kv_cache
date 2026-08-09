from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate independent 128K Full/Sparse cases into one quality "
            "and system result with explicit fixed-cost amortization."
        )
    )
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument(
        "--full_root",
        type=Path,
        default=None,
        help=(
            "Optional separate root providing matched Full KV case summaries."
        ),
    )
    parser.add_argument("--bootstrap_json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--projection_steps", default="64,256,1024")
    return parser.parse_args()


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def main() -> None:
    args = parse_args()
    projection_steps = [
        int(item) for item in args.projection_steps.split(",") if item.strip()
    ]
    pairs = []
    for path in sorted(args.input_root.glob("*/case_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_method = {str(row["method"]): row for row in payload}
        full_root = (
            args.full_root
            if args.full_root is not None
            else args.input_root
        )
        full_path = full_root / path.parent.name / "case_summary.json"
        full_payload = json.loads(full_path.read_text(encoding="utf-8"))
        full_by_method = {
            str(row["method"]): row for row in full_payload
        }
        pairs.append(
            {
                "case": path.parent.name,
                "full": full_by_method["full_attention"],
                "sparse": by_method["direct_countcap"],
            }
        )
    if not pairs:
        raise ValueError(f"no case_summary.json files under {args.input_root}")

    full_rows = [pair["full"] for pair in pairs]
    sparse_rows = [pair["sparse"] for pair in pairs]
    full_step = mean(full_rows, "steady_sparse_seconds_per_step")
    sparse_step = mean(sparse_rows, "steady_sparse_seconds_per_step")
    fixed = mean(sparse_rows, "fixed_sparse_overhead_seconds")
    bootstrap = json.loads(args.bootstrap_json.read_text(encoding="utf-8"))
    total_full_decode = sum(
        float(row["sparse_decode_seconds"]) for row in full_rows
    )
    total_sparse_decode = sum(
        float(row["sparse_decode_seconds"]) for row in sparse_rows
    )

    system: dict[str, Any] = {
        "full_model_ms_per_token": full_step * 1000.0,
        "sparse_model_steady_ms_per_token": sparse_step * 1000.0,
        "whole_model_steady_speedup": full_step / sparse_step,
        "fixed_index_seconds": fixed,
        "measured_256_token_speedup_including_fixed_cost": (
            total_full_decode / total_sparse_decode
        ),
        "break_even_generated_tokens": fixed / (full_step - sparse_step),
        "actual_attention_tokens_per_head": mean(
            sparse_rows, "actual_attention_tokens_mean"
        ),
        "actual_attention_fraction": mean(
            sparse_rows, "actual_attention_tokens_mean"
        )
        / mean(sparse_rows, "history_tokens"),
        "index_ratio_of_full_fp16_kv": mean(
            sparse_rows, "packed_index_ratio_of_full_kv"
        ),
        "candidate_overflow_rate": mean(
            sparse_rows, "candidate_overflow_rate_mean"
        ),
    }
    if all("top1_agreement" in row for row in sparse_rows):
        system["top1_agreement"] = mean(
            sparse_rows,
            "top1_agreement",
        )
    if all("kl_full_to_sparse_mean" in row for row in sparse_rows):
        system["kl_full_to_sparse"] = mean(
            sparse_rows,
            "kl_full_to_sparse_mean",
        )
    for steps in projection_steps:
        system[f"projected_speedup_{steps}_tokens"] = (
            full_step * steps / (fixed + sparse_step * steps)
        )

    output = {
        "protocol": {
            "sparse_root": str(args.input_root),
            "full_root": str(
                args.full_root
                if args.full_root is not None
                else args.input_root
            ),
            "cases": len(pairs),
            "tokens": sum(int(row["tokens"]) for row in sparse_rows),
            "history_tokens": int(sparse_rows[0]["history_tokens"]),
            "score_mode": sparse_rows[0]["score_mode"],
            "full_kv_residency": (
                "Full FP16 K/V remains GPU-resident for exact sparse "
                "candidate attention. The index ratio is additional logical "
                "K-index storage, not total runtime KV memory."
            ),
        },
        "quality": {
            **bootstrap["point_estimate"],
            "quality_retention_ci95": [
                bootstrap["bootstrap_95_percent"]["quality_retention"][
                    "lower_2p5"
                ],
                bootstrap["bootstrap_95_percent"]["quality_retention"][
                    "upper_97p5"
                ],
            ],
            **bootstrap["bootstrap_probabilities"],
        },
        "system": system,
        "cases": [
            {
                "case": pair["case"],
                "quality_retention": next(
                    float(case["quality_retention"])
                    for case in bootstrap["cases"]
                    if case["case"] == pair["case"]
                ),
                "full_ms_per_token": (
                    pair["full"]["steady_sparse_seconds_per_step"] * 1000.0
                ),
                "sparse_steady_ms_per_token": (
                    pair["sparse"]["steady_sparse_seconds_per_step"] * 1000.0
                ),
                "fixed_index_seconds": pair["sparse"][
                    "fixed_sparse_overhead_seconds"
                ],
            }
            for pair in pairs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
