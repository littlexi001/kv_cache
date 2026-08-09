"""Summarize directly measured Full and QKSieve model-forward latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def summarize(run_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("n*/summary.json")):
        if not (path.parent / "ALL_COMPLETE").exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_variant = {row["variant"]: row for row in payload["rows"]}
        full = by_variant.get("full_attention")
        sparse_rows = [
            row
            for variant, row in by_variant.items()
            if variant != "full_attention"
        ]
        if full is None or len(sparse_rows) != 1:
            raise ValueError(f"expected one Full/sparse pair in {path}")
        sparse = sparse_rows[0]
        if int(full["tokens"]) != int(sparse["tokens"]):
            raise ValueError(f"token mismatch in {path}")
        full_decode = float(full["sparse_decode_seconds"])
        sparse_decode = float(sparse["sparse_decode_seconds"])
        full_prefill = float(full["dense_prompt_seconds"])
        sparse_prefill = float(sparse["dense_prompt_seconds"])
        full_steady = float(full["steady_sparse_seconds_per_step"])
        sparse_steady = float(sparse["steady_sparse_seconds_per_step"])
        rows.append(
            {
                "history_tokens": int(payload["history_tokens"]),
                "evaluated_tokens": int(full["tokens"]),
                "method": sparse["variant"],
                "full_ppl": float(full["ppl"]),
                "sparse_ppl": float(sparse["ppl"]),
                "quality_retention": float(sparse["quality_retention"]),
                "attention_tokens_per_head": float(
                    sparse["actual_attention_tokens_mean"]
                ),
                "attention_token_ratio": float(
                    sparse["actual_attention_tokens_mean"]
                )
                / int(payload["history_tokens"]),
                "index_ratio_of_full_kv": float(
                    sparse["packed_index_ratio_of_full_kv"]
                ),
                "full_steady_ms_per_forward": 1000.0 * full_steady,
                "sparse_steady_ms_per_forward": 1000.0 * sparse_steady,
                "steady_decode_speedup_direct": ratio(
                    full_steady, sparse_steady
                ),
                "full_measured_decode_seconds": full_decode,
                "sparse_measured_decode_seconds": sparse_decode,
                "measured_decode_horizon_speedup": ratio(
                    full_decode, sparse_decode
                ),
                "full_measured_request_seconds": full_prefill + full_decode,
                "sparse_measured_request_seconds": (
                    sparse_prefill + sparse_decode
                ),
                "measured_request_speedup": ratio(
                    full_prefill + full_decode,
                    sparse_prefill + sparse_decode,
                ),
            }
        )
    if not rows:
        raise ValueError("no complete direct-decode cases found")
    rows.sort(key=lambda row: row["history_tokens"])
    return {
        "schema": "qksieve_direct_whole_model_decode_length_v1",
        "run_root": str(run_root),
        "timing_contract": {
            "steady_decode": "direct wall time around complete model forwards",
            "measured_decode_horizon": (
                "direct elapsed decode time including method initialization"
            ),
            "measured_request": "direct prefill plus direct decode elapsed time",
            "latency_decomposition_used": False,
        },
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    report = summarize(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
