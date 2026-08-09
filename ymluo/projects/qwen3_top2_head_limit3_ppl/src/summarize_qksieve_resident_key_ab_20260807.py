from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit query-independent resident Key-factor reuse."
    )
    parser.add_argument("--run_root", required=True, type=Path)
    return parser.parse_args()


def sparse_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload["rows"] if row["method"] != "full_attention"]
    if len(rows) != 1:
        raise AssertionError(f"expected one sparse row, found {len(rows)}")
    return rows[0]


def close(left: float, right: float, atol: float = 1e-9) -> bool:
    return abs(left - right) <= atol


def summarize(run_root: Path) -> dict[str, Any]:
    payloads = {
        mode: json.loads(
            (run_root / mode / "quality" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for mode in ("off", "on")
    }
    rows = {mode: sparse_row(payload) for mode, payload in payloads.items()}
    if (
        payloads["off"]["target_token_ids_sha256"]
        != payloads["on"]["target_token_ids_sha256"]
    ):
        raise AssertionError("A/B target tokens differ")
    for field in (
        "score_mode",
        "max_exact_tokens_per_head",
        "requested_quantile_sample_count_per_head",
        "packed_mean_bits_by_band",
        "packed_active_fraction_by_band",
    ):
        if rows["off"][field] != rows["on"][field]:
            raise AssertionError(f"numerical contract differs: {field}")
    quality_fields = ("nll", "ppl", "target_nll_delta_mean")
    numerically_identical = all(
        close(float(rows["off"][field]), float(rows["on"][field]))
        for field in quality_fields
    )

    off_qk = rows["off"]["packed_parallel_qk_prebuild"]
    on_qk = rows["on"]["packed_parallel_qk_prebuild"]
    layers = int(on_qk["layers"])
    if int(off_qk["resident_key_hits"]) != 0:
        raise AssertionError("resident-off run unexpectedly reused Key factors")
    if int(on_qk["resident_key_hits"]) != layers:
        raise AssertionError("resident-on run did not reuse every layer")

    resident_build = payloads["on"]["resident_key_factor_precompute"]
    resident_seconds = float(resident_build["total_seconds"])
    off_fixed = float(rows["off"]["fixed_sparse_overhead_seconds"])
    on_fixed = float(rows["on"]["fixed_sparse_overhead_seconds"])
    off_qk_seconds = float(off_qk["total_seconds"])
    on_qk_seconds = float(on_qk["total_seconds"])
    saved_per_request = off_fixed - on_fixed
    break_even_requests = (
        math.floor(resident_seconds / saved_per_request) + 1
        if saved_per_request > 0
        else None
    )
    return {
        "schema": "qksieve_resident_key_ab_v1",
        "history_tokens": int(payloads["off"]["history_tokens"]),
        "eval_tokens": int(payloads["off"]["eval_tokens"]),
        "numerically_identical": numerically_identical,
        "bit_allocation_identical": True,
        "target_token_ids_sha256": payloads["off"][
            "target_token_ids_sha256"
        ],
        "quality": {
            mode: {
                "nll": float(row["nll"]),
                "ppl": float(row["ppl"]),
                "quality_retention": float(row["quality_retention"]),
            }
            for mode, row in rows.items()
        },
        "quality_delta_on_minus_off": {
            field: float(rows["on"][field]) - float(rows["off"][field])
            for field in quality_fields
        },
        "resident_key_precompute_seconds": resident_seconds,
        "request_local_qk_prebuild_seconds": {
            "off": off_qk_seconds,
            "on": on_qk_seconds,
            "speedup": off_qk_seconds / on_qk_seconds,
        },
        "subsequent_request_fixed_seconds": {
            "off": off_fixed,
            "on": on_fixed,
            "speedup": off_fixed / on_fixed,
        },
        "first_request_fixed_seconds": {
            "off": off_fixed,
            "on_including_resident_build": on_fixed + resident_seconds,
            "speedup": off_fixed / (on_fixed + resident_seconds),
        },
        "steady_seconds_per_token": {
            mode: float(row["steady_sparse_seconds_per_step"])
            for mode, row in rows.items()
        },
        "resident_hits": {
            "off": int(off_qk["resident_key_hits"]),
            "on": int(on_qk["resident_key_hits"]),
            "layers": layers,
        },
        "multi_request_break_even_requests": break_even_requests,
        "claim_boundary": (
            "The resident and legacy factor solvers are algebraically "
            "equivalent but not necessarily bitwise identical. The resident "
            "path cannot be called lossless until held-out quality is checked. "
            "Its one-time build may be amortized only when one KV cache is reused."
        ),
    }


def main() -> None:
    args = parse_args()
    payload = summarize(args.run_root)
    output = args.run_root / "summary.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
