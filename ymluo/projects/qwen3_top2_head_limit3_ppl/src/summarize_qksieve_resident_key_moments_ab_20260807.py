from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MODES = ("off", "moments", "factors")


def sparse_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload["rows"] if row["method"] != "full_attention"]
    if len(rows) != 1:
        raise AssertionError(f"expected one sparse row, found {len(rows)}")
    return rows[0]


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else math.inf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    payloads = {
        mode: json.loads(
            (args.run_root / mode / "quality" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for mode in MODES
    }
    rows = {mode: sparse_row(payloads[mode]) for mode in MODES}
    target_hashes = {
        payloads[mode]["target_token_ids_sha256"] for mode in MODES
    }
    if len(target_hashes) != 1:
        raise AssertionError("target-token hashes differ across modes")

    contract_fields = (
        "score_mode",
        "max_exact_tokens_per_head",
        "requested_quantile_sample_count_per_head",
        "packed_mean_bits_by_band",
        "packed_active_fraction_by_band",
        "actual_attention_tokens_mean",
        "actual_attention_tokens_min",
        "actual_attention_tokens_max",
        "candidate_overflow_count_max",
    )
    for field in contract_fields:
        reference = rows["off"][field]
        for mode in MODES[1:]:
            if rows[mode][field] != reference:
                raise AssertionError(f"{field} differs for {mode}")

    qk_stats = {mode: rows[mode]["packed_parallel_qk_prebuild"] for mode in MODES}
    layers = int(qk_stats["off"]["layers"])
    expected_hits = {
        "off": (0, 0),
        "moments": (layers, 0),
        "factors": (layers, layers),
    }
    for mode, (moment_hits, factor_hits) in expected_hits.items():
        actual = (
            int(qk_stats[mode].get("resident_key_moment_hits", 0)),
            int(qk_stats[mode].get("resident_key_hits", 0)),
        )
        if actual != (moment_hits, factor_hits):
            raise AssertionError(
                f"unexpected resident hits for {mode}: {actual}"
            )

    fixed = {
        mode: float(rows[mode]["fixed_sparse_overhead_seconds"])
        for mode in MODES
    }
    resident_build = {
        mode: float(
            payloads[mode].get("resident_key_factor_precompute", {}).get(
                "total_seconds", 0.0
            )
        )
        for mode in MODES
    }
    off_nll = float(rows["off"]["nll"])
    nll_delta = {
        mode: float(rows[mode]["nll"]) - off_nll for mode in MODES
    }
    moment_numerically_identical = abs(nll_delta["moments"]) <= 1e-9
    moment_fixed_speedup = ratio(fixed["off"], fixed["moments"])
    passed = bool(
        moment_numerically_identical
        and moment_fixed_speedup >= 1.05
        and all(
            rows[mode][field] == rows["off"][field]
            for mode in ("moments",)
            for field in contract_fields
        )
    )

    output = {
        "schema": "qksieve_resident_key_moments_ab_v1",
        "history_tokens": int(payloads["off"]["history_tokens"]),
        "eval_tokens": int(payloads["off"]["eval_tokens"]),
        "layers": layers,
        "target_token_ids_sha256": next(iter(target_hashes)),
        "modes": {
            mode: {
                "nll": float(rows[mode]["nll"]),
                "ppl": float(rows[mode]["ppl"]),
                "nll_delta_vs_off": nll_delta[mode],
                "fixed_seconds": fixed[mode],
                "resident_build_seconds": resident_build[mode],
                "steady_seconds_per_token": float(
                    rows[mode]["steady_sparse_seconds_per_step"]
                ),
                "qk_prebuild_seconds": float(qk_stats[mode]["total_seconds"]),
                "resident_moment_hits": int(
                    qk_stats[mode].get("resident_key_moment_hits", 0)
                ),
                "resident_factor_hits": int(
                    qk_stats[mode].get("resident_key_hits", 0)
                ),
            }
            for mode in MODES
        },
        "moments_only": {
            "numerically_identical_nll": moment_numerically_identical,
            "fixed_speedup": moment_fixed_speedup,
            "fixed_seconds_saved": fixed["off"] - fixed["moments"],
            "first_request_speedup": ratio(
                fixed["off"], fixed["moments"] + resident_build["moments"]
            ),
            "passed_screen": passed,
        },
        "factors_speed_upper_bound": {
            "fixed_speedup": ratio(fixed["off"], fixed["factors"]),
            "nll_delta_vs_off": nll_delta["factors"],
        },
        "claim_boundary": (
            "This screen checks identical task inputs, allocation/candidate-count "
            "metadata, target NLL, and fixed latency. Active packed-index hashes "
            "remain a separate promotion gate."
        ),
    }
    (args.run_root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
