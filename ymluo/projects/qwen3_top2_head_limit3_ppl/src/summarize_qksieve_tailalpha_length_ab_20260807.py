from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha1_root", type=Path, required=True)
    parser.add_argument("--alpha0_root", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def method_row(payload: dict[str, Any], sparse: bool) -> dict[str, Any]:
    rows = [
        row
        for row in payload["rows"]
        if (row["method"] != "full_attention") == sparse
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one {'sparse' if sparse else 'full'} row")
    return rows[0]


def summarize(alpha1_root: Path, alpha0_root: Path) -> dict[str, Any]:
    lengths: dict[str, Any] = {}
    for history_tokens in (32768, 65536, 131072):
        alpha1 = load(alpha1_root / f"n{history_tokens}" / "summary.json")
        alpha0 = load(alpha0_root / f"n{history_tokens}" / "summary.json")
        if alpha1["target_token_ids_sha256"] != alpha0["target_token_ids_sha256"]:
            raise ValueError(f"target tokens differ at {history_tokens}")
        full1 = method_row(alpha1, sparse=False)
        full0 = method_row(alpha0, sparse=False)
        sparse1 = method_row(alpha1, sparse=True)
        sparse0 = method_row(alpha0, sparse=True)
        if abs(float(full1["nll"]) - float(full0["nll"])) > 1e-8:
            raise ValueError(f"Full NLL differs at {history_tokens}")
        if abs(float(sparse1["packed_value_sketch_tail_alpha"]) - 1.0) > 1e-8:
            raise ValueError(f"alpha=1 contract failed at {history_tokens}")
        if abs(float(sparse0["packed_value_sketch_tail_alpha"])) > 1e-8:
            raise ValueError(f"alpha=0 contract failed at {history_tokens}")
        full_nll = float(full0["nll"])
        alpha1_nll = float(sparse1["nll"])
        alpha0_nll = float(sparse0["nll"])
        lengths[str(history_tokens)] = {
            "full_nll": full_nll,
            "alpha1_nll": alpha1_nll,
            "alpha0_nll": alpha0_nll,
            "alpha1_quality_retention": math.exp(full_nll - alpha1_nll),
            "alpha0_quality_retention": math.exp(full_nll - alpha0_nll),
            "alpha0_minus_alpha1_nll": alpha0_nll - alpha1_nll,
            "attention_tokens": float(sparse0["actual_attention_tokens_mean"]),
            "attention_fraction": float(sparse0["actual_attention_fraction_mean"]),
            "alpha0_steady_seconds_per_token": float(
                sparse0["steady_sparse_seconds_per_step"]
            ),
            "alpha0_fixed_seconds": float(
                sparse0["fixed_sparse_overhead_seconds"]
            ),
        }
    return {
        "schema": "qksieve_tailalpha_length_ab_v1",
        "lengths": lengths,
        "claim_boundary": (
            "The A/B changes only the Value-tail correction coefficient. "
            "It diagnoses quality; alpha=1 timing was collected with stage "
            "events and is therefore not compared against alpha=0 timing."
        ),
    }


def main() -> None:
    args = parse_args()
    payload = summarize(args.alpha1_root, args.alpha0_root)
    output = args.alpha0_root / "tailalpha_length_ab_summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
