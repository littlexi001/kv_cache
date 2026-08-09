from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


ROLE_ORDER = ("start_key", "hop1_result", "hop2_input", "hop2_result")
FACTORS = ("query_content", "key_content", "relative_position")


def rotate_half(values: np.ndarray) -> np.ndarray:
    half = values.shape[-1] // 2
    return np.concatenate((-values[..., half:], values[..., :half]), axis=-1)


def apply_rope(values: np.ndarray, cosine: np.ndarray, sine: np.ndarray) -> np.ndarray:
    return values * cosine + rotate_half(values) * sine


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-30)


def load_case(root: Path, placement: str, length: int) -> dict[str, Any]:
    directory = root / placement
    metadata_path = directory / f"{placement}_length_{length}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(directory / metadata["vector_file"]) as payload:
        arrays = {name: payload[name].astype(np.float32) for name in payload.files}
    return {"metadata": metadata, "arrays": arrays}


def expand_keys(keys: np.ndarray, query_heads: int) -> np.ndarray:
    groups = query_heads // keys.shape[-2]
    return np.repeat(keys, groups, axis=-2)


def counterfactual(
    query_case: dict[str, Any],
    key_case: dict[str, Any],
    position_case: dict[str, Any],
    role_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    query = query_case["arrays"]["q_pre"]
    key = key_case["arrays"]["k_pre"][role_index]
    query = apply_rope(
        query,
        position_case["arrays"]["query_cos"],
        position_case["arrays"]["query_sin"],
    )
    key = apply_rope(
        key,
        position_case["arrays"]["key_cos"][role_index],
        position_case["arrays"]["key_sin"][role_index],
    )
    key = expand_keys(key, query.shape[-2])
    logits = np.sum(query * key, axis=-1) / math.sqrt(query.shape[-1])
    return logits, cosine(query, key)


def shapley_decomposition(
    left: dict[str, Any],
    right: dict[str, Any],
    role_index: int,
) -> dict[str, Any]:
    corners: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for query_source, key_source, position_source in product((0, 1), repeat=3):
        corners[query_source, key_source, position_source] = counterfactual(
            right if query_source else left,
            right if key_source else left,
            right if position_source else left,
            role_index,
        )
    contributions: dict[str, dict[str, Any]] = {}
    for factor_index, factor in enumerate(FACTORS):
        logit_contribution = np.zeros_like(corners[0, 0, 0][0])
        cosine_contribution = np.zeros_like(corners[0, 0, 0][1])
        other_factors = [index for index in range(3) if index != factor_index]
        for subset_bits in product((0, 1), repeat=2):
            subset_size = sum(subset_bits)
            weight = math.factorial(subset_size) * math.factorial(2 - subset_size) / math.factorial(3)
            lower = [0, 0, 0]
            for other_index, bit in zip(other_factors, subset_bits):
                lower[other_index] = bit
            upper = list(lower)
            upper[factor_index] = 1
            lower_values = corners[tuple(lower)]
            upper_values = corners[tuple(upper)]
            logit_contribution += weight * (upper_values[0] - lower_values[0])
            cosine_contribution += weight * (upper_values[1] - lower_values[1])
        contributions[factor] = {
            "mean_logit_contribution": float(logit_contribution.mean()),
            "median_logit_contribution": float(np.median(logit_contribution)),
            "fraction_negative_logit_contribution": float((logit_contribution < 0).mean()),
            "mean_cosine_contribution": float(cosine_contribution.mean()),
            "median_cosine_contribution": float(np.median(cosine_contribution)),
            "fraction_negative_cosine_contribution": float((cosine_contribution < 0).mean()),
            "per_layer_mean_logit_contribution": logit_contribution.mean(axis=1).tolist(),
            "per_layer_mean_cosine_contribution": cosine_contribution.mean(axis=1).tolist(),
        }
    start_logit, start_cosine = corners[0, 0, 0]
    end_logit, end_cosine = corners[1, 1, 1]
    return {
        "start_mean_logit": float(start_logit.mean()),
        "end_mean_logit": float(end_logit.mean()),
        "delta_mean_logit": float((end_logit - start_logit).mean()),
        "start_mean_cosine": float(start_cosine.mean()),
        "end_mean_cosine": float(end_cosine.mean()),
        "delta_mean_cosine": float((end_cosine - start_cosine).mean()),
        "contributions": contributions,
    }


def vector_stability(left: dict[str, Any], right: dict[str, Any], role_index: int) -> dict[str, float]:
    left_query = left["arrays"]["q_pre"]
    right_query = right["arrays"]["q_pre"]
    left_key = left["arrays"]["k_pre"][role_index]
    right_key = right["arrays"]["k_pre"][role_index]
    query_cosine = cosine(left_query, right_query)
    key_cosine = cosine(left_key, right_key)
    query_relative_l2 = np.linalg.norm(right_query - left_query, axis=-1) / np.maximum(
        np.linalg.norm(left_query, axis=-1), 1e-30
    )
    key_relative_l2 = np.linalg.norm(right_key - left_key, axis=-1) / np.maximum(
        np.linalg.norm(left_key, axis=-1), 1e-30
    )
    return {
        "mean_pre_rope_query_cosine": float(query_cosine.mean()),
        "median_pre_rope_query_cosine": float(np.median(query_cosine)),
        "mean_pre_rope_query_relative_l2": float(query_relative_l2.mean()),
        "mean_pre_rope_key_cosine": float(key_cosine.mean()),
        "median_pre_rope_key_cosine": float(np.median(key_cosine)),
        "mean_pre_rope_key_relative_l2": float(key_relative_l2.mean()),
        "max_pre_rope_key_absolute_difference": float(np.abs(right_key - left_key).max()),
    }


def validate_reconstruction(case: dict[str, Any], role_index: int) -> dict[str, float]:
    reconstructed_logit, reconstructed_cosine = counterfactual(case, case, case, role_index)
    stored_logit = case["arrays"]["post_logits"][role_index]
    stored_cosine = case["arrays"]["post_cosines"][role_index]
    return {
        "max_abs_logit_error": float(np.abs(reconstructed_logit - stored_logit).max()),
        "mean_abs_logit_error": float(np.abs(reconstructed_logit - stored_logit).mean()),
        "max_abs_cosine_error": float(np.abs(reconstructed_cosine - stored_cosine).max()),
        "mean_abs_cosine_error": float(np.abs(reconstructed_cosine - stored_cosine).mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shapley-decompose Q/K direction changes into Q, K, and RoPE position factors.")
    parser.add_argument("--probe_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_root = Path(args.probe_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        (placement, length): load_case(probe_root, placement, length)
        for placement in ("prefix", "middle", "recent")
        for length in (8000, 128000)
    }
    role_index = ROLE_ORDER.index("hop2_result")
    comparisons = [
        ("prefix_8k_to_128k", ("prefix", 8000), ("prefix", 128000)),
        ("middle_8k_to_128k", ("middle", 8000), ("middle", 128000)),
        ("recent_8k_to_128k", ("recent", 8000), ("recent", 128000)),
        ("128k_prefix_to_middle", ("prefix", 128000), ("middle", 128000)),
        ("128k_middle_to_recent", ("middle", 128000), ("recent", 128000)),
        ("128k_prefix_to_recent", ("prefix", 128000), ("recent", 128000)),
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": "Qwen3-8B",
        "role": "hop2_result",
        "method": (
            "Exact algebraic 2^3 counterfactual over saved pre-RoPE query content, "
            "pre-RoPE evidence-key content, and the query/key RoPE positions; Shapley values "
            "allocate the endpoint change including interactions."
        ),
        "cases": {
            f"{placement}_{length}": {
                **case["metadata"],
                "reconstruction": validate_reconstruction(case, role_index),
            }
            for (placement, length), case in cases.items()
        },
        "comparisons": {},
    }
    flat_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for label, left_key, right_key in comparisons:
        left = cases[left_key]
        right = cases[right_key]
        decomposition = shapley_decomposition(left, right, role_index)
        stability = vector_stability(left, right, role_index)
        record = {
            "left": f"{left_key[0]}_{left_key[1]}",
            "right": f"{right_key[0]}_{right_key[1]}",
            "left_distance": left["metadata"]["relative_distances"]["hop2_result"],
            "right_distance": right["metadata"]["relative_distances"]["hop2_result"],
            "left_gold_ppl": left["metadata"]["gold_ppl"],
            "right_gold_ppl": right["metadata"]["gold_ppl"],
            "vector_stability": stability,
            **decomposition,
        }
        summary["comparisons"][label] = record
        row: dict[str, Any] = {
            "comparison": label,
            "left": record["left"],
            "right": record["right"],
            "left_distance": record["left_distance"],
            "right_distance": record["right_distance"],
            "left_gold_ppl": record["left_gold_ppl"],
            "right_gold_ppl": record["right_gold_ppl"],
            "delta_mean_logit": record["delta_mean_logit"],
            "delta_mean_cosine": record["delta_mean_cosine"],
            **stability,
        }
        for factor in FACTORS:
            contribution = decomposition["contributions"][factor]
            row[f"{factor}_logit_contribution"] = contribution["mean_logit_contribution"]
            row[f"{factor}_cosine_contribution"] = contribution["mean_cosine_contribution"]
            for layer_index, (logit_value, cosine_value) in enumerate(
                zip(
                    contribution["per_layer_mean_logit_contribution"],
                    contribution["per_layer_mean_cosine_contribution"],
                )
            ):
                layer_rows.append(
                    {
                        "comparison": label,
                        "factor": factor,
                        "layer": layer_index,
                        "mean_logit_contribution": logit_value,
                        "mean_cosine_contribution": cosine_value,
                    }
                )
        flat_rows.append(row)
    (output_dir / "qk_direction_decomposition.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "qk_direction_comparisons.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    with (output_dir / "qk_direction_layer_contributions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    print(json.dumps({"output_dir": str(output_dir), "comparisons": flat_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
