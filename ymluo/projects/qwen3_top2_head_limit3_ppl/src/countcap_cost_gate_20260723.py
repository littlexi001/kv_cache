from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SPARSE_METHODS = ("countcap_fullprompt", "countcap_fullprompt_keypca")


def load_cost_profile(path: str | Path) -> dict[str, Any]:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if not profile.get("lengths"):
        raise ValueError("cost profile contains no measured lengths")
    return profile


def _bracket(
    rows: list[dict[str, Any]], history_tokens: int
) -> tuple[dict[str, Any], dict[str, Any], float]:
    ordered = sorted(rows, key=lambda row: float(row["mean_prompt_tokens"]))
    target = float(history_tokens)
    if target <= float(ordered[0]["mean_prompt_tokens"]):
        return ordered[0], ordered[0], 0.0
    if target >= float(ordered[-1]["mean_prompt_tokens"]):
        return ordered[-1], ordered[-1], 0.0
    for left, right in zip(ordered, ordered[1:]):
        left_tokens = float(left["mean_prompt_tokens"])
        right_tokens = float(right["mean_prompt_tokens"])
        if left_tokens <= target <= right_tokens:
            weight = (
                (math.log(target) - math.log(left_tokens))
                / (math.log(right_tokens) - math.log(left_tokens))
            )
            return left, right, weight
    raise RuntimeError("failed to bracket history length")


def _interpolate(left: float, right: float, weight: float) -> float:
    return (1.0 - weight) * float(left) + weight * float(right)


def _cost_model_at(
    profile: dict[str, Any], history_tokens: int, method: str
) -> dict[str, float]:
    left, right, weight = _bracket(profile["lengths"], history_tokens)
    if method == "full_kv":
        left_model = left["full_decode_cost_model"]
        right_model = right["full_decode_cost_model"]
        retention = 1.0
    else:
        left_metrics = left["methods"][method]
        right_metrics = right["methods"][method]
        left_model = left_metrics["decode_cost_model"]
        right_model = right_metrics["decode_cost_model"]
        # A conservative envelope avoids creating a quality claim between two
        # measurements that is stronger than either endpoint.
        retention = min(
            float(left_metrics["quality_retention"]),
            float(right_metrics["quality_retention"]),
        )
    values = {}
    for key in ("fixed_seconds", "step_seconds"):
        left_value = left_model[key]
        right_value = right_model[key]
        if left_value is None or right_value is None:
            raise ValueError(f"missing {key} for {method}")
        values[key] = _interpolate(left_value, right_value, weight)
    values["quality_retention"] = retention
    return values


def choose_countcap_path(
    profile: dict[str, Any],
    history_tokens: int,
    expected_generated_tokens: int,
    quality_floor: float = 0.95,
    speed_margin: float = 1.03,
) -> dict[str, Any]:
    if history_tokens <= 0:
        raise ValueError("history_tokens must be positive")
    if expected_generated_tokens < 0:
        raise ValueError("expected_generated_tokens must be non-negative")
    if not 0.0 < quality_floor <= 1.0:
        raise ValueError("quality_floor must be in (0, 1]")
    if speed_margin < 1.0:
        raise ValueError("speed_margin must be at least one")

    decode_forwards = max(0, expected_generated_tokens - 1)
    dense = _cost_model_at(profile, history_tokens, "full_kv")
    dense_seconds = dense["fixed_seconds"] + decode_forwards * dense["step_seconds"]
    candidates = []
    diagnostics = {}
    available_methods = set.intersection(
        *(
            set(row.get("methods", {}))
            for row in profile["lengths"]
        )
    )
    for method in SPARSE_METHODS:
        if method not in available_methods:
            diagnostics[method] = {
                "available": False,
                "eligible": False,
            }
            continue
        sparse = _cost_model_at(profile, history_tokens, method)
        sparse_seconds = (
            sparse["fixed_seconds"] + decode_forwards * sparse["step_seconds"]
        )
        predicted_speedup = (
            dense_seconds / sparse_seconds if sparse_seconds > 0.0 else 0.0
        )
        eligible = (
            sparse["quality_retention"] >= quality_floor
            and sparse["step_seconds"] < dense["step_seconds"]
            and predicted_speedup >= speed_margin
        )
        diagnostics[method] = {
            **sparse,
            "available": True,
            "predicted_decode_seconds": sparse_seconds,
            "predicted_decode_speedup": predicted_speedup,
            "eligible": eligible,
        }
        if eligible:
            candidates.append((predicted_speedup, method))

    selected_speedup, selected = max(candidates, default=(1.0, "full_kv"))
    return {
        "selected_path": selected,
        "history_tokens": int(history_tokens),
        "expected_generated_tokens": int(expected_generated_tokens),
        "predicted_dense_decode_seconds": dense_seconds,
        "predicted_decode_speedup": selected_speedup,
        "quality_floor": quality_floor,
        "speed_margin": speed_margin,
        "candidates": diagnostics,
    }
