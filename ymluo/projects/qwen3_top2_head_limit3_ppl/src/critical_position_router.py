from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any


ATTENTION_MASS_FEATURES = [
    "retained_mass_mean",
    "retained_mass_min",
    "retained_mass_p10",
    "retained_mass_p25",
    "retained_mass_lt90_fraction",
    "retained_mass_lt95_fraction",
    "retained_mass_lt99_fraction",
    "retained_mass_early_mean",
    "retained_mass_middle_mean",
    "retained_mass_late_mean",
    "top1_attention_mass_mean",
]

FEATURE_NAMES = [
    "top1_probability",
    "top1_top2_logit_margin",
    "entropy",
    "log1p_top1_history_frequency",
    "relative_prediction_position",
    "is_sports",
    "top1_has_digit",
    "top1_has_alpha",
    "top1_is_punctuation",
    "top1_text_length",
    *ATTENTION_MASS_FEATURES,
    "log1p_input_history_frequency",
    "input_has_digit",
    "input_has_alpha",
    "input_is_punctuation",
    "input_text_length",
]


def text_shape_values(text: str) -> list[float]:
    stripped = text.replace("\\n", "\n").replace("\\r", "\r").strip()
    return [
        float(any(char.isdigit() for char in stripped)),
        float(any(char.isalpha() for char in stripped)),
        float(bool(stripped) and not any(char.isalnum() for char in stripped)),
        float(len(stripped)),
    ]


def build_feature_vector(
    *,
    logit_features: dict[str, float | int],
    attention_features: dict[str, float],
    top1_text: str,
    top1_history_frequency: int,
    prediction_index: int,
    prediction_horizon: int,
    topic: str,
    input_text: str,
    input_history_frequency: int,
) -> list[float]:
    denominator = max(1, int(prediction_horizon) - 1)
    vector = [
        float(logit_features["top1_probability"]),
        float(logit_features["top1_top2_logit_margin"]),
        float(logit_features["entropy"]),
        math.log1p(max(0, int(top1_history_frequency))),
        min(1.0, max(0.0, float(prediction_index) / denominator)),
        float(topic == "sports"),
        *text_shape_values(top1_text),
        *[float(attention_features[name]) for name in ATTENTION_MASS_FEATURES],
        math.log1p(max(0, int(input_history_frequency))),
        *text_shape_values(input_text),
    ]
    if len(vector) != len(FEATURE_NAMES):
        raise AssertionError(f"feature length {len(vector)} != {len(FEATURE_NAMES)}")
    return vector


def load_router_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    if list(artifact.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("router artifact feature schema does not match runtime schema")
    return artifact


def predict_risk(artifact: dict[str, Any], features: list[float]) -> float:
    return float(artifact["model"].predict([features])[0])
