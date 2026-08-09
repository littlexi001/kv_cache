from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from critical_position_router import (  # noqa: E402
    ATTENTION_MASS_FEATURES,
    FEATURE_NAMES,
    build_feature_vector,
    text_shape_values,
)


def test_text_shape_values() -> None:
    assert text_shape_values(" nine") == [0.0, 1.0, 0.0, 4.0]
    assert text_shape_values(" 42") == [1.0, 0.0, 0.0, 2.0]
    assert text_shape_values(".") == [0.0, 0.0, 1.0, 1.0]


def test_feature_vector_matches_frozen_schema() -> None:
    vector = build_feature_vector(
        logit_features={
            "top1_probability": 0.8,
            "top1_top2_logit_margin": 2.0,
            "entropy": 1.5,
        },
        attention_features={name: 0.9 for name in ATTENTION_MASS_FEATURES},
        top1_text=" and",
        top1_history_frequency=100,
        prediction_index=128,
        prediction_horizon=256,
        topic="sports",
        input_text=" nine",
        input_history_frequency=2,
    )
    assert len(vector) == len(FEATURE_NAMES)
    assert vector[5] == 1.0
