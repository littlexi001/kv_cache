from __future__ import annotations

import json

import numpy as np
import pytest

from collect_onpolicy_counterfactual_router_20260715 import minimal_safe_action
from train_onpolicy_counterfactual_router_20260715 import (
    calibrate_teacher_recall,
    load_counterfactual_rows,
)

from train_shifted_physical_budget_router_20260715 import (
    FEATURE_NAMES,
    calibrate_threshold,
    shifted_feature_vector,
)
from train_shifted_three_action_router_20260715 import calibrate_two_thresholds


def make_features() -> dict[str, float | int]:
    return {
        "top1_probability": 0.7,
        "top1_top2_logit_margin": 2.0,
        "entropy": 1.2,
        "normalized_entropy": 0.1,
        "retrieval_feature_valid": 1.0,
        "retrieval_score_spread": 0.4,
        "retrieval_candidate_stability": 0.75,
        "retrieval_refreshed_fraction": 1.0,
        "top1_history_frequency": 4,
        "top1_is_digit_token": 0,
        "top1_is_number_word": 0,
        "top1_is_numeric_token": 0,
        "top1_is_alpha_token": 1,
        "top1_is_punctuation_token": 0,
        "input_history_frequency": 8,
        "input_is_digit_token": 1,
        "input_is_number_word": 0,
        "input_is_numeric_token": 1,
        "input_is_alpha_token": 0,
        "input_is_punctuation_token": 0,
        "history_tokens": 128000,
    }


def test_shifted_feature_vector_matches_frozen_schema() -> None:
    features = shifted_feature_vector(
        make_features(), make_features(), target_index=2, target_count=10
    )

    assert len(features) == len(FEATURE_NAMES)
    assert features[FEATURE_NAMES.index("previous_retrieval_score_spread")] == 0.4
    assert (
        features[FEATURE_NAMES.index("previous_retrieval_candidate_stability")]
        == 0.75
    )
    assert features[-2] == 2 / 9


def test_calibration_selects_only_the_useful_high_action() -> None:
    predictions = np.asarray([0.9, 0.1])
    metadata = [
        {"low_nll": 2.0, "high_nll": 1.0, "full_nll": 1.0},
        {"low_nll": 1.0, "high_nll": 1.0, "full_nll": 1.0},
    ]

    threshold, result = calibrate_threshold(
        predictions, metadata, target_retention=0.95
    )

    assert threshold == 0.9
    assert result["high_positions"] == 1


def test_three_action_calibration_uses_high_only_for_top_risk() -> None:
    predictions = np.asarray([0.9, 0.5, 0.1])
    metadata = [
        {"low_nll": 2.0, "mid_nll": 1.5, "high_nll": 1.0, "full_nll": 1.0},
        {"low_nll": 1.0, "mid_nll": 1.0, "high_nll": 1.0, "full_nll": 1.0},
        {"low_nll": 1.0, "mid_nll": 1.0, "high_nll": 1.0, "full_nll": 1.0},
    ]

    mid_threshold, high_threshold, result = calibrate_two_thresholds(
        predictions,
        metadata,
        target_retention=0.95,
        mid_cost=1.0,
        high_cost=3.0,
    )

    assert mid_threshold == 0.9
    assert high_threshold == 0.9
    assert result["high_count"] == 1


def test_minimal_safe_action_uses_cheapest_action_close_to_high() -> None:
    assert minimal_safe_action([1.02, 0.98, 1.0], tolerance=0.05) == 0
    assert minimal_safe_action([1.20, 1.03, 1.0], tolerance=0.05) == 1
    assert minimal_safe_action([1.20, 1.10, 1.0], tolerance=0.05) == 2


def test_minimal_safe_action_can_use_full_kv_reference() -> None:
    assert minimal_safe_action(
        [1.02, 0.98, 1.0], tolerance=0.05, reference_nll=0.8
    ) == 2
    assert minimal_safe_action(
        [0.82, 0.90, 1.0], tolerance=0.05, reference_nll=0.8
    ) == 0


def test_counterfactual_rows_train_on_low_to_high_gain(tmp_path) -> None:
    path = tmp_path / "counterfactual.json"
    path.write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "action_fractions": [0.01, 0.015, 0.025],
                "action_stream_group_sizes": [2, 2, 1],
                "feature_vectors": [[0.0] * len(FEATURE_NAMES)],
                "counterfactual_nll": [[1.4, 1.2, 1.0]],
                "teacher_actions": [2],
            }
        ),
        encoding="utf-8",
    )

    features, targets, metadata, config = load_counterfactual_rows(
        [path], "low_high_gain"
    )

    assert features.shape == (1, len(FEATURE_NAMES))
    assert targets.tolist() == pytest.approx([0.4])
    assert metadata[0]["high_nll"] == pytest.approx(1.0)
    assert config["action_fractions"] == [0.01, 0.015, 0.025]
    with pytest.raises(ValueError, match="not FullKV-referenced"):
        load_counterfactual_rows(
            [path], "teacher_action", require_full_reference_labels=True
        )


def test_teacher_recall_calibration_protects_required_high_actions() -> None:
    predictions = np.asarray([0.0, 0.2, 0.8, 1.0])
    metadata = [
        {
            "teacher_action": action,
            "low_nll": 1.0 + 0.1 * action,
            "mid_nll": 1.0 + 0.05 * action,
            "high_nll": 1.0,
            "full_nll": 1.0,
        }
        for action in [0, 1, 2, 2]
    ]

    _, _, result = calibrate_teacher_recall(
        predictions,
        metadata,
        minimum_recall=1.0,
        mid_cost=1.0,
        high_cost=3.0,
    )

    assert result["required_mid_recall"] == pytest.approx(1.0)
    assert result["required_high_recall"] == pytest.approx(1.0)
    assert result["high_count"] == 2
    assert "quality_retention" in result
