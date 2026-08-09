import math

import pytest

from run_sample_calibrated_longbench_20260717 import (
    countcap_config,
    countcap_direct_budget,
)


def test_countcap_uses_two_percent_below_cap():
    config = countcap_config(32_000)

    assert config["attention_tokens"] == 640
    assert math.isclose(config["budget_fractions"][0], 0.02)
    assert math.isclose(config["candidate_fraction"], 0.06)
    assert config["projection_dim"] == 48


def test_countcap_caps_attention_and_reduces_candidate_fraction():
    config = countcap_config(128_000)

    assert config["attention_tokens"] == 1280
    assert math.isclose(config["budget_fractions"][0], 0.01)
    assert math.isclose(config["candidate_fraction"], 0.04)


def test_countcap_candidate_fraction_has_three_percent_floor():
    config = countcap_config(256_000)

    assert config["attention_tokens"] == 1280
    assert math.isclose(config["budget_fractions"][0], 0.005)
    assert math.isclose(config["candidate_fraction"], 0.03)


@pytest.mark.parametrize(
    ("history_tokens", "expected_tokens", "expected_fraction"),
    (
        (2_000, 256, 0.128),
        (4_000, 256, 0.064),
        (8_000, 480, 0.06),
        (16_000, 960, 0.06),
        (24_000, 1_280, 1_280 / 24_000),
        (32_000, 1_280, 0.04),
        (64_000, 1_280, 0.02),
        (128_000, 1_280, 0.01),
    ),
)
def test_frozen_direct_budget(
    history_tokens,
    expected_tokens,
    expected_fraction,
):
    tokens, fraction = countcap_direct_budget(history_tokens)
    assert tokens == expected_tokens
    assert math.isclose(fraction, expected_fraction)


def test_frozen_direct_budget_rejects_empty_history():
    with pytest.raises(ValueError, match="history_tokens"):
        countcap_direct_budget(0)
