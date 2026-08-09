from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_critical_position_budget_probe_20260715 import (  # noqa: E402
    fraction_name,
    logit_features,
    parse_fractions,
    token_shape,
)


def test_fraction_parser_and_names() -> None:
    assert parse_fractions("0.02,0.005,0.02") == [0.005, 0.02]
    assert fraction_name(0.0025) == "head_top0p25pct"
    assert fraction_name(0.02) == "head_top2pct"
    assert fraction_name(None) == "full_attention"


def test_token_shape_covers_digits_number_words_and_punctuation() -> None:
    assert token_shape(" nine")["is_numeric_token"] == 1
    assert token_shape(" 42")["is_digit_token"] == 1
    assert token_shape(" and")["is_numeric_token"] == 0
    assert token_shape(".")["is_punctuation_token"] == 1


def test_logit_features_match_cross_entropy() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32)
    features = logit_features(logits, label_id=2)
    expected_nll = float(torch.nn.functional.cross_entropy(logits, torch.tensor([2])).item())
    assert math.isclose(float(features["nll"]), expected_nll, rel_tol=0.0, abs_tol=1e-7)
    assert features["top1_id"] == 1
    assert features["top1_correct"] == 0
    assert math.isclose(float(features["top1_top2_logit_margin"]), 1.0)
