from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from count_capped_support_policy import count_capped_support


def test_policy_matches_verified_length_points() -> None:
    support32 = count_capped_support(32_000)
    support64 = count_capped_support(64_000)
    support128 = count_capped_support(128_000)

    assert support32.final_fraction == 0.02
    assert support32.candidate_fraction == 0.06
    assert support64.final_fraction == 0.02
    assert support64.candidate_fraction == 0.06
    assert support128.final_fraction == 0.01
    assert support128.candidate_fraction == 0.04


def test_policy_caps_longer_context_support() -> None:
    support = count_capped_support(256_000)

    assert support.final_token_count == 1280
    assert support.final_fraction == 0.005
    assert support.candidate_fraction == 0.03
