from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from summarize_qk_variable_physical_128k_20260727 import (
    VARIANTS,
    configured_candidate_count,
)


def test_configured_candidate_count_applies_floor_and_cap() -> None:
    payload = {
        "final_cache_length": 128_257,
        "candidate_fraction": 0.06,
        "candidate_min_tokens": 256,
        "candidate_max_tokens": 1280,
    }
    assert configured_candidate_count(payload) == 1280

    payload["final_cache_length"] = 2_001
    assert configured_candidate_count(payload) == 256

    payload["final_cache_length"] = 8_001
    assert configured_candidate_count(payload) == 480


def test_summary_compares_auto_and_fixed_sampled_endpoints() -> None:
    assert ("sampled_compact", "qksampled") in VARIANTS
    assert (
        "fixed4421_sampled_compact",
        "qkfixed4421sampled",
    ) in VARIANTS
