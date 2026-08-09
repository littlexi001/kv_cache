from __future__ import annotations

import pytest

import run_direct_countcap_denseprompt_ppl_20260725 as direct_runner
from benchmark_qksieve_per_head_cold_skip_20260730 import (
    ragged_attention_split_count,
)
from run_head_top2_targeted_ppl_20260714 import (
    _configure_packed_qmse_state,
)


@pytest.mark.parametrize(
    ("mode", "hot_fraction"),
    (
        (
            direct_runner.PACKED_QMSE_QKMETRIC_FREQSKIP50_SCORE_MODE,
            0.50,
        ),
        (
            direct_runner.PACKED_QMSE_QKMETRIC_FREQSKIP60_SCORE_MODE,
            0.60,
        ),
    ),
)
def test_frequency_cold_skip_mode_contract(
    mode: str,
    hot_fraction: float,
) -> None:
    assert mode in direct_runner.PACKED_QMSE_SCORE_MODES
    assert mode in direct_runner.PACKED_PREFILL_QUERY_SCORE_MODES
    assert mode in direct_runner.PACKED_FREQUENCY_PREFILL_QUERY_SCORE_MODES
    state: dict[str, object] = {}
    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_transform"] == "qk_metric"
    assert state["packed_qmse_full_topk"] is True
    assert state["packed_qmse_frequency_tiered"] is True
    assert state["packed_qmse_frequency_hard_skip"] is True
    assert state["packed_qmse_frequency_hot_fraction"] == hot_fraction
    assert state["packed_qmse_frequency_block_size"] == 0
    assert state["packed_qmse_frequency_cold_shards"] == 4
    assert state["packed_qmse_frequency_recent_tokens"] == 256
    assert state["packed_qmse_frequency_carry_previous"] is True


@pytest.mark.parametrize(
    ("candidate_capacity", "expected"),
    (
        (4096, 8),
        (4097, 4),
        (44_000, 4),
        (44_001, 8),
        (88_000, 8),
        (88_001, 16),
        (176_000, 16),
    ),
)
def test_ragged_attention_split_capacity(
    candidate_capacity: int,
    expected: int,
) -> None:
    assert ragged_attention_split_count(candidate_capacity) == expected


def test_ragged_attention_split_rejects_unsupported_capacity() -> None:
    with pytest.raises(RuntimeError, match="more than 16 splits"):
        ragged_attention_split_count(176_001)
