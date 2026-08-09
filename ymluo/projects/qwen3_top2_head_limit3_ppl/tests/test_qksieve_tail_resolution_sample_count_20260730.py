from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_sample_calibrated_longbench_20260717 import (  # noqa: E402
    tail_resolution_sample_count,
)
from run_direct_countcap_denseprompt_ppl_20260725 import (  # noqa: E402
    PACKED_QKBALANCED_AUTOKEY_REALLOC_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_FIXED41111100_FULLTOPK_SCORE_MODE,
    SUPPORTED_METHODS,
    tail_resolution_sample_count as direct_tail_resolution_sample_count,
)
from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _PACKED_QMSE_FIXED_ALLOCATIONS,
    _packed_qmse_mode_contract,
)
from run_qksieve_coldskip_longcontext_quality_20260730 import (  # noqa: E402
    EXACT_FP16_QK_ORACLE,
    VARIANTS,
    variant_attention_token_budget,
    variant_integer,
)


def test_native_256k_c64_counts_are_aligned_and_conservative() -> None:
    history_tokens = 262_080
    cases = {
        10_240: 1_792,
        12_800: 1_536,
        15_728: 1_280,
    }
    for budget, expected in cases.items():
        actual = tail_resolution_sample_count(
            64,
            budget / history_tokens,
        )
        assert actual == expected
        assert actual % 256 == 0
        assert actual * budget / history_tokens >= 64


def test_six_percent_longbench_rate_uses_1280_samples() -> None:
    assert tail_resolution_sample_count(64, 0.06) == 1_280


def test_sample_count_respects_kernel_cap() -> None:
    assert tail_resolution_sample_count(64, 0.001) == 8_192


def test_direct_runner_uses_the_same_tail_resolution_rule() -> None:
    for target_tail_count, target_fraction in (
        (64, 0.04),
        (64, 0.06),
        (64, 0.10),
        (16, 0.01),
    ):
        assert direct_tail_resolution_sample_count(
            target_tail_count,
            target_fraction,
        ) == tail_resolution_sample_count(
            target_tail_count,
            target_fraction,
        )


def test_absolute_exact_qk_oracle_variants_are_registered() -> None:
    assert "exact_top_k" in SUPPORTED_METHODS
    assert VARIANTS["exact_qk_oracle_k256"] == EXACT_FP16_QK_ORACLE
    assert VARIANTS["exact_qk_oracle_k1280"] == EXACT_FP16_QK_ORACLE
    assert VARIANTS["exact_qk_oracle_k2560"] == EXACT_FP16_QK_ORACLE


def test_256k_selector_cause_split_variants_are_registered() -> None:
    for budget in (1280, 2560):
        assert (
            VARIANTS[
                f"qksieve_keymse_requestlocal_fulltopk_k{budget}"
            ]
            == VARIANTS[f"qksieve_keymse_fulltopk_k{budget}"]
        )
        assert (
            VARIANTS[f"qksieve_keymse_highbit_fulltopk_k{budget}"]
            == VARIANTS[f"qksieve_keymse_fulltopk_k{budget}"]
        )


def test_256k_transform_allocation_split_variants_are_registered() -> None:
    for budget in (1280, 2560):
        assert (
            VARIANTS[
                f"qksieve_keymse_requestlocal_fixedalloc_fulltopk_k{budget}"
            ]
            == PACKED_QKBALANCED_FIXED41111100_FULLTOPK_SCORE_MODE
        )
        assert (
            VARIANTS[
                f"qksieve_keymse_frozenbasis_realloc_fulltopk_k{budget}"
            ]
            == PACKED_QKBALANCED_AUTOKEY_REALLOC_FULLTOPK_SCORE_MODE
        )


def test_conditional_residual_has_a_matched_value_sketch_control() -> None:
    for budget in (256, 492, 1280, 2560):
        control = VARIANTS[
            f"qksieve_qmse_oas_requestlocal_valuesketch16_k{budget}"
        ]
        conditional = VARIANTS[
            f"qksieve_qmse_oas_requestlocal_condres8_k{budget}"
        ]
        assert "autoqmsetotal15z_qkmetric" in control
        assert "autoqmsetotal15z_qkmetric" in conditional
        assert "valuesketch16i4shared_wometric" in control
        assert "valuesketch16i4shared_wometric" in conditional
        assert "condres" not in control
        assert "condres8global" in conditional


def test_sampled_valuesketch_sample_count_variants_are_registered() -> None:
    for budget in (256, 1280):
        base = VARIANTS[
            f"qksieve_qmse_oas_requestlocal_valuesketch16_sampled_k{budget}"
        ]
        for sample_count in (512, 1024):
            name = (
                "qksieve_qmse_oas_requestlocal_valuesketch16_sampled_"
                f"s{sample_count}_k{budget}"
            )
            assert VARIANTS[name] == base
            assert variant_integer(name, "_s", 256) == sample_count
            assert variant_integer(name, "_k", 1280) == budget
            sorted_name = (
                "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_"
                f"s{sample_count}_k{budget}"
            )
            assert sorted_name in VARIANTS
            assert variant_integer(sorted_name, "_s", 256) == sample_count


def test_deterministic_valuesketch_tail_anchor_variants_are_registered() -> None:
    for budget in (256, 1280):
        for tail_anchors in (32, 64):
            name = (
                "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_"
                f"c{tail_anchors}_k{budget}"
            )
            assert name in VARIANTS
            assert variant_integer(name, "_c", 0) == tail_anchors
            assert variant_integer(name, "_k", 1280) == budget


def test_c32_and_c64_scale_sample_count_with_context_length() -> None:
    target_fraction = 1280 / 131_072
    assert direct_tail_resolution_sample_count(32, target_fraction) == 3_328
    assert direct_tail_resolution_sample_count(64, target_fraction) == 6_656


def test_frozen_c64_uses_the_documented_length_only_budget() -> None:
    variant = (
        "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280"
    )
    assert variant_attention_token_budget(variant, 4096) == 256
    assert variant_attention_token_budget(variant, 8192) == 492
    assert variant_attention_token_budget(variant, 16384) == 984
    assert variant_attention_token_budget(variant, 32768) == 1280
    assert variant_attention_token_budget(variant, 131072) == 1280


def test_oas_requestlocal_variants_do_not_require_a_frozen_template() -> None:
    source = (
        SRC / "run_qksieve_coldskip_longcontext_quality_20260730.py"
    ).read_text(encoding="utf-8")
    assert '"qksieve_qmse_oas_requestlocal_"' in source


def test_fixed_240bit_allocation_matches_physical_budget() -> None:
    allocation = _PACKED_QMSE_FIXED_ALLOCATIONS[
        PACKED_QKBALANCED_FIXED41111100_FULLTOPK_SCORE_MODE
    ]
    assert allocation == (4, 1, 1, 1, 1, 1, 0, 0)
    normalized_rate = sum(bits + int(bits > 0) for bits in allocation)
    assert 16 * normalized_rate == 240


def test_frozen_basis_reallocation_mode_uses_key_mse() -> None:
    assert _packed_qmse_mode_contract(
        PACKED_QKBALANCED_AUTOKEY_REALLOC_FULLTOPK_SCORE_MODE
    ) == ("qk_metric", "key_mse")
