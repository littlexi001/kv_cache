from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_qksieve_tail_partition_calibration_20260803 import (
    append_query_crossfit_conditional_rows,
    clipped_wiener_gain,
    vector_bernstein_radius,
)


def test_vector_bernstein_radius_matches_closed_form() -> None:
    variance = torch.tensor(4.0)
    maximum = torch.tensor(0.5)
    delta = 0.01
    dimension = 128
    logarithm = math.log(2.0 * dimension / delta)
    expected = math.sqrt(8.0 * logarithm) + logarithm / 3.0

    actual = vector_bernstein_radius(
        variance,
        maximum,
        failure_probability=delta,
        dimension=dimension,
    )

    torch.testing.assert_close(actual, torch.tensor(expected))


def test_vector_bernstein_radius_validates_contract() -> None:
    with pytest.raises(ValueError):
        vector_bernstein_radius(
            torch.tensor(1.0),
            torch.tensor(1.0),
            failure_probability=1.0,
            dimension=128,
        )
    with pytest.raises(ValueError):
        vector_bernstein_radius(
            torch.tensor(1.0),
            torch.tensor(1.0),
            failure_probability=0.01,
            dimension=0,
        )


def test_clipped_wiener_gain_preserves_exact_predictor() -> None:
    prediction = torch.tensor([[1.0, -2.0], [3.0, 4.0]])

    gain, reduction = clipped_wiener_gain(prediction, prediction)

    torch.testing.assert_close(gain, torch.tensor(1.0))
    torch.testing.assert_close(reduction, torch.tensor(1.0))


def test_clipped_wiener_gain_rejects_orthogonal_or_harmful_predictor() -> None:
    prediction = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    orthogonal_target = torch.tensor([[0.0, 1.0], [0.0, -1.0]])

    orthogonal_gain, orthogonal_reduction = clipped_wiener_gain(
        prediction, orthogonal_target
    )
    harmful_gain, harmful_reduction = clipped_wiener_gain(
        prediction, -prediction
    )

    torch.testing.assert_close(orthogonal_gain, torch.tensor(0.0))
    torch.testing.assert_close(orthogonal_reduction, torch.tensor(0.0))
    torch.testing.assert_close(harmful_gain, torch.tensor(0.0))
    torch.testing.assert_close(harmful_reduction, torch.tensor(0.0))


def test_clipped_wiener_gain_validates_shapes() -> None:
    with pytest.raises(ValueError):
        clipped_wiener_gain(torch.ones(2, 2), torch.ones(2, 3))


def _crossfit_row(
    *, record_index: int, method: str, output: torch.Tensor, full: torch.Tensor
) -> dict[str, object]:
    return {
        "model_name_or_path": "model",
        "trace": "trace",
        "record_index": record_index,
        "step": record_index,
        "topic": "topic",
        "history_tokens": 4096,
        "token_count": 4096,
        "layer": 0,
        "kv_head": 0,
        "query_head": 0,
        "query_head_count": 1,
        "candidate_mode": "proxy",
        "top_k": 256,
        "selected_mass": 0.5,
        "value_explained_variance": 0.9,
        "method": method,
        "sample_count": 0,
        "block_size": 4096,
        "tail_partition_relative_error": 0.0,
        "alpha": 1.0,
        "affine_slope": math.nan,
        "affine_residual_std": math.nan,
        "residual_risk_absolute": 1.0,
        "residual_risk_relative": 1.0,
        "residual_risk_range_absolute": 1.0,
        "residual_risk_bernstein_absolute": 1.0,
        "residual_risk_bernstein_relative": 1.0,
        "tail_correction_l2": 0.0,
        "tail_effective_tokens": 1.0,
        "proxy_selected_mass": 0.5,
        "conditional_gain": math.nan,
        "conditional_holdout_error_reduction": math.nan,
        "absolute_l2": 0.0,
        "full_output_l2": float(torch.linalg.vector_norm(full)),
        "relative_l2": 0.0,
        "cosine": 1.0,
        "_output_tensor": output.float(),
        "_full_output_tensor": full.float(),
    }


def test_query_crossfit_gain_uses_other_queries_and_recovers_half_scale() -> None:
    rows: list[dict[str, object]] = []
    for record_index, baseline in enumerate(
        (torch.tensor([1.0, 2.0]), torch.tensor([-1.0, 3.0]))
    ):
        direction = torch.tensor([2.0, -4.0])
        full = baseline + 0.5 * direction
        rows.append(
            _crossfit_row(
                record_index=record_index,
                method="block_residual_mean_proxy",
                output=baseline,
                full=full,
            )
        )
        rows.append(
            _crossfit_row(
                record_index=record_index,
                method="block_conditional_residual_proxy_d8",
                output=baseline + direction,
                full=full,
            )
        )

    appended = append_query_crossfit_conditional_rows(rows)
    crossfit = [row for row in rows if "query_crossfit" in str(row["method"])]

    assert appended == 2
    assert len(crossfit) == 2
    for row in crossfit:
        assert row["conditional_gain"] == pytest.approx(0.5)
        torch.testing.assert_close(
            row["_output_tensor"], row["_full_output_tensor"]
        )


def test_query_crossfit_rejects_harmful_direction() -> None:
    rows: list[dict[str, object]] = []
    for record_index in range(2):
        baseline = torch.tensor([1.0, -1.0])
        direction = torch.tensor([2.0, 1.0])
        full = baseline - direction
        rows.append(
            _crossfit_row(
                record_index=record_index,
                method="block_residual_mean_proxy",
                output=baseline,
                full=full,
            )
        )
        rows.append(
            _crossfit_row(
                record_index=record_index,
                method="block_conditional_residual_proxy_d8",
                output=baseline + direction,
                full=full,
            )
        )

    append_query_crossfit_conditional_rows(rows)
    crossfit = [row for row in rows if "query_crossfit" in str(row["method"])]

    assert len(crossfit) == 2
    for row in crossfit:
        assert row["conditional_gain"] == pytest.approx(0.0)
        torch.testing.assert_close(
            row["_output_tensor"], torch.tensor([1.0, -1.0])
        )


def test_conditional_wiener_score_mode_is_registered_end_to_end() -> None:
    mode = (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_"
        "valuesketch16i4shared_wometric_condres8wienerglobal_"
        "packed_fulltopk_oas"
    )
    for relative_path in (
        "src/run_head_top2_targeted_ppl_20260714.py",
        "src/run_direct_countcap_denseprompt_ppl_20260725.py",
    ):
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        normalized_source = "".join(source.replace('"', "").split())
        assert mode in normalized_source
    variant_source = (
        PROJECT_ROOT / "src/run_qksieve_coldskip_longcontext_quality_20260730.py"
    ).read_text(encoding="utf-8")
    assert "qksieve_qmse_oas_requestlocal_condres8wiener" in variant_source


def test_conditional_query_score_mode_is_registered_end_to_end() -> None:
    mode = (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_"
        "valuesketch16i4shared_wometric_condres8queryglobal_"
        "packed_fulltopk_oas"
    )
    for relative_path in (
        "src/run_head_top2_targeted_ppl_20260714.py",
        "src/run_direct_countcap_denseprompt_ppl_20260725.py",
    ):
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        normalized_source = "".join(source.replace('"', "").split())
        assert mode in normalized_source
    variant_source = (
        PROJECT_ROOT / "src/run_qksieve_coldskip_longcontext_quality_20260730.py"
    ).read_text(encoding="utf-8")
    assert "qksieve_qmse_oas_requestlocal_condres8query" in variant_source
