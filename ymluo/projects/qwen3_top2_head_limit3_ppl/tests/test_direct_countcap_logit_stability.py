from __future__ import annotations

import math

import pytest
import torch

from run_direct_countcap_denseprompt_ppl_20260725 import (
    logit_stability_metrics,
)


def test_identical_logits_have_zero_divergence_and_keep_top1():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    metrics = logit_stability_metrics(logits, logits.clone(), label=2)

    assert metrics["top1_agreement"] == 1
    assert metrics["margin_certificate_satisfied"] == 1
    assert metrics["shift_invariant_logit_delta_range"] == 0.0
    assert metrics["kl_full_to_sparse"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["kl_range_upper_bound"] == pytest.approx(0.0)
    assert metrics["kl_range_bound_satisfied"] == 1
    assert metrics["js_divergence"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["target_nll_delta"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["target_nll_range_bound_satisfied"] == 1


def test_common_logit_shift_is_softmax_invariant():
    full = torch.tensor([[-2.0, 0.0, 4.0, 1.0]])
    sparse = full + 17.0
    metrics = logit_stability_metrics(full, sparse, label=0)

    assert metrics["top1_agreement"] == 1
    assert metrics["shift_invariant_logit_delta_range"] == pytest.approx(0.0)
    assert metrics["kl_full_to_sparse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["target_nll_delta"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["kl_range_bound_satisfied"] == 1
    assert metrics["target_nll_range_bound_satisfied"] == 1


def test_margin_certificate_is_sufficient_but_not_necessary():
    full = torch.tensor([[5.0, 3.0, 0.0]])
    certified = torch.tensor([[5.4, 3.0, -0.2]])
    uncertified = torch.tensor([[3.1, 3.0, 1.0]])

    certified_metrics = logit_stability_metrics(
        full,
        certified,
        label=0,
    )
    uncertified_metrics = logit_stability_metrics(
        full,
        uncertified,
        label=0,
    )

    assert certified_metrics["margin_certificate_satisfied"] == 1
    assert certified_metrics["top1_agreement"] == 1
    assert uncertified_metrics["margin_certificate_satisfied"] == 0
    assert uncertified_metrics["top1_agreement"] == 1
    assert math.isfinite(uncertified_metrics["kl_full_to_sparse"])
    assert certified_metrics["kl_range_bound_satisfied"] == 1
    assert certified_metrics["target_nll_range_bound_satisfied"] == 1


def test_logit_stability_validates_shapes_and_label():
    with pytest.raises(ValueError, match="same shape"):
        logit_stability_metrics(
            torch.zeros(3),
            torch.zeros(4),
            label=0,
        )
    with pytest.raises(ValueError, match="outside"):
        logit_stability_metrics(
            torch.zeros(3),
            torch.zeros(3),
            label=3,
        )
