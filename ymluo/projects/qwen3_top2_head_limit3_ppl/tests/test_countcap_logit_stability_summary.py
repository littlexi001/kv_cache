from __future__ import annotations

import pytest

from summarize_countcap_logit_stability_20260726 import summarize_rows


def make_row(
    topic: str,
    agreement: int,
    certificate: int,
    kl: float,
) -> dict[str, str]:
    return {
        "topic": topic,
        "method": "direct_countcap",
        "top1_agreement": str(agreement),
        "margin_certificate_satisfied": str(certificate),
        "full_top1_margin": "2.0",
        "shift_invariant_logit_delta_range": "1.0",
        "kl_full_to_sparse": str(kl),
        "kl_range_upper_bound": "0.5",
        "kl_range_bound_satisfied": "1",
        "js_divergence": str(kl / 2),
        "target_nll_delta": "0.1",
        "target_nll_range_bound_satisfied": "1",
    }


def test_summarize_rows_reports_certificates_and_flips():
    rows = [
        make_row("sports", 1, 1, 0.01),
        make_row("sports", 1, 0, 0.02),
        make_row("sports", 0, 0, 0.03),
        {"topic": "sports", "method": "full_attention"},
    ]

    summaries = {
        row["topic"]: row for row in summarize_rows(rows)
    }

    assert summaries["sports"]["tokens"] == 3
    assert summaries["sports"]["top1_agreement_mean"] == pytest.approx(2 / 3)
    assert summaries["sports"]["certified_violations"] == 0
    assert summaries["sports"]["uncertified_flip_rate"] == pytest.approx(0.5)
    assert summaries["sports"]["kl_full_to_sparse_mean"] == pytest.approx(0.02)
    assert summaries["ALL"]["tokens"] == 3
