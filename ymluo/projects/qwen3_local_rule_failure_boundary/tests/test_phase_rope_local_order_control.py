from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import run_phase_rope_local_order_control_8b as target  # noqa: E402


class TinyTokenizer:
    def __init__(self) -> None:
        self.ids = {
            " alpha": 1,
            " beta": 2,
            " gamma": 3,
            " delta": 4,
        }

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [self.ids[text]]}


def test_order_pair_changes_only_order_and_gold_successor() -> None:
    first, second = target.make_order_pair(("alpha", "beta", "gamma", "delta"))

    assert sorted(first["sequence_words"]) == sorted(second["sequence_words"])
    assert first["anchor"] == second["anchor"] == "beta"
    assert first["gold"] == "gamma"
    assert second["gold"] == "delta"
    assert first["sequence_words"].index(first["gold"]) == 2
    assert second["sequence_words"].index(second["gold"]) == 2


def test_choice_metrics_uses_restricted_candidates_and_margin() -> None:
    tokenizer = TinyTokenizer()
    logits = torch.zeros(1, 1, 8)
    logits[0, 0, 3] = 4.5
    logits[0, 0, 4] = 2.0

    metrics = target.choice_metrics(
        tokenizer,
        logits,
        "gamma",
        ("alpha", "beta", "gamma", "delta"),
    )

    assert metrics["candidate_correct"] == 1
    assert metrics["candidate_prediction"] == "gamma"
    assert metrics["candidate_margin"] == 2.5


def _row(
    *,
    family: str,
    variant: str,
    seed: int,
    member: int,
    correct: int,
    prediction: str,
    nll: float,
) -> dict[str, object]:
    return {
        "task_family": family,
        "target_context_tokens": 8192,
        "variant": variant,
        "seed": seed,
        "pair_id": "pair" if family == "local_order" else "",
        "pair_member": member,
        "candidate_correct": correct,
        "candidate_prediction": prediction,
        "next_token_correct": correct,
        "candidate_margin": 1.0 if correct else -1.0,
        "gold_nll": nll,
        "gold_evidence_token_recall": 0.8,
        "gold_evidence_line_hit_rate": 0.9,
        "gold_chain_complete_rate": 0.7,
        "gold_evidence_attention_mass": 0.02,
        "nontrigger_exact_noop_max": 0.0,
        "query_seconds": 0.1,
    }


def test_summary_reports_counterfactual_pair_and_full_delta() -> None:
    rows = [
        _row(
            family="local_order",
            variant="full_rope",
            seed=0,
            member=0,
            correct=1,
            prediction="gamma",
            nll=1.0,
        ),
        _row(
            family="local_order",
            variant="full_rope",
            seed=0,
            member=1,
            correct=1,
            prediction="delta",
            nll=1.2,
        ),
        _row(
            family="local_order",
            variant="rope_top2",
            seed=0,
            member=0,
            correct=1,
            prediction="gamma",
            nll=1.1,
        ),
        _row(
            family="local_order",
            variant="rope_top2",
            seed=0,
            member=1,
            correct=0,
            prediction="gamma",
            nll=1.5,
        ),
        _row(
            family="remote_retrieval",
            variant="full_rope",
            seed=0,
            member=-1,
            correct=0,
            prediction="alpha",
            nll=2.0,
        ),
    ]

    summary = target.summarize(rows)
    indexed = {(row["task_family"], row["variant"]): row for row in summary}
    full = indexed[("local_order", "full_rope")]
    sparse = indexed[("local_order", "rope_top2")]

    assert full["counterfactual_pair_accuracy"] == 1.0
    assert full["prediction_changes_with_order_rate"] == 1.0
    assert sparse["counterfactual_pair_accuracy"] == 0.0
    assert sparse["prediction_changes_with_order_rate"] == 0.0
    assert sparse["candidate_accuracy_delta_vs_full"] == -0.5
    assert abs(float(sparse["gold_nll_delta_vs_full"]) - 0.2) < 1e-12
    assert sparse["nontrigger_exact_noop_max"] == 0.0


def test_protocol_explicitly_separates_local_and_remote_metrics() -> None:
    protocol = target.protocol()

    assert "counterfactual_pair_accuracy" in protocol["local_order"]["primary_metrics"]
    assert "gold_evidence_token_recall" in protocol["remote_retrieval"]["primary_metrics"]
    assert "frozen" in protocol["model_state"]
    assert "same exact pre-RoPE" in protocol["matched_support"]
    assert "nontrigger_exact_noop_max" in protocol["no_op_audit"]


def test_default_methods_cover_matched_exact_strict_and_npe() -> None:
    expected = {
        "full_rope",
        "rope_top2",
        "exact_pre_top2_postscore",
        "strict_mpr_pre_w128_lift25_gap1_f8_cap0p25",
        "strict_mpr_pre_w128_lift25_gap1_f8_cap0p25_masspreserve",
        "npe_native_pre_top2",
        "npe_rollback_pre_top2",
        "npe_rollback_masspreserve_pre_top2",
    }
    assert set(target.DEFAULT_VARIANTS) == expected
    for variant in expected - {"full_rope", "rope_top2"}:
        assert target.uses_shared_exact_pre_support(variant)


def test_explicit_adapter_dispatches_npe_then_falls_back_to_phase() -> None:
    # Simulate an unrelated import changing the mutable runner dispatch and
    # verify that the local-order experiment restores its intended adapter.
    target.runner.local_global_attention_forward = lambda *args, **kwargs: None
    target.install_attention_adapter()

    assert (
        target.runner.local_global_attention_forward
        is target.npe.native_phase_envelope_attention_forward
    )
    assert target.npe._PHASE_FORWARD is target.phase.phase_kernel_attention_forward


def test_unified_noop_metric_reads_strict_and_npe_audits() -> None:
    strict_variant = "strict_mpr_pre_w128_lift25_gap1_f8_cap0p25"
    npe_variant = "npe_rollback_pre_top2"
    metrics = {
        "strict_phase_nontrigger_noop_max": 0.0,
        "npe_unmodified_native_max_error": 0.0,
    }

    assert target.nontrigger_noop_error(strict_variant, metrics) == 0.0
    assert target.nontrigger_noop_error(npe_variant, metrics) == 0.0
    assert target.nontrigger_noop_error("exact_pre_top2_postscore", metrics) == 0.0
