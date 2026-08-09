from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analyze_diagnostics import summarize_overlap, summarize_tokens, token_category  # noqa: E402
from compare_selectors import block_bootstrap_ci, paired_delta  # noqa: E402
from run_selector_ppl import build_sanity_checks  # noqa: E402
from run_selector_ppl import Top2Diagnostics  # noqa: E402


def test_token_category() -> None:
    assert token_category("\n") == "newline"
    assert token_category(" ") == "whitespace"
    assert token_category("...") == "punctuation"
    assert token_category("42") == "number"
    assert token_category(" attention") == "word"
    assert token_category("R2D2") == "alphanumeric"


def test_token_summary_uses_exposure_enrichment() -> None:
    rows = [
        {
            "token_index": "0",
            "token_text": " word",
            "token_piece": "word",
            "eligible_event_count": "100",
            "top2_selected_count": "20",
            "top2_attention_mass_sum": "4",
            "sink_role_count": "20",
            "recent_role_count": "0",
            "remote_role_count": "0",
        },
        {
            "token_index": "1",
            "token_text": ".",
            "token_piece": ".",
            "eligible_event_count": "300",
            "top2_selected_count": "20",
            "top2_attention_mass_sum": "1",
            "sink_role_count": "0",
            "recent_role_count": "5",
            "remote_role_count": "15",
        },
    ]
    categories, roles, top = summarize_tokens(rows, top_n=1)
    word = next(row for row in categories if row["token_category"] == "word")
    assert math.isclose(float(word["selection_enrichment_vs_exposure"]), 2.0)
    assert sum(int(row["selected_events"]) for row in roles) == 40
    assert len(top) == 1


def test_overlap_summary_is_event_weighted() -> None:
    rows = [
        {
            "sink_tokens": "2",
            "query_count": "2",
            "top2_selected_events": "10",
            "overlap_events": "5",
            "top2_attention_mass_sum": "4",
            "overlap_attention_mass_sum": "3",
            "mean_sink_recent_full_attention_mass": "0.5",
            "mean_pruned_distribution_cosine": "0.8",
        },
        {
            "sink_tokens": "2",
            "query_count": "1",
            "top2_selected_events": "10",
            "overlap_events": "10",
            "top2_attention_mass_sum": "2",
            "overlap_attention_mass_sum": "2",
            "mean_sink_recent_full_attention_mass": "0.2",
            "mean_pruned_distribution_cosine": "0.5",
        },
    ]
    summary = summarize_overlap(rows)[0]
    assert math.isclose(float(summary["overlap_event_recall"]), 0.75)
    assert math.isclose(float(summary["mean_pruned_distribution_cosine"]), 0.7)


def test_block_bootstrap_constant_delta() -> None:
    mean, low, high = block_bootstrap_ci([0.25] * 128, 16, 100, 7)
    assert mean == low == high == 0.25
    assert paired_delta({1: 3.0, 2: 4.0}, {1: 1.0, 2: 1.5}) == [2.0, 2.5]


def test_sanity_checks_compare_equal_budget_modes() -> None:
    ppl_rows = [
        {
            "mode": "full_attention",
            "selector": "full_attention",
            "ratio": "",
            "delta_loss_vs_full": 0.0,
            "kept_percent_actual_history": 100.0,
        },
        {
            "mode": "top_attention_r1",
            "selector": "top_attention",
            "ratio": 1.0,
            "delta_loss_vs_full": 0.0,
            "kept_percent_actual_history": 100.0,
        },
        {
            "mode": "top_attention_r0p02",
            "selector": "top_attention",
            "ratio": 0.02,
            "delta_loss_vs_full": -0.1,
            "kept_percent_actual_history": 2.1,
        },
        {
            "mode": "sink_recent_s0_r0p02",
            "selector": "sink_recent",
            "ratio": 0.02,
            "delta_loss_vs_full": 0.1,
            "kept_percent_actual_history": 2.1,
        },
        {
            "mode": "recent_r0p02",
            "selector": "recent",
            "ratio": 0.02,
            "delta_loss_vs_full": 0.1,
            "kept_percent_actual_history": 2.1,
        },
    ]
    token_rows = [
        {"mode": mode, "token_index": 10, "nll": 1.25}
        for mode in [
            "full_attention",
            "top_attention_r1",
            "top_attention_r0p02",
            "sink_recent_s0_r0p02",
            "recent_r0p02",
        ]
    ]
    checks = build_sanity_checks(ppl_rows, token_rows, 0.02)
    assert checks["top_attention_100_matches_full_at_1e_6"]
    assert checks["all_modes_score_same_token_indices"]
    assert checks["sink_recent_s0_matches_recent_at_1e_7"]
    assert checks["equal_budget_selectors_match_at_1e_9_percent"]


def test_top2_union_tracks_layer_model_and_temporal_unions() -> None:
    diagnostics = Top2Diagnostics(
        layer_count=2,
        head_count=2,
        total_tokens=5,
        ratio=0.25,
        sink_sweep=[0],
        always_keep_self=True,
        role_sink_tokens=1,
        role_recent_tokens=1,
        random_seed=7,
    )
    scores = torch.zeros((1, 2, 1, 5), dtype=torch.float32)
    keep0 = torch.zeros_like(scores, dtype=torch.bool)
    history0 = torch.zeros_like(scores, dtype=torch.bool)
    history0[0, 0, 0, 0] = True
    history0[0, 1, 0, 1] = True
    keep0 |= history0
    keep0[:, :, :, 4] = True
    diagnostics.update(SimpleNamespace(layer_idx=0), scores, keep0, history0)

    keep1 = torch.zeros_like(scores, dtype=torch.bool)
    history1 = torch.zeros_like(scores, dtype=torch.bool)
    history1[0, 0, 0, 1] = True
    history1[0, 1, 0, 2] = True
    keep1 |= history1
    keep1[:, :, :, 4] = True
    diagnostics.update(SimpleNamespace(layer_idx=1), scores, keep1, history1)

    assert [row["union_tokens"] for row in diagnostics.layer_query_union_rows] == [2, 2]
    model_row = diagnostics.model_union_rows()[0]
    assert model_row["union_tokens"] == 3
    assert model_row["selected_head_token_events"] == 4
    assert math.isclose(float(model_row["union_fraction_of_history"]), 0.75)
    temporal = diagnostics.temporal_union_rows()
    assert [row["union_tokens_across_queries"] for row in temporal] == [2, 2, 3]
