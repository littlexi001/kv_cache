from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from run_data_protocol_audit_20260717 import protocol_events  # noqa: E402
from run_length_causal_mechanism_20260717 import build_bundle, mechanism_rows  # noqa: E402


def test_fixed_candidate_pool_contains_start_and_all_diagnostic_answers() -> None:
    bundle = build_bundle(7)
    candidates = bundle["candidates"]
    assert len(candidates) == 13
    assert len(set(candidates)) == 13
    assert bundle["gold_codes"][0] in candidates
    assert bundle["gold_codes"][1] in candidates
    assert bundle["gold_codes"][2] in candidates
    assert bundle["conflict_codes"][1] in candidates
    assert bundle["conflict_codes"][2] in candidates
    assert bundle["candidate_roles"][bundle["gold_codes"][0]] == "gold_start"


def test_ambiguous_protocol_removes_gold_conflict_authority_labels() -> None:
    _, gold, conflict = protocol_events(0, "ambiguous")
    text = "".join(event.text for event in [*gold, *conflict])
    assert "VERIFIED" not in text
    assert "DECOY" not in text
    assert "OFFICIAL" not in text
    assert all(event.text.startswith("RULE R") for event in [*gold, *conflict])


def test_mechanism_uses_input_excluded_cloze_for_access_diagnosis() -> None:
    common = {
        "seed": 0,
        "target_context_tokens": 65536,
        "placement": "middle",
        "condition": "clean",
        "candidate_prediction_role": "gold_start",
    }
    rows = [
        {
            **common,
            "query_mode": "full2",
            "generation_final_correct": 0,
            "candidate_correct": 0,
            "start_excluded_candidate_correct": 0,
        },
        {
            **common,
            "query_mode": "hop1",
            "candidate_correct": 1,
            "start_excluded_candidate_correct": 1,
        },
        {
            **common,
            "query_mode": "oracle_hop2",
            "candidate_correct": 0,
            "start_excluded_candidate_correct": 1,
        },
    ]
    detailed, summary = mechanism_rows(rows)
    assert detailed[0]["failure_mechanism"] == "composition_or_state_update_failure"
    assert detailed[0]["oracle_hop2_cloze_strict_correct"] == 0
    assert detailed[0]["oracle_hop2_cloze_correct"] == 1
    assert summary[0]["failure_mechanism"] == "composition_or_state_update_failure"
