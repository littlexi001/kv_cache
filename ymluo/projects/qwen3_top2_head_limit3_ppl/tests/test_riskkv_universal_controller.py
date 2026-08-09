from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from riskkv_universal_controller import UniversalKVController, match_is_covered


def test_match_coverage_requires_the_whole_signature() -> None:
    keep = set(range(100, 107))
    assert not match_is_covered(100, 8, keep)
    keep.add(107)
    assert match_is_covered(100, 8, keep)


def test_no_match_keeps_local_and_semantic_memory() -> None:
    action = UniversalKVController().choose_action(None, 31_744, set())
    assert action.name == "local_semantic_2k"
    assert action.semantic_tokens == 480
    assert not action.rebuild_cache


def test_local_match_does_not_rebuild() -> None:
    action = UniversalKVController().choose_action(31_000, 31_744, set(range(30_000, 31_744)))
    assert action.name == "local_semantic_2k"
    assert "already" in action.reason


def test_near_remote_match_uses_content_addressed_echo() -> None:
    action = UniversalKVController().choose_action(29_100, 31_744, set())
    assert action.name == "recurrence_echo_2k"
    assert action.budget_tokens == 2048
    assert action.echo_tokens == 480
    assert action.rebuild_cache


def test_far_remote_match_uses_echo_slot_without_budget_growth() -> None:
    action = UniversalKVController().choose_action(10_000, 31_744, set())
    assert action.name == "recurrence_echo_2k"
    assert action.echo_tokens == 480
    assert action.budget_tokens == 2048


def test_continuity_risk_is_the_only_recent_budget_upgrade_signal() -> None:
    action = UniversalKVController().choose_action(
        None, 31_744, set(), continuity_risk=True
    )
    assert action.name == "expanded_recent_2p8k"
    assert action.budget_tokens == 2816
    assert action.recent_tokens == 2784


def test_unstable_match_is_ignored() -> None:
    action = UniversalKVController().choose_action(10_000, 31_744, set(), stable_match=False)
    assert action.name == "local_semantic_2k"
    assert not action.rebuild_cache
