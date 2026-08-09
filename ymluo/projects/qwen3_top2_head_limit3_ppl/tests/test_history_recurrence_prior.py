from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_history_recurrence_prior_20260714 import (
    label_candidate_starts,
    mine_recurrence_episodes,
)


def test_episode_mining_and_candidate_labeling() -> None:
    remote = list(range(40))
    target = [20, 21, 22, 23, 24, 25, 90]
    episodes = mine_recurrence_episodes(remote, target, 3)
    assert episodes[0]["source_start"] == 20
    assert episodes[0]["matched_tokens"] > 3
    positives = label_candidate_starts(episodes, [0, 8, 16, 24, 32], span_tokens=12)
    assert positives == {16}
