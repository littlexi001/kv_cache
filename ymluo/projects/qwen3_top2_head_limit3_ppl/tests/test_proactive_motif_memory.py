from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from diagnose_proactive_motif_memory_20260714 import minmax, rank_desc


def test_minmax_and_rank_desc() -> None:
    assert minmax([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert minmax([3.0, 3.0]) == [0.0, 0.0]
    assert rank_desc([0.1, 0.8, 0.4, 0.4], 2) == 2
