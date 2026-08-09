from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_streaming_hypergraph_layout import streaming_hypergraph_positions


def test_streaming_layout_is_permutation_and_packs_recent_hyperedge() -> None:
    training = [
        np.asarray([0, 4, 5, 6], dtype=np.int32),
        np.asarray([1, 5, 6, 7], dtype=np.int32),
    ]
    positions = streaming_hypergraph_positions(training, token_count=8)

    assert sorted(positions.tolist()) == list(range(8))
    recent_pages = np.unique(positions[[1, 5, 6, 7]] // 4)
    assert recent_pages.size == 1
