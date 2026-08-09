from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_pca_kv_layout import balanced_kd_positions, symmetric_int4


def test_balanced_kd_positions_is_a_permutation_and_groups_clusters() -> None:
    vectors = np.asarray(
        [[0.0], [10.0], [0.2], [10.2], [0.1], [10.1], [0.3], [10.3]],
        dtype=np.float32,
    )
    positions = balanced_kd_positions(vectors, page_size=4)

    assert sorted(positions.tolist()) == list(range(8))
    assert len(set((positions[[0, 2, 4, 6]] // 4).tolist())) == 1
    assert len(set((positions[[1, 3, 5, 7]] // 4).tolist())) == 1


def test_symmetric_int4_uses_valid_signed_range() -> None:
    values = np.asarray([[-4.0, 0.0], [0.0, 2.0], [4.0, -2.0]], dtype=np.float32)
    quantized = symmetric_int4(values)

    assert quantized.dtype == np.int8
    assert quantized.min() >= -7
    assert quantized.max() <= 7
    assert quantized[0, 0] == -7
    assert quantized[-1, 0] == 7
