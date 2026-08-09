from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_coaccess_page_gather import build_page_workload


def test_build_page_workload_keeps_flat_page_ids_in_range(tmp_path: Path) -> None:
    indices = np.asarray(
        [
            [
                [[0, 1], [0, 4], [1, 5], [4, 5]],
                [[1, 2], [0, 5], [2, 6], [5, 6]],
            ]
        ],
        dtype=np.int32,
    )
    archive = tmp_path / "selection_indices.npz"
    np.savez_compressed(
        archive,
        indices=indices,
        selected_heads=np.arange(2, dtype=np.int32),
        context_token_ids=np.arange(8, dtype=np.int64),
    )

    chronological, coaccess, metadata = build_page_workload(
        archive,
        train_observations=2,
        test_queries=2,
        group_size=2,
        page_size=2,
        microblock_size=1,
        neighbor_count=2,
    )

    assert len(chronological) == len(coaccess) == 2
    assert all(values.min() >= 0 for values in chronological + coaccess)
    assert all(values.max() < metadata["total_pages"] for values in chronological + coaccess)
    assert metadata["physical_kv_groups"] == 1


def test_build_page_workload_accepts_streaming_layout(tmp_path: Path) -> None:
    indices = np.asarray(
        [
            [
                [[0, 1], [0, 4], [1, 5], [4, 5]],
                [[1, 2], [0, 5], [2, 6], [5, 6]],
            ]
        ],
        dtype=np.int32,
    )
    archive = tmp_path / "selection_indices.npz"
    np.savez_compressed(
        archive,
        indices=indices,
        selected_heads=np.arange(2, dtype=np.int32),
        context_token_ids=np.arange(8, dtype=np.int64),
    )

    _, _, metadata = build_page_workload(
        archive,
        train_observations=2,
        test_queries=2,
        group_size=2,
        page_size=2,
        microblock_size=1,
        neighbor_count=2,
        layout_method="streaming_hypergraph",
    )

    assert metadata["layout_method"] == "streaming_hypergraph"
