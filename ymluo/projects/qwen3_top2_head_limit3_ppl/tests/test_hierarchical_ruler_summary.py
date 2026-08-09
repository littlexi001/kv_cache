from __future__ import annotations

from summarize_hierarchical_ruler_shards_20260716 import build_length_gated_rows


def make_row(
    task: str, sample_id: str, method: str, length: int, score: float
) -> dict[str, str]:
    return {
        "task": task,
        "sample_id": sample_id,
        "method": method,
        "requested_length": str(length),
        "score": str(score),
    }


def test_length_gate_uses_full_below_threshold_and_sparse_at_threshold() -> None:
    rows = [
        make_row("short", "0", "full_kv", 8192, 1.0),
        make_row("short", "0", "hierarchical_pca_perhead", 8192, 0.5),
        make_row("long", "0", "full_kv", 16384, 0.25),
        make_row("long", "0", "hierarchical_pca_perhead", 16384, 0.75),
    ]

    gated = build_length_gated_rows(rows, 16384)
    by_task = {row["task"]: row for row in gated}

    assert by_task["short"]["score"] == "1.0"
    assert by_task["long"]["score"] == "0.75"
    assert {row["method"] for row in gated} == {"hierarchical_length_gate_16384"}
