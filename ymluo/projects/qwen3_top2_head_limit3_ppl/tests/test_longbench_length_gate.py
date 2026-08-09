from __future__ import annotations

from summarize_hierarchical_longbench_shards_20260716 import add_auto_length_gate


def row(sample: str, method: str, prompt_tokens: int, score: float) -> dict[str, str]:
    return {
        "task": "task",
        "sample_id": sample,
        "method": method,
        "prompt_tokens": str(prompt_tokens),
        "score": str(score),
    }


def test_length_gate_uses_full_below_threshold_and_sparse_above() -> None:
    rows = [
        row("short", "full_kv", 8000, 1.0),
        row("short", "hierarchical_pca_perhead", 8000, 0.0),
        row("long", "full_kv", 32000, 0.0),
        row("long", "hierarchical_pca_perhead", 32000, 1.0),
    ]
    gated = add_auto_length_gate(rows, 16384)
    synthetic = {
        item["sample_id"]: item
        for item in gated
        if item["method"] == "auto_length_gate"
    }
    assert synthetic["short"]["score"] == "1.0"
    assert synthetic["long"]["score"] == "1.0"
