from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from summarize_short_context_optimization_20260723 import (
    paired_speed,
    summarize_methods,
    validate,
)


def row(task, sample_id, method, score, query, decode, total):
    return {
        "task": task,
        "sample_id": sample_id,
        "method": method,
        "score": str(score),
        "query_seconds": str(query),
        "decode_seconds": str(decode),
        "online_seconds": str(query + decode),
        "total_seconds": str(total),
    }


def test_summary_uses_task_macro_and_paired_ratio_of_sums():
    rows = [
        row("a", "0", "full_kv", 1.0, 1.0, 3.0, 6.0),
        row("a", "0", "candidate", 0.8, 0.5, 1.5, 3.0),
        row("b", "1", "full_kv", 0.5, 2.0, 2.0, 8.0),
        row("b", "1", "candidate", 0.5, 1.0, 1.0, 4.0),
    ]
    validate(rows, {"full_kv": 2, "candidate": 2}, expected_tasks=2)
    summary = {item["method"]: item for item in summarize_methods(rows)}
    assert summary["full_kv"]["macro_score"] == 0.75
    assert summary["candidate"]["macro_score"] == 0.65
    assert paired_speed(rows, "candidate", "online_seconds") == 2.0
    assert summary["candidate"]["total_speedup"] == 2.0
