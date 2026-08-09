from __future__ import annotations

from summarize_final_direct_multimodel_comparison_20260726 import (
    summarize_model,
)


def make_rows() -> list[dict[str, str]]:
    rows = []
    all_tasks = sorted(
        {
            "trec",
            "samsum",
            "passage_count",
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "qmsum",
            "triviaqa",
            "passage_retrieval_en",
            "gov_report",
            "multi_news",
            "lcc",
            "repobench-p",
        }
    )
    for task in all_tasks:
        for method, score, online, total, decode, generated in (
            ("full_kv", 0.8, 2.0, 3.0, 1.6, 8),
            ("countcap_direct", 0.76, 1.0, 2.0, 0.8, 8),
        ):
            rows.append(
                {
                    "task": task,
                    "sample_id": "0",
                    "method": method,
                    "score": str(score),
                    "prompt_tokens": "8000",
                    "configured_attention_tokens": (
                        "8000" if method == "full_kv" else "480"
                    ),
                    "configured_attention_fraction": (
                        "1.0" if method == "full_kv" else "0.06"
                    ),
                    "online_seconds": str(online),
                    "total_seconds": str(total),
                    "decode_seconds": str(decode),
                    "generated_tokens": str(generated),
                }
            )
    return rows


def test_summarize_model_reports_all_published_task_subsets():
    overall, tasks = summarize_model(
        make_rows(),
        "test-model",
        bootstrap_samples=20,
        seed=1,
    )
    by_subset = {row["subset"]: row for row in overall}

    assert by_subset["longbench16"]["tasks"] == 16
    assert by_subset["longbench16"]["paired_samples"] == 16
    assert by_subset["longbench16"]["quality_retention"] == 0.95
    assert by_subset["longbench16"]["paired_online_speedup"] == 2.0
    assert by_subset["longbench16"]["paired_decode_per_token_speedup"] == 2.0
    assert by_subset["longbench16"]["paired_online_per_token_speedup"] == 2.0
    assert by_subset["longbench16"]["aggregate_decode_per_token_speedup"] == 2.0
    assert by_subset["longbench16"]["aggregate_online_per_token_speedup"] == 2.0
    assert by_subset["rabitq13"]["tasks"] == 13
    assert by_subset["selfindex11"]["tasks"] == 11
    assert len(tasks) == 16
