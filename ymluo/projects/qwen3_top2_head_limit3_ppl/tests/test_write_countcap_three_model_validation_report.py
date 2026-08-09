import csv
import json
import sys
from pathlib import Path

from write_countcap_three_model_validation_report_20260726 import main


TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "qmsum",
    "trec",
    "triviaqa",
    "samsum",
    "passage_retrieval_en",
    "passage_count",
    "gov_report",
    "multi_news",
    "lcc",
    "repobench-p",
)


def test_writes_three_model_report(tmp_path: Path, monkeypatch) -> None:
    comparison = {
        "countcap": [],
        "countcap_by_task": [],
        "published_reference": {
            "SelfIndexingKVCache_Llama31_8B_LongBench11": {
                "full_score": 58.7,
                "self_indexing_16bit_score": 58.4,
                "self_indexing_2bit_score": 58.2,
                "qwen25_14b_full_score": 56.9,
                "qwen25_14b_self_indexing_16bit_score": 55.9,
                "qwen25_14b_self_indexing_2bit_score": 55.7,
                "budget_tokens": 160,
            }
        },
    }
    for model in ("Llama-3.1-8B-Instruct", "Qwen3-4B-Instruct"):
        comparison["countcap"].append(
            {
                "model": model,
                "subset": "longbench16",
                "paired_samples": 16,
                "full_macro": 0.5,
                "countcap_macro": 0.49,
                "quality_retention": 0.98,
                "macro_delta_ci95_low": -0.02,
                "macro_delta_ci95_high": 0.0,
                "mean_attention_tokens": 400,
                "mean_attention_fraction": 0.07,
                "paired_online_per_token_speedup": 0.9,
            }
        )
        comparison["countcap"].append(
            {
                "model": model,
                "subset": "selfindex11",
                "paired_samples": 11,
                "full_macro": 0.52,
                "countcap_macro": 0.515,
                "quality_retention": 0.515 / 0.52,
                "mean_attention_tokens": 375,
            }
        )
        for task in TASKS:
            comparison["countcap_by_task"].append(
                {
                    "model": model,
                    "task": task,
                    "full_score": 0.5,
                    "countcap_score": 0.49,
                }
            )
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    qwen_path = tmp_path / "qwen25.csv"
    qwen_rows = []
    for task in TASKS:
        for method, score, seconds in (
            ("full_kv", 0.5, 2.0),
            ("countcap", 0.49, 2.2),
        ):
            qwen_rows.append(
                {
                    "task": task,
                    "sample_id": "0",
                    "method": method,
                    "score": score,
                    "prompt_tokens": 7000,
                    "configured_attention_tokens": (
                        7000 if method == "full_kv" else 420
                    ),
                    "configured_attention_fraction": (
                        1.0 if method == "full_kv" else 0.06
                    ),
                    "online_seconds": seconds,
                    "total_seconds": seconds + 1.0,
                    "generated_tokens": 20,
                    "decode_seconds": seconds / 2.0,
                }
            )
    with qwen_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(qwen_rows[0]))
        writer.writeheader()
        writer.writerows(qwen_rows)

    long_speed = [
        {
            "model": "qwen3_4b",
            "history_tokens": 64000,
            "paired_cases": 4,
            "ppl_retention": 0.9,
            "actual_attention_tokens": 1450,
            "actual_attention_tokens_min": 100,
            "actual_attention_tokens_max": 3800,
            "decode_speedup": 2.7,
            "additional_fixed_seconds_per_case": 0.5,
            "break_even_decode_steps": 5.0,
            "protocol_speedup": 1.4,
        }
    ]
    long_path = tmp_path / "long.json"
    long_path.write_text(json.dumps(long_speed), encoding="utf-8")
    spectrum_path = tmp_path / "qwen25_spectrum.json"
    spectrum_path.write_text(
        json.dumps(
            {
                "by_model": [
                    {
                        "model": "qwen25_7b",
                        "cases": 40,
                        "key_effective_rank_mean": 20.0,
                        "centered_qk_effective_rank_mean": 5.0,
                        "centered_qk_energy_retained_optimal_rank48_mean": 0.99,
                        "centered_full_key_pca_qk_fidelity_mean": 0.92,
                        "centered_production_prefix_pca_qk_fidelity_mean": 0.61,
                        "centered_production_prefix_pca_qk_score_cosine_mean": 0.78,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "writer",
            "--comparison",
            str(comparison_path),
            "--qwen25_csv",
            str(qwen_path),
            "--long_speed",
            str(long_path),
            "--qwen25_spectrum",
            str(spectrum_path),
            "--output",
            str(output),
            "--bootstrap_samples",
            "20",
        ],
    )
    main()
    text = output.read_text(encoding="utf-8")
    assert "Qwen2.5-7B-Instruct" in text
    assert "16 个英文任务" in text
    assert "90.00%" in text
    assert "Self-Indexing KVCache" in text
    assert "SALS" in text
    assert "centered QK 谱外推" in text
    assert "99.00%" in text
    assert "Qwen2.5-14B" in text
    assert "160 token" in text
