from __future__ import annotations

import json
import sys

from write_countcap_final_research_report_20260726 import main, percent


def test_percent_formats_fraction() -> None:
    assert percent(0.9966) == "99.66%"


def test_main_writes_all_evidence_sections(tmp_path, monkeypatch) -> None:
    comparison = {
        "countcap": [
            {
                "model": model,
                "subset": "longbench16",
                "full_macro": 0.4,
                "countcap_macro": 0.396,
                "quality_retention": 0.99,
                "macro_delta_ci95_low": -0.01,
                "macro_delta_ci95_high": 0.002,
                "mean_attention_tokens": 450,
                "mean_attention_fraction": 0.06,
                "paired_online_speedup": 0.9,
                "paired_total_speedup": 0.95,
                "paired_decode_per_token_speedup": 1.1,
                "paired_online_per_token_speedup": 1.05,
            }
            for model in (
                "Llama-3.1-8B-Instruct",
                "Qwen3-4B-Instruct",
            )
        ]
        + [
            {
                "model": "Llama-3.1-8B-Instruct",
                "subset": "rabitq13",
                "full_macro": 0.45,
                "countcap_macro": 0.445,
                "mean_attention_fraction": 0.06,
            },
            {
                "model": "Llama-3.1-8B-Instruct",
                "subset": "selfindex11",
                "full_macro": 0.46,
                "countcap_macro": 0.455,
                "quality_retention": 0.455 / 0.46,
                "mean_attention_tokens": 420,
            },
        ],
        "countcap_by_task": [
            {
                "model": model,
                "task": "qasper",
                "full_score": 0.5,
                "countcap_score": 0.49,
                "quality_retention": 0.98,
            }
            for model in (
                "Llama-3.1-8B-Instruct",
                "Qwen3-4B-Instruct",
            )
        ],
        "same_environment_baselines": [
            {"method": "FullAttention", "score": 0.4, "budget": "100%"},
            {"method": "SnapKV", "score": 0.39, "budget": "1024"},
        ],
        "published_reference": {
            "RaBitQCache_Llama31_8B_LongBench13": {
                "full_score": 50.58,
                "rabitq_score": 50.63,
                "mean_budget_ratio": 0.1733,
            },
            "RaBitQCache_paper_LongBench13_table": [
                {
                    "method": "Full",
                    "setting": "-",
                    "score": 50.58,
                    "budget_ratio": 1.0,
                },
                {
                    "method": "RaBitQCache",
                    "setting": "top-p=0.95",
                    "score": 50.63,
                    "budget_ratio": 0.1733,
                },
            ],
            "SelfIndexingKVCache_Llama31_8B_LongBench11": {
                "full_score": 58.7,
                "self_indexing_16bit_score": 58.4,
                "self_indexing_2bit_score": 58.2,
                "budget_tokens": 160,
                "sink_tokens": 64,
                "dynamic_tokens": 96,
            },
        },
    }
    logit = {
        "topic": "ALL",
        "tokens": 100,
        "top1_agreement_mean": 0.95,
        "margin_certificate_satisfied_mean": 0.8,
        "kl_full_to_sparse_mean": 0.01,
        "js_divergence_mean": 0.002,
        "target_nll_delta_mean": 0.01,
        "kl_range_bound_satisfied_mean": 1.0,
        "target_nll_range_bound_satisfied_mean": 1.0,
    }
    crossing = {
        "candidate_overall": [
            {
                "method": "production_pca48_int4k_int8q",
                "fraction": 0.04,
                "topk_recall_mean": 0.69,
                "attention_mass_weighted_topk_recall_mean": 0.9,
                "retained_attention_mass_mean": 0.9,
                "retained_attention_mass_regret_mean": 0.01,
                "deterministic_mass_bound_satisfied_mean": 1.0,
                "output_bound_satisfied_mean": 1.0,
            },
            {
                "method": (
                    "production_pca48_int4k_int8q_"
                    "sampled_quantile_uncapped"
                ),
                "fraction": 0.04,
                "sampled_selected_fraction_mean": 0.041,
                "sampled_selected_fraction_p90": 0.055,
                "sampled_candidate_overflow_mean": 0.01,
                "sampled_threshold_absolute_error_mean": 0.03,
                "retained_attention_mass_mean": 0.895,
            },
        ]
    }
    qk_rows = []
    for model in ("llama31_8b", "qwen3_4b"):
        qk_rows.append(
            {
                "model": model,
                "key_effective_rank_mean": 25,
                "qk_effective_rank_mean": 20,
                "centered_qk_effective_rank_mean": 22,
                "qk_rank1_energy_fraction_mean": 0.75,
                "centered_qk_rank1_energy_fraction_mean": 0.55,
                "softmax_invariant_row_mean_energy_fraction_mean": 0.40,
                "qk_top_right_vector_constant_alignment_mean": 0.70,
                "centered_qk_energy_retained_optimal_rank16_mean": 0.8,
                "centered_qk_energy_retained_optimal_rank32_mean": 0.9,
                "centered_qk_energy_retained_optimal_rank48_mean": 0.95,
                "centered_qk_energy_retained_optimal_rank48_p10": 0.90,
                "centered_qk_energy_retained_uncentered_key_pca48_mean": 0.93,
                "centered_qk_uncentered_key_pca_optimality_gap_mean": 0.02,
                "centered_full_key_pca_qk_fidelity_mean": 0.93,
                "centered_key_pca_qk_fidelity_mean": 0.935,
                "centered_sampled_full_pca_qk_fidelity_mean": 0.925,
                "centered_production_prefix_pca_qk_fidelity_mean": 0.91,
                "centered_production_prefix_pca_qk_fidelity_p10": 0.75,
                    "centered_production_prefix_pca_qk_score_cosine_mean": 0.95,
                    "key_query_covariance_commutator_ratio_mean": 0.2,
                    "centered_key_query_covariance_commutator_ratio_mean": 0.1,
                    "sampled_full_pca_subspace_overlap_mean": 0.9,
                    "production_prefix_pca_subspace_overlap_mean": 0.85,
                "prefix512_pca_centered_qk_fidelity_mean": 0.80,
                "prefix1024_pca_centered_qk_fidelity_mean": 0.86,
                "prefix2048_pca_centered_qk_fidelity_mean": 0.91,
                "prefix4096_pca_centered_qk_fidelity_mean": 0.92,
                "prefix8192_pca_centered_qk_fidelity_mean": 0.925,
            }
        )
    fixed_rows = [
        {
            "model": model,
                "online_macro": 0.4,
                "fixed_macro": 0.39,
                "fixed_relative": 0.975,
                "fixed_minus_online_macro_ci95_low": -0.02,
                "fixed_minus_online_macro_ci95_high": 0.001,
                "prediction_agreement": 0.9,
            "fixed_index_build_speedup": 10,
            "fixed_total_speedup": 1.1,
        }
        for model in ("Llama-3.1-8B-Instruct", "Qwen3-4B-Instruct")
    ]
    budget_rows = [
        {
            "model": model,
            "samples": 64,
            "prompt_tokens_mean": 7000,
            "target_fraction_mean": 0.06,
            "actual_fraction_mean": 0.065,
            "actual_fraction_p95_mean": 0.07,
            "actual_fraction_max": 0.09,
            "actual_count_mean": 455,
            "actual_count_p95_mean": 490,
            "actual_count_max": 650,
            "candidate_overflow_head_fraction_mean": 0.01,
            "sampled_quantile_fallback_rate_mean": 0.0,
        }
        for model in ("Llama-3.1-8B-Instruct", "Qwen3-4B-Instruct")
    ]
    long_speed_rows = [
        {
            "model": model,
            "history_tokens": length,
            "paired_cases": 4,
            "full_ppl": 10.0,
            "direct_ppl": 10.2,
            "ppl_retention": 10.0 / 10.2,
            "actual_attention_tokens": 1400,
            "actual_attention_ratio": 1400 / length,
            "full_milliseconds_per_step": 200,
            "direct_milliseconds_per_step": 70,
            "decode_speedup": 200 / 70,
            "protocol_speedup": 1.5,
        }
        for model in ("llama31_8b", "qwen3_4b")
        for length in (64000, 128000)
    ]

    fixtures = {
        "comparison": comparison,
        "llama_logit": [logit],
        "qwen_logit": [logit],
        "crossing": crossing,
        "qk_spectrum": {"by_model": qk_rows},
        "fixed_basis": {"overall": fixed_rows},
        "actual_budget": {"overall": budget_rows},
        "long_speed": long_speed_rows,
    }
    paths = {}
    for name, payload in fixtures.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "report.md"
    arguments = ["report"]
    for name, path in paths.items():
        arguments.extend([f"--{name}", str(path)])
    arguments.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", arguments)

    main()

    text = output.read_text(encoding="utf-8")
    assert "sampled-quantile 实际消费审计" in text
    assert "中心化 rank-16" in text
    assert "softmax 精确抵消" in text
    assert "真实 first-2K basis" in text
    assert "Prefix 长度的纯数值消融" in text
    assert "当前请求首段 PCA 基与跨请求固定基" in text
    assert "64K/128K 冻结方法配对测速" in text
    assert "45.00" in text
    assert "Self-Indexing" in text
