from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    pick_input_device,
    resolve_dtype,
)
from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    collect_head_top_fraction_stats,
    head_top_fraction_mode,
    install_llama_head_top_fraction_patch,
    parse_int_list,
    set_attention_implementation,
)
from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    TOPICS,
    encode_topic_stream,
    make_bundle,
    topic_names,
)


NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "hundred",
    "thousand",
    "million",
    "billion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-token critical-position diagnostics across exact head-wise attention budgets."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topics", default="sports,medicine")
    parser.add_argument("--window_indices", default="0,1,2")
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--top_fractions", default="0.0025,0.005,0.01,0.02,0.04")
    parser.add_argument("--include_full", action="store_true")
    parser.add_argument("--only_full", action="store_true")
    parser.add_argument("--collect_attention_stats", action="store_true")
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def parse_fractions(spec: str) -> list[float]:
    fractions = sorted({float(item.strip()) for item in spec.split(",") if item.strip()})
    if not fractions or fractions[0] <= 0.0 or fractions[-1] > 1.0:
        raise ValueError("top_fractions must contain values in (0, 1]")
    return fractions


def fraction_name(fraction: float | None) -> str:
    if fraction is None:
        return "full_attention"
    percent = 100.0 * fraction
    return f"head_top{percent:g}pct".replace(".", "p")


def token_shape(text: str) -> dict[str, int]:
    stripped = text.strip()
    lowered = stripped.lower()
    has_digit = any(char.isdigit() for char in stripped)
    is_number_word = lowered in NUMBER_WORDS
    has_alpha = any(char.isalpha() for char in stripped)
    is_punctuation = bool(stripped) and not any(char.isalnum() for char in stripped)
    return {
        "is_digit_token": int(has_digit),
        "is_number_word": int(is_number_word),
        "is_numeric_token": int(has_digit or is_number_word),
        "is_alpha_token": int(has_alpha),
        "is_punctuation_token": int(is_punctuation),
    }


def causal_logit_features(logits: torch.Tensor) -> dict[str, float | int]:
    row = logits[0].float()
    log_probs = F.log_softmax(row, dim=-1)
    top_values, top_indices = torch.topk(row, k=2, dim=-1)
    top1_id = int(top_indices[0].item())
    top1_logprob = float(log_probs[top1_id].item())
    probabilities = torch.exp(log_probs)
    entropy = float((-(probabilities * log_probs)).sum().item())
    return {
        "top1_id": top1_id,
        "top1_probability": math.exp(top1_logprob),
        "top1_top2_logit_margin": float((top_values[0] - top_values[1]).item()),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(max(2, int(row.numel()))),
    }


def logit_features(logits: torch.Tensor, label_id: int) -> dict[str, float | int]:
    features = causal_logit_features(logits)
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    features.update(
        {
            "nll": -float(log_probs[int(label_id)].item()),
            "label_probability": math.exp(float(log_probs[int(label_id)].item())),
            "top1_correct": int(int(features["top1_id"]) == int(label_id)),
        }
    )
    return features


@torch.inference_mode()
def run_one_token(
    model: torch.nn.Module,
    token_id: int,
    past_key_values: Any,
    position: int,
    input_device: torch.device,
    collect_attention_stats: bool = False,
    attention_record_sink: list[dict[str, Any]] | None = None,
) -> tuple[Any, torch.Tensor, float, dict[str, float]]:
    if input_device.type == "cuda":
        torch.cuda.synchronize(input_device)
    started = time.perf_counter()
    attention_records: list[dict[str, Any]] = []
    stats_context = (
        collect_head_top_fraction_stats(attention_records) if collect_attention_stats else nullcontext()
    )
    with stats_context:
        outputs = lb.model_forward(
            model,
            {
                "input_ids": torch.tensor([[int(token_id)]], dtype=torch.long, device=input_device),
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "cache_position": torch.tensor([position], dtype=torch.long, device=input_device),
            },
        )
    if input_device.type == "cuda":
        torch.cuda.synchronize(input_device)
    model_seconds = time.perf_counter() - started
    if attention_record_sink is not None:
        for record in attention_records:
            captured: dict[str, Any] = {
                "layer": int(record["layer"]),
            }
            for name in sorted(record):
                if not (
                    name.startswith("temporal_")
                    or name == "periodic_candidate_reuse_rate"
                ):
                    continue
                value = record.get(name)
                if isinstance(value, torch.Tensor):
                    captured[name] = value.detach().float().cpu()
                elif value is not None:
                    captured[name] = float(value)
            attention_record_sink.append(captured)
    attention_features = summarize_attention_records(attention_records) if collect_attention_stats else {}
    return (
        outputs.past_key_values,
        outputs.logits[:, -1, :].detach(),
        model_seconds,
        attention_features,
    )


def summarize_attention_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    records = sorted(records, key=lambda row: int(row["layer"]))
    summary: dict[str, float] = {}
    mass_records = [
        row
        for row in records
        if "retained_attention_mass" in row
        and "top1_attention_mass" in row
    ]
    if mass_records:
        retained = torch.stack(
            [
                row["retained_attention_mass"].float().cpu()
                for row in mass_records
            ],
            dim=0,
        )
        top1 = torch.stack(
            [
                row["top1_attention_mass"].float().cpu()
                for row in mass_records
            ],
            dim=0,
        )
        flat = retained.flatten()
        layer_count = int(retained.shape[0])
        first_end = max(1, layer_count // 3)
        second_end = max(first_end + 1, 2 * layer_count // 3)
        summary.update(
            {
                "retained_mass_mean": float(flat.mean().item()),
                "retained_mass_min": float(flat.min().item()),
                "retained_mass_p10": float(
                    torch.quantile(flat, 0.10).item()
                ),
                "retained_mass_p25": float(
                    torch.quantile(flat, 0.25).item()
                ),
                "retained_mass_lt90_fraction": float(
                    (flat < 0.90).float().mean().item()
                ),
                "retained_mass_lt95_fraction": float(
                    (flat < 0.95).float().mean().item()
                ),
                "retained_mass_lt99_fraction": float(
                    (flat < 0.99).float().mean().item()
                ),
                "retained_mass_early_mean": float(
                    retained[:first_end].mean().item()
                ),
                "retained_mass_middle_mean": float(
                    retained[first_end:second_end].mean().item()
                ),
                "retained_mass_late_mean": float(
                    retained[second_end:].mean().item()
                ),
                "top1_attention_mass_mean": float(top1.mean().item()),
            }
        )

    def summarize_tensor_metric(name: str) -> None:
        metric_records = [row for row in records if name in row]
        if not metric_records:
            return
        values = torch.stack(
            [row[name].float().cpu() for row in metric_records],
            dim=0,
        )
        flat_values = values.flatten()
        layer_count = int(values.shape[0])
        first_end = max(1, layer_count // 3)
        second_end = max(first_end + 1, 2 * layer_count // 3)
        summary.update(
            {
                f"{name}_mean": float(flat_values.mean().item()),
                f"{name}_p10": float(torch.quantile(flat_values, 0.10).item()),
                f"{name}_p50": float(torch.quantile(flat_values, 0.50).item()),
                f"{name}_p90": float(torch.quantile(flat_values, 0.90).item()),
                f"{name}_max": float(flat_values.max().item()),
                f"{name}_early_mean": float(values[:first_end].mean().item()),
                f"{name}_middle_mean": float(
                    values[first_end:second_end].mean().item()
                ),
                f"{name}_late_mean": float(values[second_end:].mean().item()),
            }
        )

    for tensor_metric in (
        "exact_boundary_gap",
        "exact_boundary_gap_over_score_std",
        "value_tail_defined",
        "value_selected_tail_gap_l2",
        "value_sparse_full_error_l2",
        "value_sparse_full_relative_error",
        "value_identity_relative_residual",
    ):
        summarize_tensor_metric(tensor_metric)
    if "selected_history_fraction" in records[0]:
        selected_fractions = torch.stack(
            [row["selected_history_fraction"].float().cpu() for row in records],
            dim=0,
        )
        selected_count_tensors = torch.stack(
            [row["selected_history_count"].float().cpu() for row in records],
            dim=0,
        )
        selected_budgets = torch.stack(
            [row["selected_budget_fraction"].float().cpu() for row in records],
            dim=0,
        )
        selected_counts = sum(
            float(row["selected_history_count"].float().sum().item()) for row in records
        )
        possible_counts = sum(
            float(int(row["history_count"]) * row["selected_history_count"].numel()) for row in records
        )
        summary.update(
            {
                "selected_history_fraction_mean": float(selected_fractions.mean().item()),
                "selected_history_fraction_p50": float(torch.quantile(selected_fractions, 0.50).item()),
                "selected_history_fraction_p90": float(torch.quantile(selected_fractions, 0.90).item()),
                "selected_history_fraction_p95": float(torch.quantile(selected_fractions, 0.95).item()),
                "selected_history_fraction_max": float(selected_fractions.max().item()),
                "selected_history_count_mean": float(selected_count_tensors.mean().item()),
                "selected_history_count_p95": float(torch.quantile(selected_count_tensors, 0.95).item()),
                "selected_history_count_max": float(selected_count_tensors.max().item()),
                "selected_history_links": selected_counts,
                "possible_history_links": possible_counts,
                "attention_link_ratio": selected_counts / max(1.0, possible_counts),
            }
        )
        for fraction in sorted(float(value) for value in torch.unique(selected_budgets)):
            name = f"budget_{100.0 * fraction:g}pct_rate".replace(".", "p")
            summary[name] = float((selected_budgets == fraction).float().mean().item())
    for name in (
        "candidate_fraction",
        "sampled_quantile_fallback",
        "sampled_candidate_fraction_max",
        "sampled_candidate_overflow_fraction",
        "packed_qmse_sample_count",
        "packed_qmse_index_bits_per_token",
        "packed_qmse_fused_query_prepare_requested",
        "packed_qmse_fused_query_prepare_executed",
        "packed_qmse_allocation_frozen_before_query",
        "packed_qmse_fixed_template_active",
        "packed_qmse_value_sketch_rank",
        "packed_qmse_value_sketch_bits",
        "packed_qmse_value_sketch_executed",
        "packed_qmse_value_sketch_tail_alpha",
        "packed_qmse_debug_value_sketch_disabled",
        "public_selector_index_bits_per_token",
        "public_selector_dimension_count",
        "public_selector_selected_page_count",
        "public_selector_is_page_granular",
        "public_selector_mean_value_correction",
        "public_selector_local_window",
        "public_selector_approximate_selected_mass",
        "final_sampled_quantile_fallback",
        "progressive_exact_qk_fraction",
        "candidate_top_gap",
        "candidate_boundary_gap",
        "candidate_temporal_stability",
        "candidate_union_fraction",
        "candidate_page16_storage_fraction",
        "candidate_page64_storage_fraction",
        "candidate_token_temporal_reuse",
        "candidate_page16_temporal_reuse",
        "candidate_page64_temporal_reuse",
        "candidate_token_lru_3p2_hit_rate",
        "candidate_token_temporal_reuse_prev2",
        "candidate_token_temporal_reuse_prev3",
        "candidate_token_temporal_reuse_prev4",
        "candidate_token_working_set_current_prev1_fraction",
        "candidate_token_working_set_current_prev2_fraction",
        "candidate_token_working_set_current_prev3_fraction",
        "candidate_token_working_set_current_prev4_fraction",
        "temporal_trace_available",
        "temporal_candidate_jaccard",
        "temporal_candidate_recall_from_previous",
        "temporal_candidate_recall_from_previous_with_new_tokens",
        "temporal_candidate_jaccard_with_new_tokens",
        "temporal_query_delta_norm",
        "temporal_key_norm_bound",
        "temporal_score_change_bound",
        "temporal_boundary_margin",
        "temporal_threshold_slack",
        "temporal_rejected_max_score",
        "temporal_certificate_safe",
        "temporal_certificate_layer_safe",
        "temporal_certificate_margin_ratio",
        "temporal_expected_score_delta_std",
        "temporal_core_margin_top25",
        "temporal_core_margin_ratio_top25",
        "temporal_core_recall_from_previous_top25",
        "temporal_expected_core_crossings",
        "temporal_expected_core_crossing_fraction",
        "temporal_reuse_output_trace_available",
        "temporal_reuse_output_relative_error",
        "temporal_reuse_output_cosine",
        "temporal_reuse_fresh_attention_mass",
        "temporal_reuse_output_error_le_1pct",
        "temporal_reuse_output_error_le_2pct",
        "temporal_reuse_mass_ge_95pct",
        "temporal_reuse_mass_ge_99pct",
        "temporal_sampled_reuse_mass_estimate",
        "temporal_sampled_reuse_mass_absolute_error",
        "temporal_gqa_union_candidate_fraction",
        "temporal_gqa_union_fresh_attention_mass",
        "temporal_gqa_union_output_relative_error",
        "temporal_gqa_union_output_error_le_1pct",
        "temporal_gqa_union_output_error_le_2pct",
        "temporal_sampled_safe_s90",
        "temporal_sampled_safe_s90_bad_output",
        "temporal_sampled_safe_s90_low_mass",
        "temporal_sampled_safe_s95",
        "temporal_sampled_safe_s95_bad_output",
        "temporal_sampled_safe_s95_low_mass",
        "temporal_sampled_safe_s97",
        "temporal_sampled_safe_s97_bad_output",
        "temporal_sampled_safe_s97_low_mass",
        "temporal_sampled_safe_s99",
        "temporal_sampled_safe_s99_bad_output",
        "temporal_sampled_safe_s99_low_mass",
        "temporal_expected_safe_r0p25",
        "temporal_expected_safe_r0p25_bad_output",
        "temporal_expected_safe_r0p25_low_mass",
        "temporal_expected_safe_r0p5",
        "temporal_expected_safe_r0p5_bad_output",
        "temporal_expected_safe_r0p5_low_mass",
        "temporal_expected_safe_r1",
        "temporal_expected_safe_r1_bad_output",
        "temporal_expected_safe_r1_low_mass",
        "temporal_expected_safe_r2",
        "temporal_expected_safe_r2_bad_output",
        "temporal_expected_safe_r2_low_mass",
        "temporal_core_safe_m0p5",
        "temporal_core_safe_m0p5_bad_output",
        "temporal_core_safe_m0p5_low_mass",
        "temporal_core_safe_m1",
        "temporal_core_safe_m1_bad_output",
        "temporal_core_safe_m1_low_mass",
        "temporal_core_safe_m2",
        "temporal_core_safe_m2_bad_output",
        "temporal_core_safe_m2_low_mass",
        "temporal_core_safe_m4",
        "temporal_core_safe_m4_bad_output",
        "temporal_core_safe_m4_low_mass",
        "temporal_mass_gate_reuse_rate",
        "temporal_mass_gate_estimated_mass",
        "temporal_mass_gate_candidate_fraction",
        "temporal_mass_gate_reused_output_relative_error",
        "temporal_mass_gate_bad_output",
        "temporal_mass_gate_cache_age",
        "periodic_candidate_reuse_rate",
        "temporal_reuse_rate",
        "temporal_query_cosine",
        "transport_refresh_rate",
        "transport_delta_rank",
        "transport_fixed_spectral_band",
        "transport_spectral_threshold",
        "transport_spectral_refresh_count",
        "transport_spectral_refresh_fraction",
        "transport_gate_signal",
        "transport_scan_dimension_fraction",
        "transport_scanned_bands",
        "transport_expected_crossings",
        "transport_tail_density_risk",
        "transport_one_shot_risk",
        "transport_risk_keep_fraction",
        "transport_score_cache_fraction",
        "auto_split_count",
    ):
        present_records = [row for row in records if name in row]
        if present_records:
            values = torch.stack(
                [torch.as_tensor(row[name]).float().cpu() for row in present_records],
                dim=0,
            )
            summary[f"{name}_mean"] = float(values.mean().item())
            summary[f"{name}_min"] = float(values.min().item())
    if "selected_projection_rank_mean" in records[0]:
        rank_means = torch.stack(
            [row["selected_projection_rank_mean"].float().cpu() for row in records]
        )
        rank_maxima = torch.stack(
            [row["selected_projection_rank_max"].float().cpu() for row in records]
        )
        coverage = torch.stack(
            [
                row["selected_projection_coverage_mean"].float().cpu()
                for row in records
            ]
        )
        summary.update(
            {
                "selected_projection_rank_mean": float(rank_means.mean().item()),
                "selected_projection_rank_max": float(rank_maxima.max().item()),
                "selected_projection_coverage_mean": float(coverage.mean().item()),
            }
        )
    return summary


def target_row(
    tokenizer: Any,
    topic: str,
    window: int,
    method: str,
    fraction: float | None,
    target_index: int,
    label_id: int,
    logits: torch.Tensor,
    history_counts: Counter[int],
    local_counts: Counter[int],
    attention_features: dict[str, float] | None = None,
) -> dict[str, Any]:
    token_text = tokenizer.decode(
        [int(label_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    top1_id = int(torch.argmax(logits[0].float()).item())
    top1_text = tokenizer.decode(
        [top1_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    row: dict[str, Any] = {
        "topic": topic,
        "window": window,
        "method": method,
        "top_fraction": "" if fraction is None else fraction,
        "target_index": target_index,
        "token_id": int(label_id),
        "token_text": token_text.replace("\n", "\\n").replace("\r", "\\r"),
        "history_token_frequency": int(history_counts[int(label_id)]),
        "local256_token_frequency": int(local_counts[int(label_id)]),
        "top1_text": top1_text.replace("\n", "\\n").replace("\r", "\\r"),
        "top1_history_frequency": int(history_counts[top1_id]),
    }
    row.update(token_shape(token_text))
    row.update(logit_features(logits, label_id))
    if attention_features:
        row.update(attention_features)
    return row


@torch.inference_mode()
def evaluate_fraction(
    model: torch.nn.Module,
    tokenizer: Any,
    prefix_cache: Any,
    remote_count: int,
    query_ids: list[int],
    target_ids: list[int],
    history_counts: Counter[int],
    topic: str,
    window: int,
    fraction: float | None,
    input_device: torch.device,
    collect_attention_stats: bool,
) -> tuple[list[dict[str, Any]], float]:
    method = fraction_name(fraction)
    if fraction is None:
        set_attention_implementation(model, "sdpa")
    else:
        set_attention_implementation(model, "eager")
    cache = prefix_cache
    previous_logits: torch.Tensor | None = None
    previous_attention_features: dict[str, float] = {}
    online_seconds = 0.0
    local_context = list(query_ids[-256:])
    local_counts: Counter[int] = Counter(local_context)

    with head_top_fraction_mode(fraction):
        for offset, token_id in enumerate(query_ids):
            cache, previous_logits, seconds, previous_attention_features = run_one_token(
                model,
                token_id,
                cache,
                remote_count + offset,
                input_device,
                collect_attention_stats=collect_attention_stats and fraction is not None,
            )
            online_seconds += seconds
        if previous_logits is None:
            raise RuntimeError("query_ids must be non-empty")

        rows = [
            target_row(
                tokenizer,
                topic,
                window,
                method,
                fraction,
                0,
                target_ids[0],
                previous_logits,
                history_counts,
                local_counts,
                previous_attention_features,
            )
        ]
        for target_index in range(len(target_ids) - 1):
            input_id = int(target_ids[target_index])
            cache, previous_logits, seconds, previous_attention_features = run_one_token(
                model,
                input_id,
                cache,
                remote_count + len(query_ids) + target_index,
                input_device,
                collect_attention_stats=collect_attention_stats and fraction is not None,
            )
            online_seconds += seconds
            local_context.append(input_id)
            local_counts[input_id] += 1
            if len(local_context) > 256:
                removed = local_context.pop(0)
                local_counts[removed] -= 1
                if local_counts[removed] <= 0:
                    del local_counts[removed]
            rows.append(
                target_row(
                    tokenizer,
                    topic,
                    window,
                    method,
                    fraction,
                    target_index + 1,
                    target_ids[target_index + 1],
                    previous_logits,
                    history_counts,
                    local_counts,
                    previous_attention_features,
                )
            )
    return rows, online_seconds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], runtimes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runtime_map = {
        (str(row["topic"]), int(row["window"]), str(row["method"])): row for row in runtimes
    }
    summary: list[dict[str, Any]] = []
    topics = ["all", *sorted({str(row["topic"]) for row in rows})]
    methods = sorted({str(row["method"]) for row in rows})
    for topic in topics:
        for method in methods:
            subset = [
                row for row in rows if row["method"] == method and (topic == "all" or row["topic"] == topic)
            ]
            if not subset:
                continue
            nll = sum(float(row["nll"]) for row in subset) / len(subset)
            keys = {(str(row["topic"]), int(row["window"]), method) for row in subset}
            seconds = sum(float(runtime_map[key]["online_seconds"]) for key in keys)
            summary.append(
                {
                    "topic": topic,
                    "method": method,
                    "cases": len(keys),
                    "tokens": len(subset),
                    "nll": nll,
                    "ppl": math.exp(min(20.0, nll)),
                    "mean_online_seconds_per_case": seconds / len(keys),
                    "top1_accuracy": sum(int(row["top1_correct"]) for row in subset) / len(subset),
                    "mean_top1_probability": sum(float(row["top1_probability"]) for row in subset) / len(subset),
                }
            )
    return summary


def load_model(args: argparse.Namespace) -> tuple[Any, torch.nn.Module, torch.device]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
    }
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    return tokenizer, model, pick_input_device(model, device)


def main() -> None:
    args = parse_args()
    topics = topic_names(args.topics)
    windows = parse_int_list(args.window_indices)
    fractions = [] if args.only_full else parse_fractions(args.top_fractions)
    methods: list[float | None] = [None] if args.only_full else fractions + ([None] if args.include_full else [])
    if not (0 < args.query_tokens < args.history_tokens):
        raise ValueError("query_tokens must be in (0, history_tokens)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    required_tokens = max(windows) * args.window_stride_tokens + args.history_tokens + args.eval_tokens
    token_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for topic in topics:
        stream = encode_topic_stream(
            tokenizer, TOPICS[topic], required_tokens, args.dataset_cache_dir, args.seed
        )
        for window in windows:
            start = window * args.window_stride_tokens
            history = stream[start : start + args.history_tokens]
            target_ids = stream[start + args.history_tokens : start + args.history_tokens + args.eval_tokens]
            remote_ids = history[: -args.query_tokens]
            query_ids = history[-args.query_tokens :]
            history_counts = Counter(history)
            bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)
            print(f"[case] topic={topic} window={window}", flush=True)

            for fraction in methods:
                method = fraction_name(fraction)
                set_attention_implementation(model, "sdpa")
                with head_top_fraction_mode(None):
                    prefix_cache, prefill_seconds = lb.prefill_prefix(
                        model, bundle, input_device, args.prefill_chunk_tokens
                    )
                rows, online_seconds = evaluate_fraction(
                    model,
                    tokenizer,
                    prefix_cache,
                    len(remote_ids),
                    query_ids,
                    target_ids,
                    history_counts,
                    topic,
                    window,
                    fraction,
                    input_device,
                    args.collect_attention_stats,
                )
                token_rows.extend(rows)
                runtime_rows.append(
                    {
                        "topic": topic,
                        "window": window,
                        "method": method,
                        "top_fraction": "" if fraction is None else fraction,
                        "prefill_seconds": prefill_seconds,
                        "online_seconds": online_seconds,
                    }
                )
                mean_nll = sum(float(row["nll"]) for row in rows) / len(rows)
                print(
                    f"  {method}: ppl={math.exp(min(20.0, mean_nll)):.4f} "
                    f"nll={mean_nll:.6f} online={online_seconds:.2f}s",
                    flush=True,
                )
                del prefix_cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                write_csv(args.output_dir / "token_results.csv", token_rows)
                write_csv(args.output_dir / "runtime_results.csv", runtime_rows)
                write_csv(args.output_dir / "summary.csv", summarize(token_rows, runtime_rows))

    summary = summarize(token_rows, runtime_rows)
    write_csv(args.output_dir / "token_results.csv", token_rows)
    write_csv(args.output_dir / "runtime_results.csv", runtime_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
