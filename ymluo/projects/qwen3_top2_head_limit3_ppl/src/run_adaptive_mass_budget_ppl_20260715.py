from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from run_critical_position_budget_probe_20260715 import (  # noqa: E402
    load_model,
    parse_fractions,
    run_one_token,
)
from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    head_adaptive_mass_mode,
    head_qabs_sampled_mass_mode,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact full-QK diagnostic for per-layer/per-head attention-mass budgets."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topics", default="sports,medicine")
    parser.add_argument("--window_indices", default="0,1,2")
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--mass_thresholds", default="0.90,0.95,0.97,0.98,0.99")
    parser.add_argument("--budget_fractions", default="0.0025,0.005,0.01,0.02,0.04")
    parser.add_argument(
        "--mass_estimator", choices=["exact", "sampled_tail", "qabs_sampled_tail"], default="exact"
    )
    parser.add_argument("--sample_fraction", type=float, default=0.0025)
    parser.add_argument("--qabs_dim_count", type=int, default=8)
    parser.add_argument("--candidate_fraction", type=float, default=0.07)
    parser.add_argument("--qabs_use_cuda_kernels", action="store_true")
    parser.add_argument("--qabs_skip_candidate_rerank", action="store_true")
    parser.add_argument("--qabs_int2_onthefly", action="store_true")
    parser.add_argument("--qabs_early_layer_count", type=int, default=0)
    parser.add_argument("--qabs_early_budget_fraction", type=float, default=0.005)
    parser.add_argument(
        "--qabs_score_mode",
        choices=[
            "qabs",
            "pca_int4",
            "pca_int4_chunked",
            "pca_int4_chunked_logscale16",
            "pca_int4_chunked_logscale16_autosplit",
            "pca_int4_chunked_logscale16_qkmetric_autosplit",
            "pca_int4_chunked_logscale16_qkmetric_sampleq_autosplit",
            "pca_int4_chunked_logscale16_qkmetric_sampleq_dp4a_autosplit",
            "pca_int4_chunked_logscale16_qkmetric_sampleq1024_autosplit",
            "pca_int4_chunked_logscale16_qkmetric_dual_sampleq_autosplit",
            "pca_int4_chunked_logscale16_qkmetric_bitplane20_autosplit",
            "pca_int4_chunked_logscale16_qkmetric_momenttail256_autosplit",
            "pca_int4_qkmetric_microblock8_o24_autosplit",
            "pca_int4_qkmetric_microblock8_o32_autosplit",
            "pca_int4_qkmetric_microblock8_q8_o16_autosplit",
            "pca_int4_qkmetric_microblock8_q8_o20_autosplit",
            "pca_int4_qkmetric_microblock8_q8_o24_autosplit",
            "pca_int4_chunked_logscale16_split8",
            "pca_int4_chunked_logscale16_split16",
            "pca_int4_chunked_logscale16_tailvalue005",
            "pca_int4_chunked_logscale16_tailvalue005_shrink50",
            "pca_int4_chunked_logscale16_tailvalue005_shrink50_mass95",
            "pca_int4_chunked_logscale16_tailvalue005_reliability",
            "pca_int4_chunked_logscale16_lowfreq32_rescue005",
            "pca_int4_chunked_logscale16_lowfreq32_int2_union005_refresh4",
            "pca_int4_chunked_logscale16_lowfreq32_int2_oldest50_union005_refresh4",
            "pca_int4_sample_calibrated",
            "pca_int4_uncertainty_band",
            "pca_int4_partition_ucb",
            "pca_int4_partition_proxy_ucb",
            "pca_int4_partition_global_ucb",
            "pca_int4_partition_global_value_bound",
            "pca_int4_partition_global_contribution",
            "pca_int4_partition_global_delta8_r4",
            "pca_int4_partition_global_delta16_r4",
            "pca_int4_partition_global_delta16_r8",
            "pca_int4_partition_global_delta16tail_r4",
            "pca_int4_partition_global_delta16tail_r8",
            "pca_int4_partition_global_delta16spectral_t08",
            "pca_int4_partition_global_delta16spectral_t10",
            "pca_int4_partition_global_delta16spectral_top1",
            "pca_int4_partition_global_delta16spectral_top2",
            "pca_int4_partition_global_delta16spectral_top3",
            "pca_int4_partition_global_delta16bandef",
            "pca_int4_partition_global_delta16ec95",
            "pca_int4_partition_global_delta16ec95_budget",
            "pca_int4_partition_global_delta16density95",
            "pca_int4_partition_global_delta16density95_budget",
            "pca_int4_partition_global_delta16oneshot95_budget",
            "pca_int4_logscale16_partition_global_delta16oneshot95_budget",
            "pca_int4_logscale16_oneshot95_fixed2_autosplit",
            "pca_int4_logscale16_oneshot95_budget_autosplit",
            "pca_int4_partition_global_temporal2",
            "pca_int4_partition_global_qgate088",
            "pca_int4_partition_global_qgate092",
            "pca_int4_direct_uncertainty",
            "pca_int4_adaptive_rank",
            "pca_int4_progressive_cascade",
            "pca_int4_two_stage16",
            "pca_int4_two_stage32",
            "pca_int4_two_stage48",
            "pca_int4_residual_sentinel",
            "pca_hierarchical_842",
            "pca_hierarchical_841",
            "pca_hierarchical_841r50",
            "pca_hierarchical_841r25",
            "pca_hierarchical_841prog30",
            "pca_hierarchical_841sample20",
            "pca_hierarchical_841sample20n512",
            "pca_hierarchical_841sample20n1024",
            "pca_hierarchical_841sample20_proxy",
            "pca_hierarchical_841sample20_proxycal64",
            "pca_hierarchical_autokey10z",
            "pca_hierarchical_autokey12z",
            "pca_hierarchical_autokey14z",
            "pca_hierarchical_autokey16z",
            "pca_hierarchical_autokey18z",
            "pca_hierarchical_autoqmse10z",
            "pca_hierarchical_autoqmse12z",
            "pca_hierarchical_autoqmse14z",
            "pca_hierarchical_autoqmse16z",
            "pca_hierarchical_autoqmsetotal14z",
            "pca_hierarchical_autoqmsetotal15z",
            "pca_hierarchical_autoqmsetotal16z",
            "pca_hierarchical_autokey16",
            "pca_hierarchical_autokey18",
            "pca_hierarchical_autokey21",
            "pca_hierarchical_autokey26",
            "pca_hierarchical_842_stratified_mass",
            "pca_int8",
        ],
        default="qabs",
    )
    parser.add_argument("--qabs_projection_dim", type=int, default=64)
    parser.add_argument("--qabs_value_mass_threshold", type=float, default=1.0)
    parser.add_argument("--qabs_partition_ucb_z", type=float, default=1.64)
    parser.add_argument("--qabs_partition_overfetch_factor", type=int, default=0)
    parser.add_argument("--qabs_adaptive_rank_energy_threshold", type=float, default=0.85)
    parser.add_argument(
        "--qabs_adaptive_rank_residual_precision",
        choices=["int4", "int2_uniform4", "binary_sign"],
        default="int4",
    )
    parser.add_argument(
        "--qabs_gqa_candidate_mode",
        choices=["independent", "shared_max", "shared_zmax", "shared_mean"],
        default="independent",
    )
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--chunked_gqa_sdpa",
        action="store_true",
        help=(
            "Evaluate GQA SDPA one KV-head group at a time to avoid "
            "materializing every repeated KV head during long prefill."
        ),
    )
    return parser.parse_args()


def install_chunked_gqa_sdpa() -> None:
    from transformers.integrations.sdpa_attention import (
        sdpa_attention_forward as default_sdpa_attention_forward,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def chunked_gqa_sdpa_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        groups = int(getattr(module, "num_key_value_groups", 1))
        if groups <= 1:
            return default_sdpa_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                is_causal=is_causal,
                **kwargs,
            )
        if query.shape[1] != key.shape[1] * groups:
            raise ValueError("query/KV head counts do not match GQA groups")
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[..., : key.shape[-2]]
        if is_causal is None:
            is_causal = (
                query.shape[2] > 1
                and attention_mask is None
                and getattr(module, "is_causal", True)
            )

        outputs = []
        for kv_head in range(key.shape[1]):
            start = kv_head * groups
            stop = start + groups
            query_group = query[:, start:stop].contiguous()
            key_group = (
                key[:, kv_head : kv_head + 1]
                .expand(-1, groups, -1, -1)
                .contiguous()
            )
            value_group = (
                value[:, kv_head : kv_head + 1]
                .expand(-1, groups, -1, -1)
                .contiguous()
            )
            if attention_mask is None or attention_mask.shape[1] == 1:
                group_mask = attention_mask
            elif attention_mask.shape[1] == query.shape[1]:
                group_mask = attention_mask[:, start:stop]
            else:
                raise ValueError("unsupported attention-mask head dimension")
            outputs.append(
                F.scaled_dot_product_attention(
                    query_group,
                    key_group,
                    value_group,
                    attn_mask=group_mask,
                    dropout_p=dropout,
                    scale=scaling,
                    is_causal=bool(is_causal),
                )
            )
        return torch.cat(outputs, dim=1).transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS.register("sdpa", chunked_gqa_sdpa_forward)


def parse_thresholds(spec: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in spec.split(",") if item.strip()})
    if not thresholds or thresholds[0] <= 0.0 or thresholds[-1] > 1.0:
        raise ValueError("mass_thresholds must contain values in (0, 1]")
    return thresholds


def method_name(threshold: float, estimator: str = "exact") -> str:
    prefixes = {
        "exact": "adaptive_mass",
        "sampled_tail": "sampled_tail_mass",
        "qabs_sampled_tail": "qabs_sampled_tail_mass",
    }
    prefix = prefixes[estimator]
    return f"{prefix}_tau{threshold:g}".replace(".", "p")


def merge_position_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    selected = sum(float(row.get("selected_history_links", 0.0)) for row in rows)
    possible = sum(float(row.get("possible_history_links", 0.0)) for row in rows)
    output = {
        "attention_link_ratio": selected / max(1.0, possible),
        "selected_history_links": selected,
        "possible_history_links": possible,
    }
    mean_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("budget_") and key.endswith("_rate")
        }
    )
    for key in mean_keys:
        output[key] = sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))
    for key in [
        "retained_mass_mean",
        "selected_history_fraction_mean",
        "candidate_fraction_mean",
        "sampled_quantile_fallback_mean",
        "sampled_candidate_fraction_max_mean",
        "sampled_candidate_overflow_fraction_mean",
        "final_sampled_quantile_fallback_mean",
        "progressive_exact_qk_fraction_mean",
        "candidate_union_fraction_mean",
        "candidate_page16_storage_fraction_mean",
        "candidate_page64_storage_fraction_mean",
        "candidate_token_temporal_reuse_mean",
        "candidate_page16_temporal_reuse_mean",
        "candidate_page64_temporal_reuse_mean",
        "candidate_token_lru_3p2_hit_rate_mean",
        "candidate_token_temporal_reuse_prev2_mean",
        "candidate_token_temporal_reuse_prev3_mean",
        "candidate_token_temporal_reuse_prev4_mean",
        "candidate_token_working_set_current_prev1_fraction_mean",
        "candidate_token_working_set_current_prev2_fraction_mean",
        "candidate_token_working_set_current_prev3_fraction_mean",
        "candidate_token_working_set_current_prev4_fraction_mean",
        "temporal_reuse_rate_mean",
        "temporal_query_cosine_mean",
        "transport_refresh_rate_mean",
        "transport_delta_rank_mean",
        "transport_fixed_spectral_band_mean",
        "transport_spectral_threshold_mean",
        "transport_spectral_refresh_count_mean",
        "transport_spectral_refresh_fraction_mean",
        "transport_gate_signal_mean",
        "transport_scan_dimension_fraction_mean",
        "transport_scanned_bands_mean",
        "transport_expected_crossings_mean",
        "transport_tail_density_risk_mean",
        "transport_one_shot_risk_mean",
        "transport_risk_keep_fraction_mean",
        "transport_score_cache_fraction_mean",
        "auto_split_count_mean",
        "selected_projection_rank_mean",
        "selected_projection_rank_max",
        "selected_projection_coverage_mean",
    ]:
        output[key] = sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_stats(logits: torch.Tensor, label: int) -> tuple[float, dict[str, Any]]:
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    probabilities = log_probs.exp()
    top_probabilities, top_ids = torch.topk(probabilities, k=5, dim=-1)
    entropy = float(-(probabilities * log_probs).sum().item())
    vocabulary_size = int(log_probs.numel())
    top1_probability = float(top_probabilities[0].item())
    top2_probability = float(top_probabilities[1].item())
    return -float(log_probs[label].item()), {
        "logit_entropy": entropy,
        "logit_entropy_normalized": entropy / math.log(vocabulary_size),
        "logit_top1_probability": top1_probability,
        "logit_top2_probability": top2_probability,
        "logit_top1_top2_margin": top1_probability - top2_probability,
        "logit_top5_mass": float(top_probabilities.sum().item()),
        "provisional_top1_token_id": int(top_ids[0].item()),
    }


@torch.inference_mode()
def evaluate_threshold(
    model: torch.nn.Module,
    prefix_cache: Any,
    remote_count: int,
    query_ids: list[int],
    target_ids: list[int],
    threshold: float,
    fractions: tuple[float, ...],
    estimator: str,
    sample_fraction: float,
    qabs_dim_count: int,
    candidate_fraction: float,
    qabs_use_cuda_kernels: bool,
    qabs_skip_candidate_rerank: bool,
    qabs_int2_onthefly: bool,
    qabs_early_layer_count: int,
    qabs_early_budget_fraction: float,
    qabs_score_mode: str,
    qabs_projection_dim: int,
    qabs_gqa_candidate_mode: str,
    qabs_adaptive_rank_energy_threshold: float,
    qabs_adaptive_rank_residual_precision: str,
    qabs_value_mass_threshold: float,
    qabs_partition_ucb_z: float,
    qabs_partition_overfetch_factor: int,
    input_device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = prefix_cache
    previous_logits: torch.Tensor | None = None
    previous_stats: dict[str, float] = {}
    query_stats: list[dict[str, float]] = []
    target_stats: list[dict[str, float]] = []
    token_rows: list[dict[str, Any]] = []
    online_seconds = 0.0
    online_step_seconds: list[float] = []

    full_baseline_mode = (
        estimator == "exact"
        and len(fractions) == 1
        and fractions[0] >= 0.999999
        and threshold >= 0.999999
    )
    set_attention_implementation(
        model, "sdpa" if full_baseline_mode else "eager"
    )
    active_sample_fraction = sample_fraction if estimator != "exact" else None
    if full_baseline_mode:
        mode_context = nullcontext()
    elif estimator == "qabs_sampled_tail":
        mode_context = head_qabs_sampled_mass_mode(
            threshold,
            fractions,
            sample_fraction,
            qabs_dim_count,
            candidate_fraction,
            qabs_use_cuda_kernels,
            qabs_skip_candidate_rerank,
            qabs_int2_onthefly,
            qabs_early_layer_count,
            qabs_early_budget_fraction,
            qabs_score_mode,
            qabs_projection_dim,
            qabs_gqa_candidate_mode,
            qabs_adaptive_rank_energy_threshold,
            qabs_adaptive_rank_residual_precision,
            qabs_value_mass_threshold,
            qabs_partition_ucb_z,
            qabs_partition_overfetch_factor,
        )
    else:
        mode_context = head_adaptive_mass_mode(
            threshold, fractions, sample_fraction=active_sample_fraction
        )

    def full_attention_stats() -> dict[str, float]:
        return {
            "retained_mass_mean": 1.0,
            "retained_mass_min": 1.0,
            "retained_mass_p10": 1.0,
            "retained_mass_p25": 1.0,
            "retained_mass_lt90_fraction": 0.0,
            "retained_mass_lt95_fraction": 0.0,
            "retained_mass_lt99_fraction": 0.0,
            "retained_mass_early_mean": 1.0,
            "retained_mass_middle_mean": 1.0,
            "retained_mass_late_mean": 1.0,
            "top1_attention_mass_mean": 0.0,
            "selected_history_fraction_mean": 1.0,
            "selected_history_links": 1.0,
            "possible_history_links": 1.0,
            "attention_link_ratio": 1.0,
            "budget_100pct_rate": 1.0,
        }

    with mode_context:
        for offset, token_id in enumerate(query_ids):
            cache, previous_logits, seconds, previous_stats = run_one_token(
                model,
                token_id,
                cache,
                remote_count + offset,
                input_device,
                collect_attention_stats=not full_baseline_mode,
            )
            if full_baseline_mode:
                previous_stats = full_attention_stats()
            query_stats.append(previous_stats)
            online_seconds += seconds
            online_step_seconds.append(seconds)
        if previous_logits is None:
            raise RuntimeError("query_ids must be non-empty")

        label = int(target_ids[0])
        nll, logit_stats = prediction_stats(previous_logits, label)
        target_stats.append(previous_stats)
        token_rows.append(
            {
                "target_index": 0,
                "input_token_id": int(query_ids[-1]),
                "token_id": label,
                "nll": nll,
                **logit_stats,
                **previous_stats,
                "forward_seconds": online_step_seconds[-1],
            }
        )

        for target_index in range(len(target_ids) - 1):
            input_id = int(target_ids[target_index])
            cache, previous_logits, seconds, previous_stats = run_one_token(
                model,
                input_id,
                cache,
                remote_count + len(query_ids) + target_index,
                input_device,
                collect_attention_stats=not full_baseline_mode,
            )
            if full_baseline_mode:
                previous_stats = full_attention_stats()
            online_seconds += seconds
            online_step_seconds.append(seconds)
            label = int(target_ids[target_index + 1])
            nll, logit_stats = prediction_stats(previous_logits, label)
            target_stats.append(previous_stats)
            token_rows.append(
                {
                    "target_index": target_index + 1,
                    "input_token_id": input_id,
                    "token_id": label,
                    "nll": nll,
                    **logit_stats,
                    **previous_stats,
                    "forward_seconds": seconds,
                }
            )

    all_stats = query_stats + target_stats[1:]
    query_summary = merge_position_stats(query_stats)
    target_summary = merge_position_stats(target_stats)
    all_summary = merge_position_stats(all_stats)
    mean_nll = sum(float(row["nll"]) for row in token_rows) / len(token_rows)
    summary: dict[str, Any] = {
        "method": method_name(threshold, estimator),
        "mass_estimator": estimator,
        "sample_scoring_fraction": active_sample_fraction or 0.0,
        "qabs_dim_count": qabs_dim_count if estimator == "qabs_sampled_tail" else 0,
        "candidate_fraction": candidate_fraction if estimator == "qabs_sampled_tail" else 0.0,
        "qabs_use_cuda_kernels": bool(
            qabs_use_cuda_kernels and estimator == "qabs_sampled_tail"
        ),
        "qabs_skip_candidate_rerank": bool(
            qabs_skip_candidate_rerank and estimator == "qabs_sampled_tail"
        ),
        "qabs_int2_onthefly": bool(
            qabs_int2_onthefly and estimator == "qabs_sampled_tail"
        ),
        "qabs_early_layer_count": int(qabs_early_layer_count),
        "qabs_early_budget_fraction": float(qabs_early_budget_fraction),
        "qabs_score_mode": qabs_score_mode,
        "qabs_projection_dim": int(qabs_projection_dim),
        "qabs_adaptive_rank_energy_threshold": float(
            qabs_adaptive_rank_energy_threshold
        ),
        "qabs_adaptive_rank_residual_precision": (
            qabs_adaptive_rank_residual_precision
        ),
        "qabs_gqa_candidate_mode": qabs_gqa_candidate_mode,
        "qabs_value_mass_threshold": float(qabs_value_mass_threshold),
        "qabs_partition_ucb_z": float(qabs_partition_ucb_z),
        "qabs_partition_overfetch_factor": int(qabs_partition_overfetch_factor),
        "mass_threshold": threshold,
        "tokens": len(token_rows),
        "nll": mean_nll,
        "ppl": math.exp(min(20.0, mean_nll)),
        "online_seconds": online_seconds,
        "first_online_step_seconds": online_step_seconds[0],
        "steady_online_seconds_per_step": (
            sum(online_step_seconds[1:]) / (len(online_step_seconds) - 1)
            if len(online_step_seconds) > 1
            else online_step_seconds[0]
        ),
        "query_attention_link_ratio": query_summary["attention_link_ratio"],
        "target_attention_link_ratio": target_summary["attention_link_ratio"],
        "attention_link_ratio": all_summary["attention_link_ratio"],
        "target_retained_mass_mean": target_summary["retained_mass_mean"],
    }
    for key, value in all_summary.items():
        if (
            key.startswith("budget_")
            or key.startswith("candidate_")
            or key.startswith("progressive_exact_")
            or key.startswith("selected_projection_")
            or key.startswith("temporal_")
            or key.startswith("transport_")
            or key.startswith("auto_")
        ):
            summary[key] = value
    return token_rows, summary


def main() -> None:
    args = parse_args()
    if args.chunked_gqa_sdpa:
        install_chunked_gqa_sdpa()
    thresholds = parse_thresholds(args.mass_thresholds)
    fractions = tuple(parse_fractions(args.budget_fractions))
    topics = topic_names(args.topics)
    windows = parse_int_list(args.window_indices)
    if not (0 < args.query_tokens < args.history_tokens):
        raise ValueError("query_tokens must be in (0, history_tokens)")
    if args.mass_estimator != "exact" and not 0.0 < args.sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    required_tokens = max(windows) * args.window_stride_tokens + args.history_tokens + args.eval_tokens
    all_token_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

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
            bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)
            print(f"[case] topic={topic} window={window}", flush=True)

            for threshold in thresholds:
                set_attention_implementation(model, "sdpa")
                with head_top_fraction_mode(None), head_adaptive_mass_mode(None):
                    prefix_cache, prefill_seconds = lb.prefill_prefix(
                        model, bundle, input_device, args.prefill_chunk_tokens
                    )
                token_rows, summary = evaluate_threshold(
                    model,
                    prefix_cache,
                    len(remote_ids),
                    query_ids,
                    target_ids,
                    threshold,
                    fractions,
                    args.mass_estimator,
                    args.sample_fraction,
                    args.qabs_dim_count,
                    args.candidate_fraction,
                    args.qabs_use_cuda_kernels,
                    args.qabs_skip_candidate_rerank,
                    args.qabs_int2_onthefly,
                    args.qabs_early_layer_count,
                    args.qabs_early_budget_fraction,
                    args.qabs_score_mode,
                    args.qabs_projection_dim,
                    args.qabs_gqa_candidate_mode,
                    args.qabs_adaptive_rank_energy_threshold,
                    args.qabs_adaptive_rank_residual_precision,
                    args.qabs_value_mass_threshold,
                    args.qabs_partition_ucb_z,
                    args.qabs_partition_overfetch_factor,
                    input_device,
                )
                summary.update({"topic": topic, "window": window, "prefill_seconds": prefill_seconds})
                for row in token_rows:
                    row.update(
                        {
                            "topic": topic,
                            "window": window,
                            "method": summary["method"],
                            "mass_threshold": threshold,
                        }
                    )
                all_token_rows.extend(token_rows)
                summaries.append(summary)
                print(
                    f"  {summary['method']}: ppl={summary['ppl']:.4f} "
                    f"links={100.0 * summary['attention_link_ratio']:.3f}% "
                    f"online={summary['online_seconds']:.2f}s",
                    flush=True,
                )
                del prefix_cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                write_csv(args.output_dir / "token_results.csv", all_token_rows)
                write_csv(args.output_dir / "summary.csv", summaries)

    write_csv(args.output_dir / "token_results.csv", all_token_rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
