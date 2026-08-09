#!/usr/bin/env python
"""Independent all-layer PPL probe for the persistent joint K/V residual code."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from analyze_qaware_binarypc_blockmean_layer0_20260802 import (
    assert_numeric_backend_sane,
    binary_proxy_scores,
    encode_binary_principal,
    encode_joint_kv_residual_codebook,
    evenly_spaced_indices,
    fit_binary_principal_projection,
    fit_joint_kv_residual_codebook,
    quantize_log_error_norms,
    query_metric_factors,
)
from analyze_qmetric_global_holdout_layer0_20260802 import (
    fit_rvq_value_centroids,
    replan_exact_after_refinement,
    rerank_exact_after_refinement,
    rms_standardized_error_scale,
    rvq_tail_output,
    select_by_output_rms_bound,
    solve_three_action_rms_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train_texts", type=Path, nargs="+", required=True)
    parser.add_argument("--test_texts", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration_tokens", type=int, default=512)
    parser.add_argument("--history_tokens", type=int, default=512)
    parser.add_argument("--eval_tokens", type=int, default=16)
    parser.add_argument("--query_samples_per_text", type=int, default=128)
    parser.add_argument("--key_samples_per_text", type=int, default=256)
    parser.add_argument("--fraction", type=float, default=0.04)
    parser.add_argument("--recent_tokens", type=int, default=0)
    parser.add_argument("--sink_tokens", type=int, default=0)
    parser.add_argument(
        "--sparse_layers",
        default="all",
        help="all or comma-separated layer indices/ranges such as 0-6,14,20-27",
    )
    parser.add_argument(
        "--layer_fraction_overrides",
        default="",
        help="Comma-separated layer/rate rules such as 0-6:0.16,7-13:0.04",
    )
    parser.add_argument("--binary_bits", type=int, default=64)
    parser.add_argument("--projection_iterations", type=int, default=6)
    parser.add_argument("--residual_vq_bits", type=int, default=6)
    parser.add_argument("--residual_vq_iterations", type=int, default=6)
    parser.add_argument("--residual_binary_bits", type=int, default=0)
    parser.add_argument("--residual_binary_iterations", type=int, default=6)
    parser.add_argument(
        "--residual_binary_candidate_fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--candidate_selection",
        choices=("global", "warp_local"),
        default="global",
    )
    parser.add_argument("--warp_local_base_keep", type=int, default=8)
    parser.add_argument(
        "--warp_local_shortlist_multiplier", type=float, default=2.0
    )
    parser.add_argument(
        "--adaptive_error_tolerance",
        type=float,
        default=0.0,
        help="Positive values replace fixed remote fractions with an RMS bound.",
    )
    parser.add_argument(
        "--adaptive_action_mode",
        choices=(
            "full_residual",
            "joint_cost",
            "joint_cost_replan",
            "joint_cost_rerank",
        ),
        default="full_residual",
    )
    parser.add_argument("--residual_refinement_cost_bits", type=float, default=48.0)
    parser.add_argument("--exact_kv_cost_bits", type=float, default=4096.0)
    parser.add_argument(
        "--head_sensitivity_mode",
        choices=("none", "one_shot_hutchinson", "one_shot_fisher"),
        default="none",
    )
    parser.add_argument("--head_sensitivity_probes", type=int, default=2)
    parser.add_argument(
        "--score_error_calibration_samples",
        type=int,
        default=0,
        help="Zero disables; negative uses min(256, ceil(sqrt(remote tokens))).",
    )
    parser.add_argument(
        "--adaptive_budget_coupling",
        choices=("independent", "previous_layer_max", "previous_layer_quantile"),
        default="independent",
    )
    parser.add_argument(
        "--adaptive_budget_quantile",
        type=float,
        default=-1.0,
        help="Negative values use the leave-one-head-out quantile 1 - 1/H.",
    )
    parser.add_argument("--joint_value_weight", type=float, default=0.5)
    parser.add_argument("--risk_lambda", type=float, default=1.0)
    parser.add_argument(
        "--priority_mode",
        choices=("qk_risk", "output_bound"),
        default="qk_risk",
    )
    parser.add_argument("--risk_error_bits", type=int, default=4)
    parser.add_argument("--risk_error_block_size", type=int, default=256)
    parser.add_argument("--metric_shrinkage", default="oas")
    parser.add_argument("--value_mean_bits", type=int, default=4)
    parser.add_argument("--selected_conditioned_tail", action="store_true")
    parser.add_argument(
        "--tail_mode",
        choices=("joint_tail", "selected_only"),
        default="joint_tail",
    )
    parser.add_argument("--rebuild_index_each_step", action="store_true")
    parser.add_argument(
        "--refit_key_bits",
        type=int,
        default=0,
        help="Positive values refit request-local K residual centroids per joint ID.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "sdpa"),
        default="eager",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--codebook_cache", type=Path, default=None)
    parser.add_argument("--store_budget_records", action="store_true")
    parser.add_argument(
        "--one_shot_output_error_feedback",
        action="store_true",
        help=(
            "Return exact attention on the first decode step while measuring each "
            "head's sparse-output error, then tighten later analytic budgets by "
            "the measured error multiplier."
        ),
    )
    parser.add_argument(
        "--one_shot_feedback_norm",
        choices=(
            "head_relative",
            "projected_layer_global",
            "projected_layer_rss",
            "residual_stream_rss",
        ),
        default="head_relative",
        help="Normalization used by one-shot output-error feedback.",
    )
    parser.add_argument(
        "--one_shot_feedback_source",
        choices=("first_decode", "suffix_token"),
        default="first_decode",
        help=(
            "Use the first evaluated decode token as the dense probe, or use "
            "the final history token so every evaluated answer token is sparse."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def encode_text(tokenizer: Any, path: Path, tokens: int) -> torch.Tensor:
    text = path.read_text(encoding="utf-8", errors="ignore")
    ids = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=tokens,
    ).input_ids
    if ids.shape[-1] < tokens:
        raise ValueError(f"{path} has {ids.shape[-1]} tokens, need {tokens}")
    return ids[:, :tokens]


@torch.no_grad()
def extract_all_layer_qkv(
    model: Any,
    input_ids: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    outputs = model(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("model did not return hidden states")
    position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device)[None, :]
    position_embeddings = model.model.rotary_emb(hidden_states[0], position_ids)
    extracted = []
    for layer_index, layer in enumerate(model.model.layers):
        normalized = layer.input_layernorm(hidden_states[layer_index])
        batch, length, _ = normalized.shape
        attention = layer.self_attn
        query = attention.q_norm(
            attention.q_proj(normalized).view(
                batch, length, attention.config.num_attention_heads, attention.head_dim
            )
        ).transpose(1, 2)
        key = attention.k_norm(
            attention.k_proj(normalized).view(
                batch, length, attention.config.num_key_value_heads, attention.head_dim
            )
        ).transpose(1, 2)
        value = attention.v_proj(normalized).view(
            batch, length, attention.config.num_key_value_heads, attention.head_dim
        ).transpose(1, 2)
        query, key = apply_rotary_pos_emb(
            query, key, position_embeddings[0], position_embeddings[1]
        )
        extracted.append((query[0].float(), key[0].float(), value[0].float()))
    return extracted


@torch.no_grad()
def fit_codebooks(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    per_layer_samples: list[list[dict[str, list[torch.Tensor]]]] | None = None
    metadata: dict[str, Any] | None = None
    for path in args.train_texts:
        ids = encode_text(tokenizer, path, args.calibration_tokens).to(args.device)
        layers = extract_all_layer_qkv(model, ids)
        if per_layer_samples is None:
            query_heads = layers[0][0].shape[0]
            kv_heads = layers[0][1].shape[0]
            head_dim = layers[0][1].shape[-1]
            metadata = {
                "layers": len(layers),
                "query_heads": query_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "gqa_groups": query_heads // kv_heads,
            }
            per_layer_samples = [
                [{"q": [], "k": [], "v": []} for _ in range(kv_heads)]
                for _ in layers
            ]
        assert per_layer_samples is not None and metadata is not None
        key_indices = evenly_spaced_indices(
            args.calibration_tokens,
            args.key_samples_per_text,
            ids.device,
        )
        query_indices = evenly_spaced_indices(
            args.calibration_tokens,
            args.query_samples_per_text,
            ids.device,
        )
        group_size = int(metadata["gqa_groups"])
        head_dim = int(metadata["head_dim"])
        for layer_index, (query, key, value) in enumerate(layers):
            for kv_head in range(int(metadata["kv_heads"])):
                bucket = per_layer_samples[layer_index][kv_head]
                bucket["k"].append(key[kv_head].index_select(0, key_indices))
                bucket["v"].append(value[kv_head].index_select(0, key_indices))
                bucket["q"].append(
                    query[
                        kv_head * group_size : (kv_head + 1) * group_size
                    ]
                    .transpose(0, 1)
                    .index_select(0, query_indices)
                    .reshape(-1, head_dim)
                )
    if per_layer_samples is None or metadata is None:
        raise ValueError("no calibration text")

    codebooks = []
    for layer_index, layer_samples in enumerate(per_layer_samples):
        layer_codebooks = []
        for kv_head, samples in enumerate(layer_samples):
            queries = torch.cat(samples["q"])
            keys = torch.cat(samples["k"])
            values = torch.cat(samples["v"])
            query_factor, key_factor, shrinkage = query_metric_factors(
                queries, args.metric_shrinkage
            )
            metric_keys = keys @ key_factor
            projection = fit_binary_principal_projection(
                metric_keys,
                bits=args.binary_bits,
                iterations=args.projection_iterations,
                seed=20_000 + 100 * layer_index + kv_head,
            )
            codes, _ = encode_binary_principal(metric_keys, projection)
            residuals = metric_keys - codes @ projection
            joint_model = fit_joint_kv_residual_codebook(
                residuals,
                values,
                clusters=1 << args.residual_vq_bits,
                iterations=args.residual_vq_iterations,
                value_weight=args.joint_value_weight,
            )
            residual_projection = None
            if args.residual_binary_bits > 0:
                assignments, _, key_centroids = (
                    encode_joint_kv_residual_codebook(
                        residuals,
                        values,
                        joint_model,
                    )
                )
                second_residual = residuals - key_centroids.index_select(
                    0, assignments
                )
                residual_projection = fit_binary_principal_projection(
                    second_residual,
                    bits=args.residual_binary_bits,
                    iterations=args.residual_binary_iterations,
                    seed=(
                        40_000
                        + 100 * layer_index
                        + kv_head
                        + args.residual_binary_bits
                    ),
                )
            layer_codebooks.append(
                {
                    "query_factor": query_factor,
                    "key_factor": key_factor,
                    "projection": projection,
                    "joint_model": joint_model,
                    "residual_projection": residual_projection,
                    "shrinkage": shrinkage,
                }
            )
        codebooks.append(layer_codebooks)
    return codebooks, metadata


def output_error_feedback_multipliers(
    full_heads: torch.Tensor,
    sparse_heads: torch.Tensor,
    output_weight: torch.Tensor,
    residual_stream_norm: torch.Tensor | float,
    tolerances: torch.Tensor,
    mode: str,
) -> dict[str, torch.Tensor | float]:
    """Calibrate analytic budgets against error that reaches the residual stream."""
    if full_heads.ndim != 2 or sparse_heads.shape != full_heads.shape:
        raise ValueError("full and sparse head outputs must be aligned matrices")
    head_count, head_dim = full_heads.shape
    if output_weight.ndim != 2 or output_weight.shape[1] != head_count * head_dim:
        raise ValueError("output projection does not match concatenated heads")
    if tolerances.shape != (head_count,) or bool((tolerances <= 0).any()):
        raise ValueError("one positive tolerance is required per query head")
    if mode not in {
        "head_relative",
        "projected_layer_global",
        "projected_layer_rss",
        "residual_stream_rss",
    }:
        raise ValueError(f"unknown one-shot feedback normalization: {mode}")

    full_heads = full_heads.float()
    sparse_heads = sparse_heads.float()
    output_weight = output_weight.float()
    errors = sparse_heads - full_heads
    full_projected = F.linear(full_heads.reshape(-1), output_weight)
    projected_error = F.linear(errors.reshape(-1), output_weight)
    projected_layer_norm = full_projected.norm().clamp_min(1.0e-8)
    projected_layer_relative = projected_error.norm() / projected_layer_norm

    projected_head_errors = []
    for head in range(head_count):
        start = head * head_dim
        stop = start + head_dim
        projected_head_errors.append(
            F.linear(errors[head], output_weight[:, start:stop]).norm()
        )
    projected_head_errors_tensor = torch.stack(projected_head_errors)

    if mode == "head_relative":
        error_norms = errors.norm(dim=-1)
        normalizers = full_heads.norm(dim=-1).clamp_min(1.0e-8)
    elif mode == "projected_layer_global":
        error_norms = projected_error.norm().expand(head_count)
        normalizers = projected_layer_norm.expand(head_count)
    elif mode == "projected_layer_rss":
        error_norms = projected_head_errors_tensor
        normalizers = (
            projected_layer_norm / math.sqrt(head_count)
        ).expand(head_count)
    else:
        error_norms = projected_head_errors_tensor
        residual_norm = torch.as_tensor(
            residual_stream_norm,
            device=full_heads.device,
            dtype=torch.float32,
        ).clamp_min(1.0e-8)
        normalizers = (residual_norm / math.sqrt(head_count)).expand(head_count)

    multipliers = torch.maximum(
        torch.ones_like(error_norms),
        error_norms / (tolerances.float() * normalizers),
    )
    return {
        "multipliers": multipliers,
        "projected_error_l2": projected_head_errors_tensor,
        "normalizer_l2": normalizers,
        "projected_layer_relative_l2": float(projected_layer_relative),
    }


def augment_selected_indices(
    selected: torch.Tensor,
    priorities: torch.Tensor,
    target_count: int,
) -> torch.Tensor:
    """Add the highest-priority unselected indices up to a target cardinality."""
    if selected.ndim != 1 or priorities.ndim != 1:
        raise ValueError("selected indices and priorities must be vectors")
    if target_count < selected.numel() or target_count > priorities.numel():
        raise ValueError("target count must contain the existing selection")
    if selected.numel() and (
        int(selected.min()) < 0 or int(selected.max()) >= priorities.numel()
    ):
        raise ValueError("selected index is out of bounds")
    if selected.unique().numel() != selected.numel():
        raise ValueError("selected indices must be unique")
    if target_count == selected.numel():
        return selected
    available = torch.ones(
        priorities.numel(), dtype=torch.bool, device=priorities.device
    )
    available[selected] = False
    masked = priorities.float().masked_fill(~available, float("-inf"))
    additions = torch.topk(
        masked,
        target_count - selected.numel(),
        sorted=False,
    ).indices
    return torch.cat((selected, additions))


def warp_local_refined_topk(
    base_priority: torch.Tensor,
    refined_priority: torch.Tensor,
    keep_count: int,
    *,
    base_keep: int,
    shortlist_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Emulate the CUDA warp-local candidate hierarchy for quality probes."""
    if base_priority.ndim != 1 or refined_priority.shape != base_priority.shape:
        raise ValueError("base and refined priorities must be aligned vectors")
    token_count = int(base_priority.numel())
    if not 0 < keep_count <= token_count:
        raise ValueError("keep count must lie in the priority-vector range")
    if not 1 <= base_keep <= 32:
        raise ValueError("warp-local base keep must lie in [1,32]")
    if shortlist_multiplier < 1.0:
        raise ValueError("shortlist multiplier must be at least one")

    padded_tokens = math.ceil(token_count / 32) * 32
    padded_base = torch.nn.functional.pad(
        base_priority.float(),
        (0, padded_tokens - token_count),
        value=-torch.inf,
    )
    base_values, base_lanes = torch.topk(
        padded_base.reshape(-1, 32),
        k=base_keep,
        dim=1,
        sorted=False,
    )
    warp_offsets = (
        torch.arange(base_lanes.shape[0], device=base_lanes.device)[:, None] * 32
    )
    candidates = (base_lanes + warp_offsets).reshape(-1)
    valid_candidates = torch.isfinite(base_values.reshape(-1))
    candidates = candidates[valid_candidates]
    candidate_scores = refined_priority.float().index_select(0, candidates)

    candidate_warps = math.ceil(candidates.numel() / 32)
    target_shortlist = min(
        candidates.numel(), math.ceil(shortlist_multiplier * keep_count)
    )
    required_per_warp = math.ceil(target_shortlist / candidate_warps)
    residual_keep = max(4, min(32, 1 << (required_per_warp - 1).bit_length()))
    padded_candidates = candidate_warps * 32
    padded_scores = torch.nn.functional.pad(
        candidate_scores,
        (0, padded_candidates - candidates.numel()),
        value=-torch.inf,
    )
    padded_indices = torch.nn.functional.pad(
        candidates, (0, padded_candidates - candidates.numel()), value=0
    )
    shortlist_values, shortlist_lanes = torch.topk(
        padded_scores.reshape(-1, 32),
        k=residual_keep,
        dim=1,
        sorted=False,
    )
    shortlist_indices = torch.gather(
        padded_indices.reshape(-1, 32), 1, shortlist_lanes
    ).reshape(-1)
    shortlist_values = shortlist_values.reshape(-1)
    valid_shortlist = torch.isfinite(shortlist_values)
    shortlist_indices = shortlist_indices[valid_shortlist]
    shortlist_values = shortlist_values[valid_shortlist]
    if shortlist_indices.numel() < keep_count:
        raise RuntimeError("warp-local shortlist is smaller than the final budget")
    final_within = torch.topk(
        shortlist_values, keep_count, sorted=False
    ).indices
    return (
        shortlist_indices.index_select(0, final_within),
        candidates,
        residual_keep,
    )


def install_sparse_attention(
    model: Any,
    codebooks: list[list[dict[str, Any]]],
    args: argparse.Namespace,
    head_sensitivities: torch.Tensor | None = None,
) -> tuple[list[tuple[Any, Any]], list[dict[str, float | int]]]:
    originals = []
    budget_records: list[dict[str, float | int]] = []
    sparse_layers = parse_layer_spec(args.sparse_layers, len(model.model.layers))
    resolved_budget_quantile = (
        args.adaptive_budget_quantile
        if args.adaptive_budget_quantile >= 0.0
        else 1.0 - 1.0 / model.config.num_attention_heads
    )
    fraction_overrides = parse_layer_fraction_overrides(
        args.layer_fraction_overrides, len(model.model.layers)
    )
    for layer_index, layer in enumerate(model.model.layers):
        if layer_index not in sparse_layers:
            continue
        attention = layer.self_attn
        original = attention.forward
        originals.append((attention, original))
        layer_fraction = fraction_overrides.get(layer_index, args.fraction)
        runtime_indexes: dict[int, dict[str, torch.Tensor | int]] = {}
        runtime_budget_state: dict[str, Any] = {
            "previous_max_required_fraction": None,
            "previous_required_quantile": None,
            "output_error_multipliers": {},
            "output_error_calibrated": False,
        }

        def sparse_forward(
            self: Any,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_value: Any = None,
            cache_position: torch.Tensor | None = None,
            _layer_index: int = layer_index,
            _fraction: float = layer_fraction,
            _original: Any = original,
            _runtime_indexes: dict[int, dict[str, torch.Tensor | int]] = runtime_indexes,
            _runtime_budget_state: dict[str, Any] = runtime_budget_state,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            if hidden_states.shape[1] != 1 or hidden_states.shape[0] != 1:
                return _original(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    **kwargs,
                )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(
                self.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            key_states = self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )
            query_heads = query_states.shape[1]
            kv_heads = key_states.shape[1]
            group_size = query_heads // kv_heads
            history_count = key_states.shape[-2]
            recent_count = min(args.recent_tokens, history_count)
            sink_count = min(args.sink_tokens, history_count - recent_count)
            remote_start = sink_count
            remote_stop = history_count - recent_count
            remote_count = max(0, remote_stop - remote_start)
            fixed_keep_remote = min(
                remote_count,
                max(1, math.ceil(remote_count * _fraction))
                if remote_count
                else 0,
            )
            head_outputs = []
            probe_full_heads: list[torch.Tensor] = []
            probe_sparse_heads: list[torch.Tensor] = []
            probe_effective_tolerances: list[float] = []
            probe_records: list[dict[str, float | int]] = []
            one_shot_probe = (
                args.one_shot_output_error_feedback
                and not bool(_runtime_budget_state["output_error_calibrated"])
            )
            current_max_required_fraction = 0.0
            current_required_fractions: list[float] = []
            for kv_head in range(kv_heads):
                state = codebooks[_layer_index][kv_head]
                head_key = key_states[0, kv_head].float()
                head_value = value_states[0, kv_head].float()
                runtime = _runtime_indexes.get(kv_head)
                can_append = (
                    not args.rebuild_index_each_step
                    and runtime is not None
                    and int(runtime["length"]) + 1 == history_count
                )
                if can_append:
                    assert runtime is not None
                    new_metric_key = head_key[-1:] @ state["key_factor"]
                    new_codes, _ = encode_binary_principal(
                        new_metric_key, state["projection"]
                    )
                    new_residual = new_metric_key - new_codes @ state["projection"]
                    new_assignments, _, key_centroids = (
                        encode_joint_kv_residual_codebook(
                            new_residual,
                            head_value[-1:],
                            state["joint_model"],
                        )
                    )
                    codes = torch.cat((runtime["codes"], new_codes))
                    metric_residual = torch.cat(
                        (runtime["metric_residual"], new_residual)
                    )
                    assignments = torch.cat(
                        (runtime["assignments"], new_assignments)
                    )
                else:
                    metric_keys = head_key @ state["key_factor"]
                    codes, _ = encode_binary_principal(
                        metric_keys, state["projection"]
                    )
                    reconstruction = codes @ state["projection"]
                    metric_residual = metric_keys - reconstruction
                    assignments, _, key_centroids = (
                        encode_joint_kv_residual_codebook(
                            metric_residual,
                            head_value,
                            state["joint_model"],
                        )
                    )
                _runtime_indexes[kv_head] = {
                    "length": history_count,
                    "codes": codes,
                    "metric_residual": metric_residual,
                    "assignments": assignments,
                }
                if args.refit_key_bits > 0:
                    key_centroids, _ = fit_rvq_value_centroids(
                        metric_residual,
                        assignments,
                        1 << args.residual_vq_bits,
                        args.refit_key_bits,
                    )
                    errors = (
                        metric_residual
                        - key_centroids.float().index_select(0, assignments)
                    ).norm(dim=-1)
                else:
                    errors = (
                        metric_residual
                        - key_centroids.float().index_select(0, assignments)
                    ).norm(dim=-1)
                quantized_errors, _ = quantize_log_error_norms(
                    errors,
                    args.risk_error_bits,
                    args.risk_error_block_size,
                )
                residual_projection = state.get("residual_projection")
                residual_codes = None
                quantized_second_errors = None
                if isinstance(residual_projection, torch.Tensor):
                    second_residual = (
                        metric_residual
                        - key_centroids.float().index_select(0, assignments)
                    )
                    residual_codes, second_errors = encode_binary_principal(
                        second_residual,
                        residual_projection,
                    )
                    quantized_second_errors, _ = quantize_log_error_norms(
                        second_errors,
                        args.risk_error_bits,
                        args.risk_error_block_size,
                    )
                value_centroids, _ = fit_rvq_value_centroids(
                    head_value,
                    assignments,
                    1 << args.residual_vq_bits,
                    args.value_mean_bits,
                )
                value_errors, _ = quantize_log_error_norms(
                    (
                        head_value
                        - value_centroids.float().index_select(0, assignments)
                    ).norm(dim=-1),
                    args.risk_error_bits,
                    args.risk_error_block_size,
                )
                centroid_norms = value_centroids.float().norm(dim=-1)
                for group_offset in range(group_size):
                    query_head = kv_head * group_size + group_offset
                    head_sensitivity = (
                        float(head_sensitivities[_layer_index, query_head])
                        if head_sensitivities is not None
                        else 1.0
                    )
                    effective_tolerance = min(
                        0.999,
                        args.adaptive_error_tolerance
                        / max(head_sensitivity, 1.0e-6)
                        / max(
                            float(
                                _runtime_budget_state[
                                    "output_error_multipliers"
                                ].get(query_head, 1.0)
                            ),
                            1.0e-6,
                        ),
                    )
                    query = query_states[0, query_head, 0].float()
                    metric_query = query @ state["query_factor"]
                    principal_proxy = binary_proxy_scores(
                        codes,
                        state["projection"],
                        metric_query,
                        self.scaling,
                    )
                    residual_table = key_centroids @ metric_query * self.scaling
                    joint_proxy = principal_proxy + residual_table.index_select(
                        0, assignments
                    )
                    base_uncertainty = (
                        quantized_errors * metric_query.norm() / float(self.head_dim)
                    )
                    if args.priority_mode == "output_bound":
                        base_sensitivity = (
                            value_errors
                            + base_uncertainty
                            * (
                                centroid_norms.index_select(0, assignments)
                                + value_errors
                            )
                        ).clamp_min(1.0e-8)
                        base_priority = joint_proxy + base_sensitivity.log()
                    else:
                        base_priority = (
                            joint_proxy + args.risk_lambda * base_uncertainty
                        )
                    refined_proxy = joint_proxy
                    refined_uncertainty = base_uncertainty
                    refined_priority = base_priority
                    if (
                        isinstance(residual_projection, torch.Tensor)
                        and isinstance(residual_codes, torch.Tensor)
                        and isinstance(quantized_second_errors, torch.Tensor)
                    ):
                        refined_proxy = joint_proxy + binary_proxy_scores(
                            residual_codes,
                            residual_projection,
                            metric_query,
                            self.scaling,
                        )
                        refined_uncertainty = (
                            quantized_second_errors
                            * metric_query.norm()
                            / float(self.head_dim)
                        )
                        if args.priority_mode == "output_bound":
                            refined_sensitivity = (
                                value_errors
                                + refined_uncertainty
                                * (
                                    centroid_norms.index_select(
                                        0, assignments
                                    )
                                    + value_errors
                                )
                            ).clamp_min(1.0e-8)
                            refined_priority = (
                                refined_proxy + refined_sensitivity.log()
                            )
                        else:
                            refined_priority = (
                                refined_proxy
                                + args.risk_lambda * refined_uncertainty
                            )
                    exact_scores = head_key @ query * self.scaling
                    base_error_scale = 1.0
                    refined_error_scale = 1.0
                    calibration_samples = 0
                    if args.score_error_calibration_samples != 0 and remote_count:
                        calibration_samples = min(
                            remote_count,
                            (
                                min(256, math.ceil(math.sqrt(remote_count)))
                                if args.score_error_calibration_samples < 0
                                else args.score_error_calibration_samples
                            ),
                        )
                        sample_relative = evenly_spaced_indices(
                            remote_count,
                            calibration_samples,
                            exact_scores.device,
                        )
                        sample = sample_relative + remote_start

                        base_error_scale = rms_standardized_error_scale(
                            exact_scores,
                            joint_proxy,
                            base_uncertainty,
                            sample,
                        )
                        refined_error_scale = rms_standardized_error_scale(
                            exact_scores,
                            refined_proxy,
                            refined_uncertainty,
                            sample,
                        )
                        base_uncertainty = base_uncertainty * base_error_scale
                        refined_uncertainty = (
                            refined_uncertainty * refined_error_scale
                        )
                        if args.priority_mode == "output_bound":
                            base_sensitivity = (
                                value_errors
                                + base_uncertainty
                                * (
                                    centroid_norms.index_select(0, assignments)
                                    + value_errors
                                )
                            ).clamp_min(1.0e-8)
                            refined_sensitivity = (
                                value_errors
                                + refined_uncertainty
                                * (
                                    centroid_norms.index_select(0, assignments)
                                    + value_errors
                                )
                            ).clamp_min(1.0e-8)
                            base_priority = joint_proxy + base_sensitivity.log()
                            refined_priority = (
                                refined_proxy + refined_sensitivity.log()
                            )
                        else:
                            base_priority = (
                                joint_proxy + args.risk_lambda * base_uncertainty
                            )
                            refined_priority = (
                                refined_proxy
                                + args.risk_lambda * refined_uncertainty
                            )
                    keep_remote = fixed_keep_remote
                    adaptive_diagnostics: dict[str, float] | None = None
                    adaptive_remote: torch.Tensor | None = None
                    refined_remote: torch.Tensor | None = None
                    refined_remote_count = (
                        remote_count
                        if isinstance(residual_projection, torch.Tensor)
                        and args.residual_binary_candidate_fraction >= 1.0
                        else 0
                    )
                    tail_proxy = refined_proxy
                    if args.adaptive_error_tolerance > 0.0 and remote_count:
                        approximate_values = value_centroids.float().index_select(
                            0, assignments
                        )
                        hybrid_priority: torch.Tensor | None = None
                        if args.adaptive_action_mode in (
                            "joint_cost",
                            "joint_cost_replan",
                            "joint_cost_rerank",
                        ):
                            if not isinstance(residual_projection, torch.Tensor):
                                raise ValueError(
                                    "joint-cost control requires a residual score code"
                                )
                            refined_remote, adaptive_remote, adaptive_diagnostics = (
                                solve_three_action_rms_budget(
                                    joint_proxy[remote_start:remote_stop],
                                    base_uncertainty[remote_start:remote_stop],
                                    refined_uncertainty[remote_start:remote_stop],
                                    approximate_values[remote_start:remote_stop],
                                    value_errors[remote_start:remote_stop],
                                    effective_tolerance,
                                    args.residual_refinement_cost_bits,
                                    args.exact_kv_cost_bits,
                                )
                            )
                            if args.adaptive_action_mode == "joint_cost_replan":
                                (
                                    adaptive_remote,
                                    hybrid_proxy,
                                    hybrid_uncertainty,
                                    replan_diagnostics,
                                ) = replan_exact_after_refinement(
                                    joint_proxy[remote_start:remote_stop],
                                    refined_proxy[remote_start:remote_stop],
                                    base_uncertainty[remote_start:remote_stop],
                                    refined_uncertainty[remote_start:remote_stop],
                                    approximate_values[remote_start:remote_stop],
                                    value_errors[remote_start:remote_stop],
                                    effective_tolerance,
                                    refined_remote,
                                )
                                hybrid_sensitivity = (
                                    value_errors[remote_start:remote_stop]
                                    + hybrid_uncertainty
                                    * (
                                        centroid_norms.index_select(
                                            0,
                                            assignments[remote_start:remote_stop],
                                        )
                                        + value_errors[remote_start:remote_stop]
                                    )
                                ).clamp_min(1.0e-8)
                                hybrid_priority = (
                                    hybrid_proxy + hybrid_sensitivity.log()
                                )
                                adaptive_diagnostics.update(
                                    {
                                        f"replan_{key}": value
                                        for key, value in replan_diagnostics.items()
                                    }
                                )
                            elif args.adaptive_action_mode == "joint_cost_rerank":
                                (
                                    adaptive_remote,
                                    hybrid_proxy,
                                    hybrid_uncertainty,
                                    rerank_diagnostics,
                                ) = rerank_exact_after_refinement(
                                    joint_proxy[remote_start:remote_stop],
                                    refined_proxy[remote_start:remote_stop],
                                    base_uncertainty[remote_start:remote_stop],
                                    refined_uncertainty[remote_start:remote_stop],
                                    approximate_values[remote_start:remote_stop],
                                    value_errors[remote_start:remote_stop],
                                    refined_remote,
                                    int(adaptive_remote.numel()),
                                )
                                hybrid_sensitivity = (
                                    value_errors[remote_start:remote_stop]
                                    + hybrid_uncertainty
                                    * (
                                        centroid_norms.index_select(
                                            0,
                                            assignments[remote_start:remote_stop],
                                        )
                                        + value_errors[remote_start:remote_stop]
                                    )
                                ).clamp_min(1.0e-8)
                                hybrid_priority = (
                                    hybrid_proxy + hybrid_sensitivity.log()
                                )
                                adaptive_diagnostics.update(
                                    {
                                        f"rerank_{key}": value
                                        for key, value in rerank_diagnostics.items()
                                    }
                                )
                        else:
                            adaptive_remote, adaptive_diagnostics = (
                                select_by_output_rms_bound(
                                    refined_proxy[remote_start:remote_stop],
                                    refined_uncertainty[remote_start:remote_stop],
                                    approximate_values[remote_start:remote_stop],
                                    value_errors[remote_start:remote_stop],
                                    effective_tolerance,
                                )
                            )
                        required_keep_remote = int(adaptive_remote.numel())
                        required_fraction = required_keep_remote / remote_count
                        current_floor = current_max_required_fraction
                        previous_floor = _runtime_budget_state[
                            "previous_max_required_fraction"
                        ]
                        if previous_floor is not None:
                            current_floor = max(current_floor, previous_floor)
                        keep_remote = required_keep_remote
                        if args.adaptive_budget_coupling == "previous_layer_max":
                            keep_remote = min(
                                remote_count,
                                max(
                                    required_keep_remote,
                                    math.ceil(remote_count * current_floor),
                                ),
                            )
                            if keep_remote > required_keep_remote:
                                remote_priority = (
                                    refined_priority[remote_start:remote_stop]
                                    if args.adaptive_action_mode == "full_residual"
                                    else (
                                        hybrid_priority
                                        if hybrid_priority is not None
                                        else base_priority[remote_start:remote_stop]
                                    )
                                )
                                adaptive_remote = augment_selected_indices(
                                    adaptive_remote,
                                    remote_priority,
                                    keep_remote,
                                )
                        elif (
                            args.adaptive_budget_coupling
                            == "previous_layer_quantile"
                            and _runtime_budget_state["previous_required_quantile"]
                            is not None
                        ):
                            keep_remote = min(
                                remote_count,
                                max(
                                    required_keep_remote,
                                    math.ceil(
                                        remote_count
                                        * float(
                                            _runtime_budget_state[
                                                "previous_required_quantile"
                                            ]
                                        )
                                    ),
                                ),
                            )
                            if keep_remote > required_keep_remote:
                                remote_priority = (
                                    refined_priority[remote_start:remote_stop]
                                    if args.adaptive_action_mode == "full_residual"
                                    else (
                                        hybrid_priority
                                        if hybrid_priority is not None
                                        else base_priority[remote_start:remote_stop]
                                    )
                                )
                                adaptive_remote = augment_selected_indices(
                                    adaptive_remote,
                                    remote_priority,
                                    keep_remote,
                                )
                        if refined_remote is not None:
                            if (
                                args.adaptive_action_mode
                                not in ("joint_cost_replan", "joint_cost_rerank")
                                and adaptive_remote.numel()
                            ):
                                refined_remote = refined_remote[
                                    ~torch.isin(refined_remote, adaptive_remote)
                                ]
                            refined_remote_count = int(refined_remote.numel())
                            tail_proxy = joint_proxy.clone()
                            absolute_refined = refined_remote + remote_start
                            tail_proxy[absolute_refined] = refined_proxy.index_select(
                                0, absolute_refined
                            )
                        current_max_required_fraction = max(
                            current_max_required_fraction,
                            required_fraction,
                        )
                        current_required_fractions.append(required_fraction)
                    else:
                        required_keep_remote = keep_remote
                        required_fraction = (
                            keep_remote / remote_count if remote_count else 0.0
                        )
                    warp_residual_keep = 0
                    pieces = []
                    if sink_count:
                        pieces.append(
                            torch.arange(sink_count, device=base_priority.device)
                        )
                    if keep_remote:
                        if adaptive_remote is not None:
                            pieces.append(adaptive_remote + remote_start)
                        elif (
                            args.candidate_selection == "warp_local"
                            and isinstance(residual_projection, torch.Tensor)
                        ):
                            (
                                selected_remote,
                                remote_candidates,
                                warp_residual_keep,
                            ) = warp_local_refined_topk(
                                base_priority[remote_start:remote_stop],
                                refined_priority[remote_start:remote_stop],
                                keep_remote,
                                base_keep=args.warp_local_base_keep,
                                shortlist_multiplier=(
                                    args.warp_local_shortlist_multiplier
                                ),
                            )
                            refined_remote_count = int(remote_candidates.numel())
                            pieces.append(selected_remote + remote_start)
                            absolute_candidates = remote_candidates + remote_start
                            tail_proxy = joint_proxy.clone()
                            tail_proxy[absolute_candidates] = (
                                refined_proxy.index_select(
                                    0, absolute_candidates
                                )
                            )
                        elif (
                            isinstance(residual_projection, torch.Tensor)
                            and args.residual_binary_candidate_fraction < 1.0
                        ):
                            candidate_count = min(
                                remote_count,
                                max(
                                    keep_remote,
                                    math.ceil(
                                        remote_count
                                        * args.residual_binary_candidate_fraction
                                    ),
                                ),
                            )
                            refined_remote_count = candidate_count
                            remote_candidates = torch.topk(
                                base_priority[remote_start:remote_stop],
                                candidate_count,
                                sorted=False,
                            ).indices
                            selected_within = torch.topk(
                                refined_priority[
                                    remote_start:remote_stop
                                ].index_select(0, remote_candidates),
                                keep_remote,
                                sorted=False,
                            ).indices
                            selected_remote = remote_candidates.index_select(
                                0, selected_within
                            )
                            pieces.append(selected_remote + remote_start)
                            absolute_candidates = remote_candidates + remote_start
                            tail_proxy = joint_proxy.clone()
                            tail_proxy[absolute_candidates] = (
                                refined_proxy.index_select(
                                    0, absolute_candidates
                                )
                            )
                        else:
                            pieces.append(
                                torch.topk(
                                    refined_priority[
                                        remote_start:remote_stop
                                    ],
                                    keep_remote,
                                    sorted=False,
                                ).indices
                                + remote_start
                            )
                    if recent_count:
                        pieces.append(
                            torch.arange(
                                history_count - recent_count,
                                history_count,
                                device=base_priority.device,
                            )
                        )
                    selected = (
                        torch.cat(pieces)
                        if pieces
                        else torch.argmax(refined_priority)[None]
                    )
                    budget_record: dict[str, float | int] = {
                        "layer": _layer_index,
                        "kv_head": kv_head,
                        "query_head": query_head,
                        "history_tokens": history_count,
                        "remote_tokens": remote_count,
                        "selected_remote_tokens": keep_remote,
                        "selected_remote_fraction": (
                            keep_remote / remote_count if remote_count else 0.0
                        ),
                        "refined_remote_tokens": refined_remote_count,
                        "refined_remote_fraction": (
                            refined_remote_count / remote_count if remote_count else 0.0
                        ),
                        "head_sensitivity": head_sensitivity,
                        "effective_error_tolerance": effective_tolerance,
                        "required_remote_tokens": required_keep_remote,
                        "required_remote_fraction": required_fraction,
                        "score_error_calibration_samples": calibration_samples,
                        "base_score_error_scale": base_error_scale,
                        "refined_score_error_scale": refined_error_scale,
                        "warp_residual_keep": warp_residual_keep,
                    }
                    if adaptive_diagnostics is not None:
                        budget_record.update(adaptive_diagnostics)
                    budget_records.append(budget_record)
                    exact_weights = torch.softmax(exact_scores, dim=0)
                    full_head_output = exact_weights @ head_value
                    if args.tail_mode == "selected_only":
                        selected_scores = exact_scores.index_select(0, selected)
                        sparse_head_output = (
                            torch.softmax(selected_scores, dim=0)
                            @ head_value.index_select(0, selected)
                        )
                    else:
                        sparse_head_output = rvq_tail_output(
                            exact_scores,
                            tail_proxy,
                            head_value,
                            selected,
                            assignments,
                            value_centroids,
                            selected_conditioned=(
                                args.selected_conditioned_tail
                            ),
                        )
                    budget_record.update(
                        {
                            "local_attention_relative_l2": float(
                                (sparse_head_output - full_head_output).norm()
                                / full_head_output.norm().clamp_min(1.0e-8)
                            ),
                            "local_attention_l2": float(
                                (sparse_head_output - full_head_output).norm()
                            ),
                            "full_attention_output_norm": float(
                                full_head_output.norm()
                            ),
                            "selected_attention_mass": float(
                                exact_weights.index_select(0, selected).sum()
                            ),
                            "attention_top1_selected": int(
                                torch.isin(
                                    torch.argmax(exact_scores)[None], selected
                                )[0]
                            ),
                            "attention_entropy": float(
                                -(
                                    exact_weights
                                    * exact_weights.clamp_min(1.0e-20).log()
                                ).sum()
                            ),
                            "maximum_attention_probability": float(
                                exact_weights.max()
                            ),
                            "one_shot_full_probe": int(one_shot_probe),
                        }
                    )
                    if one_shot_probe:
                        probe_full_heads.append(full_head_output)
                        probe_sparse_heads.append(sparse_head_output)
                        probe_effective_tolerances.append(effective_tolerance)
                        probe_records.append(budget_record)
                    else:
                        budget_record["output_error_multiplier"] = float(
                            _runtime_budget_state["output_error_multipliers"].get(
                                query_head, 1.0
                            )
                        )
                    head_outputs.append(
                        full_head_output if one_shot_probe else sparse_head_output
                    )
            if args.adaptive_budget_coupling == "previous_layer_max":
                _runtime_budget_state["previous_max_required_fraction"] = (
                    current_max_required_fraction
                )
            elif (
                args.adaptive_budget_coupling == "previous_layer_quantile"
                and current_required_fractions
            ):
                _runtime_budget_state["previous_required_quantile"] = float(
                    torch.quantile(
                        torch.tensor(current_required_fractions),
                        resolved_budget_quantile,
                    )
                )
            if one_shot_probe:
                feedback = output_error_feedback_multipliers(
                    torch.stack(probe_full_heads),
                    torch.stack(probe_sparse_heads),
                    self.o_proj.weight,
                    hidden_states[0, 0].float().norm(),
                    torch.tensor(
                        probe_effective_tolerances,
                        device=hidden_states.device,
                    ),
                    args.one_shot_feedback_norm,
                )
                multipliers = feedback["multipliers"]
                projected_errors = feedback["projected_error_l2"]
                normalizers = feedback["normalizer_l2"]
                for record, multiplier, projected_error, normalizer in zip(
                    probe_records,
                    multipliers,
                    projected_errors,
                    normalizers,
                    strict=True,
                ):
                    query_head = int(record["query_head"])
                    _runtime_budget_state["output_error_multipliers"][
                        query_head
                    ] = float(multiplier)
                    record["output_error_multiplier"] = float(multiplier)
                    record["projected_attention_error_l2"] = float(
                        projected_error
                    )
                    record["output_error_normalizer_l2"] = float(normalizer)
                    record["projected_layer_relative_l2"] = float(
                        feedback["projected_layer_relative_l2"]
                    )
                _runtime_budget_state["output_error_calibrated"] = True
            attention_output = torch.stack(head_outputs).to(hidden_states.dtype)
            attention_output = attention_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attention_output), None

        attention.forward = types.MethodType(sparse_forward, attention)
    return originals, budget_records


def parse_layer_spec(spec: str, layer_count: int) -> set[int]:
    if spec == "all":
        return set(range(layer_count))
    layers: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = (int(part) for part in item.split("-", 1))
            layers.update(range(start, stop + 1))
        else:
            layers.add(int(item))
    if not layers or min(layers) < 0 or max(layers) >= layer_count:
        raise ValueError(f"invalid sparse layer specification: {spec}")
    return layers


def parse_layer_fraction_overrides(spec: str, layer_count: int) -> dict[int, float]:
    output: dict[int, float] = {}
    for rule in spec.split(","):
        rule = rule.strip()
        if not rule:
            continue
        layer_spec, fraction_text = rule.split(":", 1)
        fraction = float(fraction_text)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"invalid layer fraction: {fraction}")
        for layer in parse_layer_spec(layer_spec, layer_count):
            output[layer] = fraction
    return output


def restore_attention(originals: list[tuple[Any, Any]]) -> None:
    for attention, forward in originals:
        attention.forward = forward


def move_nested(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_nested(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_nested(item, device) for item in value]
    return value


def estimate_one_shot_head_sensitivities(
    model: Any,
    input_ids: torch.Tensor,
    history_tokens: int,
    probes: int,
    objective_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Estimate per-head downstream logit-Jacobian gains on one decode step."""
    if probes <= 0:
        raise ValueError("head-sensitivity probe count must be positive")
    model.requires_grad_(False)
    captured: list[torch.Tensor | None] = [None] * len(model.model.layers)
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        def capture_input(
            _module: Any,
            inputs: tuple[torch.Tensor, ...],
            _layer_index: int = layer_index,
        ) -> None:
            captured[_layer_index] = inputs[0]

        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(capture_input))
    try:
        with torch.no_grad():
            prefill = model(
                input_ids=input_ids[:, :history_tokens],
                use_cache=True,
                return_dict=True,
            )
        current_ids = input_ids[:, history_tokens : history_tokens + 1]
        current_embeds = (
            model.get_input_embeddings()(current_ids).detach().requires_grad_(True)
        )
        output = model(
            inputs_embeds=current_embeds,
            past_key_values=prefill.past_key_values,
            use_cache=False,
            return_dict=True,
        )
        if any(item is None for item in captured):
            raise RuntimeError("failed to capture every attention-head output")
        tensors = [item for item in captured if item is not None]
        squared = [torch.zeros_like(item[0, -1].float()) for item in tensors]
        generator = torch.Generator(device=output.logits.device).manual_seed(91_337)
        log_probabilities = torch.log_softmax(output.logits.float(), dim=-1)
        probabilities = log_probabilities.exp()
        for probe in range(probes):
            if objective_mode == "one_shot_hutchinson":
                signs = (
                    2
                    * torch.randint(
                        0,
                        2,
                        output.logits.shape,
                        generator=generator,
                        device=output.logits.device,
                    )
                    - 1
                ).to(output.logits.dtype)
                scalar = (output.logits * signs).sum() / math.sqrt(
                    output.logits.shape[-1]
                )
            elif objective_mode == "one_shot_fisher":
                sampled = torch.multinomial(
                    probabilities.flatten(),
                    1,
                    generator=generator,
                )
                scalar = log_probabilities.flatten().index_select(0, sampled).sum()
            else:
                raise ValueError(f"unsupported sensitivity objective: {objective_mode}")
            gradients = torch.autograd.grad(
                scalar,
                tensors,
                retain_graph=probe + 1 < probes,
                allow_unused=False,
            )
            for layer_index, gradient in enumerate(gradients):
                squared[layer_index] += gradient[0, -1].float().square()
        query_heads = model.config.num_attention_heads
        head_dim = model.model.layers[0].self_attn.head_dim
        sensitivities = torch.stack(
            [
                (item / probes)
                .reshape(query_heads, head_dim)
                .sum(dim=-1)
                .sqrt()
                for item in squared
            ]
        )
        rms = sensitivities.square().mean().sqrt().clamp_min(1.0e-12)
        harmonic = sensitivities.numel() / sensitivities.clamp_min(
            1.0e-12
        ).reciprocal().sum()
        normalized = sensitivities / harmonic
        return normalized.detach(), {
            "raw_rms": float(rms),
            "raw_harmonic_mean": float(harmonic),
            "normalized_min": float(normalized.min()),
            "normalized_p50": float(torch.quantile(normalized.flatten(), 0.50)),
            "normalized_p90": float(torch.quantile(normalized.flatten(), 0.90)),
            "normalized_max": float(normalized.max()),
            "probes": float(probes),
            "objective_mode": objective_mode,
        }
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def evaluate_decode(
    model: Any,
    input_ids: torch.Tensor,
    history_tokens: int,
    eval_tokens: int,
) -> tuple[list[float], list[torch.Tensor]]:
    prefill = model(
        input_ids=input_ids[:, :history_tokens],
        use_cache=True,
        return_dict=True,
    )
    cache = prefill.past_key_values
    losses = []
    logits = []
    for offset in range(eval_tokens):
        current = input_ids[:, history_tokens + offset : history_tokens + offset + 1]
        target = input_ids[:, history_tokens + offset + 1]
        output = model(
            input_ids=current,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        current_logits = output.logits[:, -1].float()
        losses.append(float(F.cross_entropy(current_logits, target)))
        logits.append(current_logits.cpu())
    return losses, logits


def paired_metrics(
    full_losses: list[float],
    sparse_losses: list[float],
    full_logits: list[torch.Tensor],
    sparse_logits: list[torch.Tensor],
) -> dict[str, float]:
    full_nll = sum(full_losses) / len(full_losses)
    sparse_nll = sum(sparse_losses) / len(sparse_losses)
    agreements = []
    kls = []
    for full, sparse in zip(full_logits, sparse_logits, strict=True):
        agreements.append(float(full.argmax(dim=-1).eq(sparse.argmax(dim=-1))[0]))
        full_probability = torch.softmax(full, dim=-1)
        kls.append(
            float(
                (full_probability * (torch.log_softmax(full, dim=-1) - torch.log_softmax(sparse, dim=-1))).sum()
            )
        )
    return {
        "full_nll": full_nll,
        "sparse_nll": sparse_nll,
        "full_ppl": math.exp(full_nll),
        "sparse_ppl": math.exp(sparse_nll),
        "quality_ratio": math.exp(full_nll - sparse_nll),
        "top1_agreement": sum(agreements) / len(agreements),
        "full_to_sparse_kl": sum(kls) / len(kls),
    }


def summarize_budget_records(
    records: list[dict[str, float | int]],
) -> dict[str, Any]:
    if not records:
        return {"record_count": 0, "per_layer": {}}

    def distribution(values: list[float]) -> dict[str, float]:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "mean": float(tensor.mean()),
            "p50": float(torch.quantile(tensor, 0.50)),
            "p90": float(torch.quantile(tensor, 0.90)),
            "max": float(tensor.max()),
        }

    selected_fractions = [
        float(record["selected_remote_fraction"]) for record in records
    ]
    selected_tokens = [
        float(record["selected_remote_tokens"]) for record in records
    ]
    refined_fractions = [
        float(record["refined_remote_fraction"]) for record in records
    ]
    refined_tokens = [float(record["refined_remote_tokens"]) for record in records]
    required_fractions = [
        float(record["required_remote_fraction"]) for record in records
    ]
    summary: dict[str, Any] = {
        "record_count": len(records),
        "one_shot_probe_record_count": sum(
            int(record.get("one_shot_full_probe", 0)) for record in records
        ),
        "selected_remote_fraction": distribution(selected_fractions),
        "selected_remote_tokens": distribution(selected_tokens),
        "refined_remote_fraction": distribution(refined_fractions),
        "refined_remote_tokens": distribution(refined_tokens),
        "required_remote_fraction": distribution(required_fractions),
        "per_layer": {},
    }
    for diagnostic_key in (
        "predicted_relative_error_after_selection",
        "predicted_relative_error_after_actions",
        "modeled_variable_cost",
        "lagrange_multiplier",
        "head_sensitivity",
        "effective_error_tolerance",
        "base_score_error_scale",
        "refined_score_error_scale",
        "score_error_calibration_samples",
        "local_attention_relative_l2",
        "local_attention_l2",
        "full_attention_output_norm",
        "selected_attention_mass",
        "attention_entropy",
        "maximum_attention_probability",
        "output_error_multiplier",
        "one_shot_full_probe",
        "projected_attention_error_l2",
        "output_error_normalizer_l2",
        "projected_layer_relative_l2",
    ):
        values = [
            float(record[diagnostic_key])
            for record in records
            if diagnostic_key in record
        ]
        if values:
            summary[diagnostic_key] = distribution(values)
    for layer in sorted({int(record["layer"]) for record in records}):
        layer_records = [record for record in records if int(record["layer"]) == layer]
        summary["per_layer"][str(layer)] = {
            "record_count": len(layer_records),
            "selected_remote_fraction": distribution(
                [float(record["selected_remote_fraction"]) for record in layer_records]
            ),
            "selected_remote_tokens": distribution(
                [float(record["selected_remote_tokens"]) for record in layer_records]
            ),
            "refined_remote_fraction": distribution(
                [float(record["refined_remote_fraction"]) for record in layer_records]
            ),
            "required_remote_fraction": distribution(
                [float(record["required_remote_fraction"]) for record in layer_records]
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.residual_binary_bits < 0:
        raise ValueError("residual binary width cannot be negative")
    if not 0.0 < args.residual_binary_candidate_fraction <= 1.0:
        raise ValueError("residual binary candidate fraction must lie in (0, 1]")
    if args.adaptive_error_tolerance != 0.0 and not (
        0.0 < args.adaptive_error_tolerance < 1.0
    ):
        raise ValueError("adaptive error tolerance must be zero or lie in (0, 1)")
    if (
        args.adaptive_error_tolerance > 0.0
        and args.residual_binary_candidate_fraction < 1.0
    ):
        raise ValueError(
            "adaptive budget validation currently requires a full residual-code scan"
        )
    if args.residual_refinement_cost_bits <= 0.0:
        raise ValueError("residual refinement cost must be positive")
    if args.exact_kv_cost_bits <= args.residual_refinement_cost_bits:
        raise ValueError("exact K/V cost must exceed residual refinement cost")
    if (
        args.adaptive_action_mode
        in ("joint_cost", "joint_cost_replan", "joint_cost_rerank")
        and args.residual_binary_bits <= 0
    ):
        raise ValueError("joint-cost control requires a residual binary code")
    if (
        args.one_shot_output_error_feedback
        and args.adaptive_error_tolerance <= 0.0
    ):
        raise ValueError(
            "one-shot output-error feedback requires an adaptive error tolerance"
        )
    if (
        args.one_shot_feedback_source == "suffix_token"
        and not args.one_shot_output_error_feedback
    ):
        raise ValueError("suffix-token feedback requires one-shot feedback")
    if args.one_shot_feedback_source == "suffix_token" and args.history_tokens < 2:
        raise ValueError("suffix-token feedback requires at least two history tokens")
    if args.head_sensitivity_probes <= 0:
        raise ValueError("head-sensitivity probe count must be positive")
    if args.adaptive_budget_quantile > 1.0:
        raise ValueError("adaptive budget quantile must be at most one")
    torch.set_num_threads(args.threads)
    assert_numeric_backend_sane()
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    cache_contract = {
        "model": str(args.model),
        "train_texts": [str(path) for path in args.train_texts],
        "calibration_tokens": args.calibration_tokens,
        "query_samples_per_text": args.query_samples_per_text,
        "key_samples_per_text": args.key_samples_per_text,
        "binary_bits": args.binary_bits,
        "projection_iterations": args.projection_iterations,
        "residual_vq_bits": args.residual_vq_bits,
        "residual_vq_iterations": args.residual_vq_iterations,
        "residual_binary_bits": args.residual_binary_bits,
        "residual_binary_iterations": args.residual_binary_iterations,
        "joint_value_weight": args.joint_value_weight,
        "metric_shrinkage": args.metric_shrinkage,
    }
    if args.codebook_cache is not None and args.codebook_cache.exists():
        cached = torch.load(args.codebook_cache, map_location="cpu", weights_only=False)
        if cached["contract"] != cache_contract:
            raise ValueError("codebook cache contract differs from this run")
        codebooks = move_nested(cached["codebooks"], args.device)
        metadata = cached["metadata"]
    else:
        codebooks, metadata = fit_codebooks(model, tokenizer, args)
        if args.codebook_cache is not None:
            args.codebook_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "contract": cache_contract,
                    "codebooks": move_nested(codebooks, "cpu"),
                    "metadata": metadata,
                },
                args.codebook_cache,
            )

    rows = []
    all_budget_records: list[dict[str, float | int]] = []
    needed = args.history_tokens + args.eval_tokens + 1
    encoded_tests = [
        (path, encode_text(tokenizer, path, needed).to(args.device))
        for path in args.test_texts
    ]
    for path, ids in encoded_tests:
        head_sensitivities = None
        head_sensitivity_diagnostics = None
        if args.head_sensitivity_mode != "none":
            head_sensitivities, head_sensitivity_diagnostics = (
                estimate_one_shot_head_sensitivities(
                    model,
                    ids,
                    args.history_tokens,
                    args.head_sensitivity_probes,
                    args.head_sensitivity_mode,
                )
            )
        full_losses, full_logits = evaluate_decode(
            model, ids, args.history_tokens, args.eval_tokens
        )
        originals, budget_records = install_sparse_attention(
            model,
            codebooks,
            args,
            head_sensitivities=head_sensitivities,
        )
        try:
            if args.one_shot_feedback_source == "suffix_token":
                suffix_losses, suffix_logits = evaluate_decode(
                    model,
                    ids,
                    args.history_tokens - 1,
                    args.eval_tokens + 1,
                )
                sparse_losses = suffix_losses[1:]
                sparse_logits = suffix_logits[1:]
            else:
                sparse_losses, sparse_logits = evaluate_decode(
                    model, ids, args.history_tokens, args.eval_tokens
                )
        finally:
            restore_attention(originals)
        all_budget_records.extend(budget_records)
        rows.append(
            {
                "text": path.stem,
                **paired_metrics(
                    full_losses,
                    sparse_losses,
                    full_logits,
                    sparse_logits,
                ),
                "full_token_nll": full_losses,
                "sparse_token_nll": sparse_losses,
                "budget": summarize_budget_records(budget_records),
                "head_sensitivity": head_sensitivity_diagnostics,
                "budget_records": (
                    budget_records if args.store_budget_records else None
                ),
            }
        )

    aggregate = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in (
            "full_nll",
            "sparse_nll",
            "full_ppl",
            "sparse_ppl",
            "quality_ratio",
            "top1_agreement",
            "full_to_sparse_kl",
        )
    }
    payload = {
        "schema": "jointkv-residual-all-layer-ppl-v1",
        "contract": {
            "scope": "independent teacher-forced decode PPL; dense prefill",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model": str(args.model),
            "attn_implementation": args.attn_implementation,
            "train_texts": [str(path) for path in args.train_texts],
            "test_texts": [str(path) for path in args.test_texts],
            "calibration_tokens": args.calibration_tokens,
            "history_tokens": args.history_tokens,
            "eval_tokens": args.eval_tokens,
            "fraction": args.fraction,
            "recent_tokens": args.recent_tokens,
            "sink_tokens": args.sink_tokens,
            "sparse_layers": args.sparse_layers,
            "layer_fraction_overrides": args.layer_fraction_overrides,
            "binary_bits": args.binary_bits,
            "residual_vq_bits": args.residual_vq_bits,
            "residual_binary_bits": args.residual_binary_bits,
            "residual_binary_candidate_fraction": (
                args.residual_binary_candidate_fraction
            ),
            "adaptive_error_tolerance": args.adaptive_error_tolerance,
            "adaptive_action_mode": args.adaptive_action_mode,
            "residual_refinement_cost_bits": args.residual_refinement_cost_bits,
            "exact_kv_cost_bits": args.exact_kv_cost_bits,
            "head_sensitivity_mode": args.head_sensitivity_mode,
            "head_sensitivity_probes": args.head_sensitivity_probes,
            "score_error_calibration_samples": args.score_error_calibration_samples,
            "adaptive_budget_coupling": args.adaptive_budget_coupling,
            "adaptive_budget_quantile": args.adaptive_budget_quantile,
            "one_shot_output_error_feedback": (
                args.one_shot_output_error_feedback
            ),
            "one_shot_feedback_norm": args.one_shot_feedback_norm,
            "one_shot_feedback_source": args.one_shot_feedback_source,
            "resolved_adaptive_budget_quantile": (
                args.adaptive_budget_quantile
                if args.adaptive_budget_quantile >= 0.0
                else 1.0 - 1.0 / model.config.num_attention_heads
            ),
            "joint_value_weight": args.joint_value_weight,
            "risk_lambda": args.risk_lambda,
            "priority_mode": args.priority_mode,
            "risk_error_bits": args.risk_error_bits,
            "risk_error_block_size": args.risk_error_block_size,
            "value_mean_bits": args.value_mean_bits,
            "refit_key_bits": args.refit_key_bits,
            "selected_conditioned_tail": args.selected_conditioned_tail,
            "tail_mode": args.tail_mode,
            "persistent_index": not args.rebuild_index_each_step,
            "store_budget_records": args.store_budget_records,
            "codebook_cache": (
                str(args.codebook_cache) if args.codebook_cache is not None else None
            ),
            "metadata": metadata,
            "resolved_shrinkage": [
                [head["shrinkage"] for head in layer] for layer in codebooks
            ],
        },
        "aggregate": aggregate,
        "budget": summarize_budget_records(all_budget_records),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
