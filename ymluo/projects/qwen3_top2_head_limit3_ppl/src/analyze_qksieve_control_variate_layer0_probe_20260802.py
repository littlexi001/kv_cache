#!/usr/bin/env python
"""CPU mechanism probe on real Qwen3 layer-0 Q/K/V activations.

This deliberately bypasses a full model forward: layer 0 Q/K/V are computed
from token embeddings and released model weights.  It is a real-attention
tensor audit, not an end-to-end quality result.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from analyze_qksieve_block_coreset_20260802 import (
    block_coreset_tail_statistics,
    coreset_corrected_proxy_scores,
    fit_block_coreset,
)
from analyze_qksieve_conditional_value_moments_20260802 import (
    combine_selected_and_tail,
    control_variate_tail_statistics,
    fit_block_models,
    fit_gaussian_tilt_moments,
    gaussian_tilt_block_control_values,
    stratified_uniform_sample_indices,
)
from analyze_qksieve_taylor_tail_20260802 import (
    fit_taylor_block_tail,
    taylor_block_tail_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--texts", type=Path, nargs="+", required=True)
    parser.add_argument("--history_tokens", type=int, default=8192)
    parser.add_argument("--query_tokens", type=int, default=4)
    parser.add_argument("--ranks", default="16,32,48,64")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--read_fraction", type=float, default=0.06)
    parser.add_argument("--samples_per_block", type=int, default=2)
    parser.add_argument("--coreset_clusters", default="2,4,8")
    parser.add_argument("--coreset_moment_bits", type=int, default=8)
    parser.add_argument("--coreset_value_bits", type=int, default=0)
    parser.add_argument("--coreset_full_key_bits", type=int, default=0)
    parser.add_argument("--taylor_variance_bits", type=int, default=4)
    parser.add_argument("--taylor_cross_bits", type=int, default=4)
    parser.add_argument("--taylor_cross_key_dim", type=int, default=16)
    parser.add_argument("--taylor_cross_value_dim", type=int, default=8)
    parser.add_argument("--joint_value_projection_dim", type=int, default=8)
    parser.add_argument("--joint_value_weight", type=float, default=0.5)
    parser.add_argument(
        "--linear_group_blocks",
        type=int,
        default=0,
        help="Blocks sharing one conditional K-to-V map; 0 shares globally.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    normalized = hidden.float() * torch.rsqrt(
        hidden.float().square().mean(dim=-1, keepdim=True) + epsilon
    )
    return normalized * weight.float()


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    half = tensor.shape[-1] // 2
    return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


def apply_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    dimension = tensor.shape[-1]
    inverse_frequency = 1.0 / (
        theta
        ** (
            torch.arange(0, dimension, 2, dtype=torch.float32)
            / float(dimension)
        )
    )
    frequencies = positions.float()[:, None] * inverse_frequency[None, :]
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos()[:, None, :]
    sine = embedding.sin()[:, None, :]
    return tensor.float() * cosine + rotate_half(tensor.float()) * sine


def find_weight_file(model: Path) -> Path:
    direct = model / "model.safetensors"
    if direct.exists():
        return direct
    index_path = model / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        required = (
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        )
        shards = {weight_map[name] for name in required}
        if len(shards) == 1:
            return model / shards.pop()
    files = sorted(model.glob("model-*-of-*.safetensors"))
    if len(files) == 1:
        return files[0]
    raise FileNotFoundError(
        "the layer-0 tensors must reside in one safetensors shard"
    )


def load_layer0_activations(
    model: Path,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    configuration = json.loads((model / "config.json").read_text(encoding="utf-8"))
    hidden_size = int(configuration["hidden_size"])
    query_heads = int(configuration["num_attention_heads"])
    kv_heads = int(configuration["num_key_value_heads"])
    head_dim = int(configuration["head_dim"])
    epsilon = float(configuration["rms_norm_eps"])
    theta = float(configuration["rope_theta"])
    weight_file = find_weight_file(model)
    with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor("model.embed_tokens.weight")
        hidden = embedding.index_select(0, token_ids).float()
        hidden = rms_norm(
            hidden,
            handle.get_tensor("model.layers.0.input_layernorm.weight"),
            epsilon,
        )
        query = F.linear(
            hidden,
            handle.get_tensor("model.layers.0.self_attn.q_proj.weight").float(),
        ).reshape(-1, query_heads, head_dim)
        key = F.linear(
            hidden,
            handle.get_tensor("model.layers.0.self_attn.k_proj.weight").float(),
        ).reshape(-1, kv_heads, head_dim)
        value = F.linear(
            hidden,
            handle.get_tensor("model.layers.0.self_attn.v_proj.weight").float(),
        ).reshape(-1, kv_heads, head_dim)
        query = rms_norm(
            query,
            handle.get_tensor("model.layers.0.self_attn.q_norm.weight"),
            epsilon,
        )
        key = rms_norm(
            key,
            handle.get_tensor("model.layers.0.self_attn.k_norm.weight"),
            epsilon,
        )
    positions = torch.arange(token_ids.numel())
    query = apply_rope(query, positions, theta)
    key = apply_rope(key, positions, theta)
    metadata = {
        "hidden_size": hidden_size,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "gqa_groups": query_heads // kv_heads,
        "rope_theta": theta,
    }
    return query, key, value.float(), metadata


def uncentered_basis(keys: torch.Tensor, rank: int) -> torch.Tensor:
    gram = keys.float().T @ keys.float()
    _, eigenvectors = torch.linalg.eigh(gram)
    return eigenvectors[:, -rank:].flip(dims=(1,)).contiguous()


def block_tail_counts(
    token_count: int, block_size: int, selected: torch.Tensor
) -> torch.Tensor:
    block_count = math.ceil(token_count / block_size)
    starts = torch.arange(block_count) * block_size
    counts = (token_count - starts).clamp(min=0, max=block_size).float()
    selected_counts = torch.zeros(block_count)
    selected_counts.index_add_(
        0,
        selected // block_size,
        torch.ones_like(selected, dtype=torch.float32),
    )
    return counts - selected_counts


def output_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(output - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
        ),
        "cosine": float(
            F.cosine_similarity(output[None], reference[None], dim=-1)[0]
        ),
        "norm_ratio": float(
            torch.linalg.vector_norm(output)
            / torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
        ),
    }


def evaluate_text(
    text_path: Path,
    token_ids: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query, key, value, model_metadata = load_layer0_activations(
        args.model, token_ids
    )
    history_count = args.history_tokens
    history_key = key[:history_count]
    history_value = value[:history_count]
    query_slice = query[history_count : history_count + args.query_tokens]
    total_reads = max(1, math.ceil(history_count * args.read_fraction))
    rows: list[dict[str, Any]] = []
    scale = model_metadata["head_dim"] ** -0.5
    group_size = int(model_metadata["gqa_groups"])
    ranks = [int(item) for item in args.ranks.split(",") if item.strip()]
    for kv_head in range(int(model_metadata["kv_heads"])):
        head_key = history_key[:, kv_head].contiguous()
        head_value = history_value[:, kv_head].contiguous()
        max_rank = max(ranks)
        full_basis = uncentered_basis(head_key, max_rank)
        reservoir = stratified_uniform_sample_indices(
            history_count,
            args.block_size,
            args.samples_per_block,
            torch.Generator().manual_seed(20260802 + kv_head),
        )
        heavy_count = max(1, total_reads - reservoir.numel())
        for rank in ranks:
            basis = full_basis[:, :rank]
            coordinates = head_key @ basis
            conditional_dim = min(16, rank)
            conditional_coordinates = coordinates[:, :conditional_dim]
            conditional_model = fit_block_models(
                conditional_coordinates,
                head_value,
                args.block_size,
                ridge=0.01,
                moment_bits=8,
                linear_group_blocks=args.linear_group_blocks,
                linear_fit_stride=2,
            )
            gaussian_model = fit_gaussian_tilt_moments(
                coordinates,
                args.block_size,
                moment_bits=8,
                covariance_mode="diag",
            )
            coreset_models: dict[str, dict[str, torch.Tensor | int | float]] = {}
            for cluster_count in (
                int(item)
                for item in args.coreset_clusters.split(",")
                if item.strip()
            ):
                coreset_models[f"coreset_k_c{cluster_count}"] = fit_block_coreset(
                    coordinates,
                    head_value,
                    args.block_size,
                    cluster_count,
                    moment_bits=args.coreset_moment_bits,
                    iterations=6,
                    full_score_coordinates=head_key,
                    value_moment_bits=(
                        args.coreset_value_bits or args.coreset_moment_bits
                    ),
                    full_score_moment_bits=(
                        args.coreset_full_key_bits or args.coreset_moment_bits
                    ),
                )
                coreset_models[f"coreset_joint_c{cluster_count}"] = (
                    fit_block_coreset(
                        coordinates,
                        head_value,
                        args.block_size,
                        cluster_count,
                        moment_bits=args.coreset_moment_bits,
                        iterations=6,
                        value_projection_dim=args.joint_value_projection_dim,
                        value_weight=args.joint_value_weight,
                        full_score_coordinates=head_key,
                        value_moment_bits=(
                            args.coreset_value_bits or args.coreset_moment_bits
                        ),
                        full_score_moment_bits=(
                            args.coreset_full_key_bits
                            or args.coreset_moment_bits
                        ),
                    )
                )
            taylor_model = fit_taylor_block_tail(
                coordinates,
                head_key,
                head_value,
                args.block_size,
                key_mean_bits=(
                    args.coreset_full_key_bits or args.coreset_moment_bits
                ),
                value_mean_bits=(
                    args.coreset_value_bits or args.coreset_moment_bits
                ),
                variance_bits=args.taylor_variance_bits,
                cross_bits=args.taylor_cross_bits,
                cross_key_dim=args.taylor_cross_key_dim,
                cross_value_dim=args.taylor_cross_value_dim,
            )
            mean = gaussian_model["mean"]
            mean_v = conditional_model["mean_v"]
            assert isinstance(mean, torch.Tensor)
            assert isinstance(mean_v, torch.Tensor)
            for query_offset in range(args.query_tokens):
                for group_offset in range(group_size):
                    query_head = kv_head * group_size + group_offset
                    current_query = query_slice[query_offset, query_head]
                    exact_scores = (head_key @ current_query) * scale
                    score_direction = (current_query @ basis) * scale
                    proxy_scores = coordinates @ score_direction
                    selected_total = torch.topk(
                        proxy_scores, total_reads, sorted=False
                    ).indices
                    selected_heavy = torch.topk(
                        proxy_scores, heavy_count, sorted=False
                    ).indices
                    oracle = torch.topk(
                        exact_scores, total_reads, sorted=False
                    ).indices
                    full_weights = torch.softmax(exact_scores, dim=0)
                    reference_output = full_weights @ head_value
                    proxy_reference = proxy_scores.index_select(
                        0, selected_heavy
                    ).amin()
                    gaussian_z, gaussian_y = gaussian_tilt_block_control_values(
                        score_direction,
                        0.0,
                        proxy_reference,
                        conditional_model,
                        gaussian_model,
                    )
                    centroid_z = torch.exp(
                        (
                            mean.float() @ score_direction.float()
                            - proxy_reference
                        ).clamp(-80.0, 80.0)
                    )
                    centroid_y = centroid_z[:, None] * mean_v.float()
                    tail_counts = block_tail_counts(
                        history_count, args.block_size, selected_heavy
                    )
                    gaussian_base_z = (tail_counts * gaussian_z).sum()
                    gaussian_base_y = (
                        tail_counts[:, None] * gaussian_y
                    ).sum(dim=0)
                    centroid_base_z = (tail_counts * centroid_z).sum()
                    centroid_base_y = (
                        tail_counts[:, None] * centroid_y
                    ).sum(dim=0)
                    cv_z, cv_y, cv_diagnostics = (
                        control_variate_tail_statistics(
                            exact_scores,
                            head_value,
                            selected_heavy,
                            reservoir,
                            args.block_size,
                            gaussian_z,
                            gaussian_y,
                            proxy_reference,
                        )
                    )
                    coreset_outputs: dict[str, torch.Tensor] = {}
                    coreset_diagnostics: dict[str, dict[str, float]] = {}
                    coreset_method_models: dict[
                        str, dict[str, torch.Tensor | int | float]
                    ] = {}
                    method_selected_mass: dict[str, float] = {}
                    method_top1_recall: dict[str, float] = {}
                    total_reference = proxy_scores.index_select(
                        0, selected_total
                    ).amin()
                    exact_reference = exact_scores.index_select(
                        0, selected_total
                    ).amin()
                    for use_variance, use_cross, taylor_name in (
                        (False, False, "taylor_mean"),
                        (True, False, "taylor_variance"),
                        (False, True, "taylor_cross"),
                        (True, True, "taylor_variance_cross"),
                    ):
                        taylor_z, taylor_y, taylor_diagnostics = (
                            taylor_block_tail_statistics(
                                current_query * scale,
                                score_direction,
                                selected_total,
                                exact_reference,
                                taylor_model,
                                use_variance=use_variance,
                                use_cross=use_cross,
                            )
                        )
                        coreset_outputs[taylor_name] = combine_selected_and_tail(
                            exact_scores,
                            exact_scores,
                            head_value,
                            selected_total,
                            taylor_y,
                            taylor_z,
                            1.0,
                        )
                        coreset_diagnostics[taylor_name] = taylor_diagnostics
                        coreset_method_models[taylor_name] = taylor_model
                    for coreset_name, coreset_model in coreset_models.items():
                        coreset_z, coreset_y, current_diagnostics = (
                            block_coreset_tail_statistics(
                                coordinates,
                                head_value,
                                score_direction,
                                selected_total,
                                total_reference,
                                coreset_model,
                                selected_conditioned=True,
                            )
                        )
                        coreset_outputs[coreset_name] = combine_selected_and_tail(
                            exact_scores,
                            exact_scores,
                            head_value,
                            selected_total,
                            coreset_y,
                            coreset_z,
                            1.0,
                        )
                        coreset_diagnostics[coreset_name] = current_diagnostics
                        coreset_method_models[coreset_name] = coreset_model

                        counttail_z, counttail_y, counttail_diagnostics = (
                            block_coreset_tail_statistics(
                                coordinates,
                                head_value,
                                score_direction,
                                selected_total,
                                total_reference,
                                coreset_model,
                                selected_conditioned=False,
                            )
                        )
                        counttail_name = f"{coreset_name}_counttail"
                        coreset_outputs[counttail_name] = combine_selected_and_tail(
                            exact_scores,
                            proxy_scores,
                            head_value,
                            selected_total,
                            counttail_y,
                            counttail_z,
                            1.0,
                        )
                        coreset_diagnostics[counttail_name] = counttail_diagnostics
                        coreset_method_models[counttail_name] = coreset_model
                        fulltail_z, fulltail_y, fulltail_diagnostics = (
                            block_coreset_tail_statistics(
                                coordinates,
                                head_value,
                                score_direction,
                                selected_total,
                                exact_reference,
                                coreset_model,
                                selected_conditioned=True,
                                full_score_coordinates=head_key,
                                full_score_direction=current_query * scale,
                            )
                        )
                        fulltail_name = f"{coreset_name}_fulltail"
                        coreset_outputs[fulltail_name] = combine_selected_and_tail(
                            exact_scores,
                            exact_scores,
                            head_value,
                            selected_total,
                            fulltail_y,
                            fulltail_z,
                            1.0,
                        )
                        coreset_diagnostics[fulltail_name] = fulltail_diagnostics
                        coreset_method_models[fulltail_name] = coreset_model

                        full_counttail_z, full_counttail_y, full_counttail_diagnostics = (
                            block_coreset_tail_statistics(
                                coordinates,
                                head_value,
                                score_direction,
                                selected_total,
                                exact_reference,
                                coreset_model,
                                selected_conditioned=False,
                                full_score_coordinates=head_key,
                                full_score_direction=current_query * scale,
                            )
                        )
                        full_counttail_name = f"{coreset_name}_full_counttail"
                        coreset_outputs[full_counttail_name] = (
                            combine_selected_and_tail(
                                exact_scores,
                                exact_scores,
                                head_value,
                                selected_total,
                                full_counttail_y,
                                full_counttail_z,
                                1.0,
                            )
                        )
                        coreset_diagnostics[full_counttail_name] = (
                            full_counttail_diagnostics
                        )
                        coreset_method_models[full_counttail_name] = coreset_model
                        corrected_scores, correction_diagnostics = (
                            coreset_corrected_proxy_scores(
                                proxy_scores,
                                score_direction,
                                current_query * scale,
                                coreset_model,
                            )
                        )
                        corrected_selected = torch.topk(
                            corrected_scores, total_reads, sorted=False
                        ).indices
                        corrected_reference = corrected_scores.index_select(
                            0, corrected_selected
                        ).amin()
                        corrected_z, corrected_y, corrected_diagnostics = (
                            block_coreset_tail_statistics(
                                coordinates,
                                head_value,
                                score_direction,
                                corrected_selected,
                                corrected_reference,
                                coreset_model,
                                selected_conditioned=True,
                                full_score_coordinates=head_key,
                                full_score_direction=current_query * scale,
                            )
                        )
                        corrected_name = f"{coreset_name}_scorecorr"
                        coreset_outputs[corrected_name] = (
                            combine_selected_and_tail(
                                exact_scores,
                                corrected_scores,
                                head_value,
                                corrected_selected,
                                corrected_y,
                                corrected_z,
                                1.0,
                            )
                        )
                        coreset_diagnostics[corrected_name] = {
                            **correction_diagnostics,
                            **corrected_diagnostics,
                        }
                        coreset_method_models[corrected_name] = coreset_model
                        method_selected_mass[corrected_name] = float(
                            full_weights.index_select(
                                0, corrected_selected
                            ).sum()
                        )
                        method_top1_recall[corrected_name] = float(
                            torch.isin(
                                torch.argmax(exact_scores)[None],
                                corrected_selected,
                            )[0]
                        )
                        rescue_count = max(1, math.ceil(0.10 * total_reads))
                        base_count = total_reads - rescue_count
                        base_selected = torch.topk(
                            proxy_scores, base_count, sorted=False
                        ).indices
                        corrected_order = torch.topk(
                            corrected_scores, total_reads, sorted=True
                        ).indices
                        already_selected = torch.zeros(
                            history_count,
                            dtype=torch.bool,
                            device=proxy_scores.device,
                        )
                        already_selected[base_selected] = True
                        rescue = corrected_order[
                            ~already_selected.index_select(0, corrected_order)
                        ][:rescue_count]
                        rescued_selected = torch.cat((base_selected, rescue))
                        if rescued_selected.numel() != total_reads:
                            raise RuntimeError("cluster rescue did not fill its budget")
                        rescued_reference = exact_scores.index_select(
                            0, rescued_selected
                        ).amin()
                        rescued_z, rescued_y, rescued_diagnostics = (
                            block_coreset_tail_statistics(
                                coordinates,
                                head_value,
                                score_direction,
                                rescued_selected,
                                rescued_reference,
                                coreset_model,
                                selected_conditioned=True,
                                full_score_coordinates=head_key,
                                full_score_direction=current_query * scale,
                            )
                        )
                        rescued_name = f"{coreset_name}_rescue10"
                        coreset_outputs[rescued_name] = combine_selected_and_tail(
                            exact_scores,
                            exact_scores,
                            head_value,
                            rescued_selected,
                            rescued_y,
                            rescued_z,
                            1.0,
                        )
                        coreset_diagnostics[rescued_name] = {
                            **correction_diagnostics,
                            **rescued_diagnostics,
                        }
                        coreset_method_models[rescued_name] = coreset_model
                        method_selected_mass[rescued_name] = float(
                            full_weights.index_select(0, rescued_selected).sum()
                        )
                        method_top1_recall[rescued_name] = float(
                            torch.isin(
                                torch.argmax(exact_scores)[None],
                                rescued_selected,
                            )[0]
                        )
                    selected_mass = float(
                        full_weights.index_select(0, selected_total).sum()
                    )
                    oracle_mass = float(
                        full_weights.index_select(0, oracle).sum()
                    )
                    methods = {
                        "proxy_topk_matched": (
                            torch.softmax(
                                exact_scores.index_select(0, selected_total),
                                dim=0,
                            )
                            @ head_value.index_select(0, selected_total)
                        ),
                        "oracle_topk_matched": (
                            torch.softmax(
                                exact_scores.index_select(0, oracle), dim=0
                            )
                            @ head_value.index_select(0, oracle)
                        ),
                        "centroid_control_no_sample": combine_selected_and_tail(
                            exact_scores,
                            proxy_scores,
                            head_value,
                            selected_heavy,
                            centroid_base_y,
                            centroid_base_z,
                            1.0,
                        ),
                        "gaussian_control_no_sample": combine_selected_and_tail(
                            exact_scores,
                            proxy_scores,
                            head_value,
                            selected_heavy,
                            gaussian_base_y,
                            gaussian_base_z,
                            1.0,
                        ),
                        "gaussian_control_variate": combine_selected_and_tail(
                            exact_scores,
                            proxy_scores,
                            head_value,
                            selected_heavy,
                            cv_y,
                            cv_z,
                            1.0,
                        ),
                        **coreset_outputs,
                    }
                    overlap = int(torch.isin(reservoir, selected_heavy).sum())
                    for method, output in methods.items():
                        rows.append(
                            {
                                "text": text_path.stem,
                                "rank": rank,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "query_offset": query_offset,
                                "method": method,
                                **output_metrics(output, reference_output),
                                "selected_mass": method_selected_mass.get(
                                    method, selected_mass
                                ),
                                "oracle_mass": oracle_mass,
                                "proxy_top1_recall": method_top1_recall.get(
                                    method,
                                    float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            selected_total,
                                        )[0]
                                    ),
                                ),
                                "total_reads": total_reads,
                                "heavy_reads": heavy_count,
                                "reservoir_reads": int(reservoir.numel()),
                                "unique_reads": (
                                    heavy_count + reservoir.numel() - overlap
                                    if method == "gaussian_control_variate"
                                    else (
                                        heavy_count
                                        if method.endswith("no_sample")
                                        else total_reads
                                    )
                                ),
                                "auxiliary_bits_per_token": (
                                    float(
                                        coreset_diagnostics[method]["bits_per_token"]
                                    )
                                    if method in coreset_method_models
                                    else (
                                        float(
                                            conditional_model[
                                                "moment_bits_per_token"
                                            ]
                                        )
                                        + float(
                                            gaussian_model[
                                                "moment_bits_per_token"
                                            ]
                                        )
                                        if method.startswith(
                                            ("centroid_", "gaussian_")
                                        )
                                        else 0.0
                                    )
                                ),
                                "cv_diagnostics": (
                                    cv_diagnostics
                                    if method == "gaussian_control_variate"
                                    else coreset_diagnostics.get(method, {})
                                ),
                            }
                        )
    return rows, model_metadata


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["rank"], row["method"])].append(row)
    result = []
    for (rank, method), group in sorted(groups.items()):
        errors = torch.tensor([row["relative_l2"] for row in group])
        result.append(
            {
                "rank": rank,
                "method": method,
                "conditions": len(group),
                "relative_l2_mean": float(errors.mean()),
                "relative_l2_p90": float(torch.quantile(errors, 0.9)),
                "relative_l2_worst": float(errors.max()),
                "cosine_mean": sum(row["cosine"] for row in group) / len(group),
                "norm_ratio_mean": sum(row["norm_ratio"] for row in group)
                / len(group),
                "selected_mass_mean": sum(
                    row["selected_mass"] for row in group
                )
                / len(group),
                "oracle_mass_mean": sum(row["oracle_mass"] for row in group)
                / len(group),
                "proxy_top1_recall_mean": sum(
                    row["proxy_top1_recall"] for row in group
                )
                / len(group),
                "unique_reads_mean": sum(row["unique_reads"] for row in group)
                / len(group),
                "auxiliary_bits_per_token": sum(
                    row["auxiliary_bits_per_token"] for row in group
                )
                / len(group),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    needed = args.history_tokens + args.query_tokens
    all_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    for text_path in args.texts:
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=needed,
        ).input_ids[0, :needed]
        if token_ids.numel() < needed:
            raise ValueError(f"{text_path} contains fewer than {needed} tokens")
        rows, current_metadata = evaluate_text(text_path, token_ids, args)
        all_rows.extend(rows)
        metadata = current_metadata
    payload = {
        "schema": "qksieve_control_variate_layer0_probe_v1",
        "contract": {
            "scope": "real Qwen3 layer-0 Q/K/V; not end-to-end model quality",
            "model": str(args.model),
            "texts": [str(path) for path in args.texts],
            "history_tokens": args.history_tokens,
            "query_tokens": args.query_tokens,
            "ranks": [int(item) for item in args.ranks.split(",")],
            "block_size": args.block_size,
            "read_fraction": args.read_fraction,
            "samples_per_block": args.samples_per_block,
            "coreset_clusters": args.coreset_clusters,
            "coreset_moment_bits": args.coreset_moment_bits,
            "coreset_value_bits": args.coreset_value_bits,
            "coreset_full_key_bits": args.coreset_full_key_bits,
            "taylor_variance_bits": args.taylor_variance_bits,
            "taylor_cross_bits": args.taylor_cross_bits,
            "taylor_cross_key_dim": args.taylor_cross_key_dim,
            "taylor_cross_value_dim": args.taylor_cross_value_dim,
            "joint_value_projection_dim": args.joint_value_projection_dim,
            "joint_value_weight": args.joint_value_weight,
            "linear_group_blocks": args.linear_group_blocks,
            "model_metadata": metadata,
        },
        "aggregate": summarize(all_rows),
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
