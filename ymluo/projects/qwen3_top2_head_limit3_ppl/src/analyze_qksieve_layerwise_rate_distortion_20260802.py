#!/usr/bin/env python
"""Request-local, layer-output-aware rate-distortion audit for QKSieve.

The practical target is a reused long-context session.  Dense calibration
queries from the beginning of the session define a request-local QK basis and
measure candidate errors.  A joint Key/Value profile is then frozen for each
KV head and evaluated only on later queries.

Unlike per-head audits, errors are first concatenated across query heads and
passed through the model's real attention output projection.  Profile choices
therefore optimize the quantity consumed by the residual stream and retain
cross-head cancellation.  No task label, learned router, test query, oracle
profile, or Full-attention fallback is used by the reported selectors.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open

from analyze_automatic_spectral_rate_allocation_20260727 import (
    FULL_KV_BITS,
    GROUP_SIZE,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors
from analyze_qk_progressive_refinement_20260727 import (
    allocation_rate,
    quantized_bands,
    reconstruct,
)
from analyze_qksieve_conditional_value_moments_20260802 import (
    combine_selected_and_tail,
    conditional_tail_numerator,
    fit_block_models,
    fit_gaussian_tilt_moments,
    gaussian_tilt_tail_statistics,
    gaussian_tilt_tail_statistics_hybrid,
    gaussian_tilt_tail_statistics_selected_conditioned,
    tail_statistics,
)
from analyze_qksieve_joint_key_value_rate_20260802 import (
    MomentProfile,
    active_dimensions,
    parse_csv,
    parse_ints,
    parse_moments,
    profile_allocation,
)


@dataclass(frozen=True)
class Candidate:
    label: str
    key_profile: str
    moment_profile: str
    rate_bits: float


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in specification.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if not tensor.numel():
        return {}
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "maximum": float(tensor.max()),
    }


def load_o_proj(model_dir: Path, layer: int) -> torch.Tensor:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    name = f"model.layers.{layer}.self_attn.o_proj.weight"
    shard = index["weight_map"].get(name)
    if shard is None:
        raise KeyError(f"{name} is absent from {index_path}")
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def relative_energy(error: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = error.double().square().sum()
    denominator = reference.double().square().sum().clamp_min(1.0e-24)
    return float(torch.sqrt(numerator / denominator))


def calibrate_proxy_scores(
    proxy_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    mode: str,
    sample_count: int,
) -> tuple[torch.Tensor, float, float]:
    if mode == "unit":
        return proxy_scores, 1.0, 0.0
    token_count = int(proxy_scores.numel())
    positions = sampled_positions(token_count, sample_count, proxy_scores.device)
    proxy_sample = proxy_scores.index_select(0, positions).float()
    exact_sample = exact_scores.index_select(0, positions).float()
    centered_proxy = proxy_sample - proxy_sample.mean()
    centered_exact = exact_sample - exact_sample.mean()
    if mode == "sample_std":
        slope = centered_exact.square().mean().sqrt() / centered_proxy.square().mean().sqrt().clamp_min(1.0e-8)
    elif mode == "sample_ls":
        slope = (centered_proxy * centered_exact).mean() / centered_proxy.square().mean().clamp_min(1.0e-8)
    else:
        raise ValueError(f"unsupported tail calibration mode: {mode}")
    slope = slope.clamp(0.25, 4.0)
    intercept = exact_sample.mean() - slope * proxy_sample.mean()
    return (
        slope * proxy_scores + intercept,
        float(slope),
        float(intercept),
    )


def sampled_positions(
    token_count: int,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    count = min(token_count, sample_count)
    return torch.floor(
        (torch.arange(count, device=device) + 0.5) * token_count / count
    ).long()


def estimate_log_partition(
    exact_scores: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Estimate log(sum(exp(score))) from uniformly spaced exact probes."""
    token_count = int(exact_scores.numel())
    positions = sampled_positions(token_count, sample_count, exact_scores.device)
    sampled = exact_scores.index_select(0, positions).float()
    return torch.logsumexp(sampled, dim=0) + math.log(token_count / positions.numel())


def shared_output_bound_priority(
    calibrated_scores: torch.Tensor,
    log_partitions: torch.Tensor,
    log_leverage: torch.Tensor,
    leverage_lambda: float,
) -> torch.Tensor:
    """Per-token upper-bound contribution after merging the GQA groups."""
    if log_leverage.dim() == 1:
        log_leverage = log_leverage[None, :]
    if log_leverage.shape[0] not in (1, calibrated_scores.shape[0]):
        raise ValueError("leverage must be shared or have one row per GQA group")
    normalized = calibrated_scores - log_partitions[:, None]
    return torch.logsumexp(
        normalized + leverage_lambda * log_leverage,
        dim=0,
    )


def shared_normalized_max_priority(
    calibrated_scores: torch.Tensor,
    log_partitions: torch.Tensor,
    log_leverage: torch.Tensor,
    leverage_lambda: float,
) -> torch.Tensor:
    """Kernel-friendly GQA approximation to the summed output-error bound."""
    if log_leverage.dim() == 1:
        log_leverage = log_leverage[None, :]
    if log_leverage.shape[0] not in (1, calibrated_scores.shape[0]):
        raise ValueError("leverage must be shared or have one row per GQA group")
    return (
        calibrated_scores
        - log_partitions[:, None]
        + leverage_lambda * log_leverage
    ).amax(dim=0)


def conditional_value_leverage(
    coordinates: torch.Tensor,
    values: torch.Tensor,
    model: dict[str, torch.Tensor | int | float],
    bits: int,
    projection_grams: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Return quantized conditional-residual leverage and storage rate.

    With ``projection_grams=None`` this is log ||V - E[V|X]||.  Otherwise each
    row is log ||W_g (V - E[V|X])|| for one GQA output-projection slice.
    """
    mean_x = model["mean_x"]
    mean_v = model["mean_v"]
    linear_map = model["linear_map"]
    linear_group_ids = model["linear_group_ids"]
    block_size = int(model["block_size"])
    assert isinstance(mean_x, torch.Tensor)
    assert isinstance(mean_v, torch.Tensor)
    assert isinstance(linear_map, torch.Tensor)
    assert isinstance(linear_group_ids, torch.Tensor)
    token_count = int(coordinates.shape[0])
    block_ids = torch.arange(token_count, device=coordinates.device) // block_size
    token_map = linear_map.index_select(
        0, linear_group_ids.index_select(0, block_ids).long()
    )
    centered = coordinates.float() - mean_x.index_select(0, block_ids).float()
    predicted = mean_v.index_select(0, block_ids).float() + torch.einsum(
        "ndi,ni->nd", token_map.float(), centered
    )
    residual = values.float() - predicted
    if projection_grams is None:
        log_norm = torch.linalg.vector_norm(residual, dim=-1)
    else:
        if projection_grams.dim() != 3:
            raise ValueError("projection grams must have shape [groups, D, D]")
        squared = torch.einsum(
            "nd,gde,ne->gn", residual, projection_grams.float(), residual
        )
        log_norm = squared.clamp_min(1.0e-16).sqrt()
    log_norm = log_norm.clamp_min(1.0e-8).log()
    squeeze = log_norm.dim() == 1
    rows = log_norm[None, :] if squeeze else log_norm
    if bits >= 16:
        return log_norm, 16.0 * rows.shape[0]
    if bits not in (2, 4, 8):
        raise ValueError("value-leverage bits must be 2, 4, 8, or 16")
    block_count = int(model["block_count"])
    padded_count = block_count * block_size
    if padded_count > token_count:
        rows = torch.nn.functional.pad(
            rows, (0, padded_count - token_count), mode="replicate"
        )
    blocked = rows.reshape(rows.shape[0], block_count, block_size)
    minimum = blocked.amin(dim=2, keepdim=True)
    maximum = blocked.amax(dim=2, keepdim=True)
    levels = float((1 << bits) - 1)
    scale = ((maximum - minimum) / levels).clamp_min(1.0e-8)
    codes = torch.round((blocked - minimum) / scale).clamp(0, levels)
    reconstructed = (minimum + codes * scale).reshape(rows.shape[0], -1)
    reconstructed = reconstructed[:, :token_count]
    metadata_bits_per_token = (
        32.0 * block_count * rows.shape[0] / token_count
    )
    rate = float(bits * rows.shape[0]) + metadata_bits_per_token
    return (reconstructed[0] if squeeze else reconstructed), rate


def per_step_relative_l2(
    error: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    return torch.linalg.vector_norm(error.float(), dim=-1) / torch.linalg.vector_norm(
        reference.float(), dim=-1
    ).clamp_min(1.0e-12)


def configuration_metrics(
    choices: tuple[str, ...],
    errors: list[dict[str, torch.Tensor]],
    candidates: list[dict[str, Candidate]],
    reference: torch.Tensor,
) -> dict[str, Any]:
    total_error = torch.stack(
        [errors[head][label] for head, label in enumerate(choices)], dim=0
    ).sum(dim=0)
    rates = [candidates[head][label].rate_bits for head, label in enumerate(choices)]
    per_step = per_step_relative_l2(total_error, reference)
    return {
        "relative_l2": relative_energy(total_error, reference),
        "relative_l2_per_step": [float(value) for value in per_step.cpu()],
        "aux_ratio": sum(rates) / (len(rates) * FULL_KV_BITS),
        "rate_bits_mean": sum(rates) / len(rates),
        "choices": list(choices),
    }


def coordinate_descent(
    *,
    start: tuple[str, ...],
    errors: list[dict[str, torch.Tensor]],
    candidates: list[dict[str, Candidate]],
    reference: torch.Tensor,
    rate_weight: float,
    maximum_passes: int,
) -> tuple[str, ...]:
    choices = list(start)
    total_error = torch.stack(
        [errors[head][label] for head, label in enumerate(choices)], dim=0
    ).sum(dim=0)
    total_rate = sum(
        candidates[head][label].rate_bits for head, label in enumerate(choices)
    )
    reference_energy = reference.double().square().sum().clamp_min(1.0e-24)
    kv_heads = len(choices)

    for _ in range(maximum_passes):
        changed = False
        for head in range(kv_heads):
            old_label = choices[head]
            old_error = errors[head][old_label]
            old_rate = candidates[head][old_label].rate_bits
            base_error = total_error - old_error
            base_rate = total_rate - old_rate
            best_label = old_label
            best_objective = math.inf
            best_error = total_error
            best_rate = total_rate
            for label, candidate in candidates[head].items():
                proposed_error = base_error + errors[head][label]
                distortion = float(
                    proposed_error.double().square().sum() / reference_energy
                )
                proposed_rate = base_rate + candidate.rate_bits
                rate_ratio = proposed_rate / (kv_heads * FULL_KV_BITS)
                objective = distortion + rate_weight * rate_ratio
                tie = (objective, proposed_rate, label)
                best_tie = (best_objective, best_rate, best_label)
                if tie < best_tie:
                    best_label = label
                    best_objective = objective
                    best_error = proposed_error
                    best_rate = proposed_rate
            if best_label != old_label:
                choices[head] = best_label
                total_error = best_error
                total_rate = best_rate
                changed = True
        if not changed:
            break
    return tuple(choices)


def pareto_frontier(configurations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in configurations:
        key = tuple(item["choices"])
        old = unique.get(key)
        if old is None or item["relative_l2"] < old["relative_l2"]:
            unique[key] = item
    ordered = sorted(
        unique.values(), key=lambda item: (item["aux_ratio"], item["relative_l2"])
    )
    frontier: list[dict[str, Any]] = []
    best_distortion = math.inf
    for item in ordered:
        if item["relative_l2"] < best_distortion - 1.0e-12:
            frontier.append(item)
            best_distortion = item["relative_l2"]
    return frontier


def select_one_standard_error(frontier: list[dict[str, Any]]) -> dict[str, Any]:
    best = min(frontier, key=lambda item: item["relative_l2"])
    squared = torch.tensor(best["relative_l2_per_step"], dtype=torch.float64).square()
    standard_error = float(squared.std(unbiased=True) / math.sqrt(len(squared)))
    threshold = float(squared.mean()) + standard_error
    eligible = [
        item
        for item in frontier
        if sum(value * value for value in item["relative_l2_per_step"])
        / len(item["relative_l2_per_step"])
        <= threshold
    ]
    selected = min(eligible, key=lambda item: (item["aux_ratio"], item["relative_l2"]))
    return {
        **selected,
        "selection_threshold_mse": threshold,
        "best_calibration_mse": float(squared.mean()),
        "best_calibration_mse_standard_error": standard_error,
    }


def select_knee(frontier: list[dict[str, Any]]) -> dict[str, Any]:
    if len(frontier) <= 2:
        return min(frontier, key=lambda item: item["relative_l2"])
    rates = torch.tensor([item["aux_ratio"] for item in frontier], dtype=torch.float64)
    distortions = torch.tensor(
        [max(item["relative_l2"], 1.0e-12) for item in frontier],
        dtype=torch.float64,
    ).log()
    x = (rates - rates.min()) / (rates.max() - rates.min()).clamp_min(1.0e-12)
    y = (distortions - distortions.min()) / (
        distortions.max() - distortions.min()
    ).clamp_min(1.0e-12)
    start = torch.stack((x[0], y[0]))
    end = torch.stack((x[-1], y[-1]))
    line = end - start
    line_norm = torch.linalg.vector_norm(line).clamp_min(1.0e-12)
    points = torch.stack((x, y), dim=-1)
    offsets = points - start
    distances = torch.abs(line[0] * offsets[:, 1] - line[1] * offsets[:, 0])
    distances = distances / line_norm
    index = int(torch.argmax(distances))
    return {**frontier[index], "knee_distance": float(distances[index])}


def build_frontier(
    *,
    errors: list[dict[str, torch.Tensor]],
    candidates: list[dict[str, Candidate]],
    reference: torch.Tensor,
    rate_weights: tuple[float, ...],
    maximum_passes: int,
) -> list[dict[str, Any]]:
    cheapest = tuple(
        min(items.values(), key=lambda item: (item.rate_bits, item.label)).label
        for items in candidates
    )
    individual_best = tuple(
        min(
            items,
            key=lambda label: float(errors[head][label].double().square().sum()),
        )
        for head, items in enumerate(candidates)
    )
    common = sorted(set.intersection(*(set(items) for items in candidates)))
    starts = {cheapest, individual_best}
    for label in common:
        starts.add(tuple(label for _ in candidates))

    configurations: list[dict[str, Any]] = []
    for start in sorted(starts):
        configurations.append(
            configuration_metrics(start, errors, candidates, reference)
        )
    for rate_weight in rate_weights:
        for start in (cheapest, individual_best):
            choices = coordinate_descent(
                start=start,
                errors=errors,
                candidates=candidates,
                reference=reference,
                rate_weight=rate_weight,
                maximum_passes=maximum_passes,
            )
            item = configuration_metrics(choices, errors, candidates, reference)
            item["rate_weight"] = rate_weight
            configurations.append(item)
    return pareto_frontier(configurations)


def profile_description(
    choices: list[str], candidates: list[dict[str, Candidate]]
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for head, label in enumerate(choices):
        candidate = candidates[head][label]
        counts[f"{candidate.key_profile}+{candidate.moment_profile}"] += 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--max_test_steps", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.04)
    parser.add_argument(
        "--candidate_policy",
        choices=(
            "per_head",
            "gqa_shared_max",
            "gqa_shared_normalized_max",
            "gqa_shared_output_bound",
        ),
        default="per_head",
    )
    parser.add_argument(
        "--tail_score_calibration",
        choices=("unit", "sample_ls", "sample_std"),
        default="unit",
    )
    parser.add_argument("--tail_score_sample_count", type=int, default=256)
    parser.add_argument(
        "--tail_estimator",
        choices=(
            "token_exact",
            "gaussian_diag",
            "gaussian_full",
            "gaussian_diag_conditioned",
            "gaussian_full_conditioned",
            "gaussian_diag_hybrid",
            "gaussian_full_hybrid",
        ),
        default="token_exact",
        help=(
            "Accumulate omitted proxy moments token-wise or approximate them "
            "from precomputed block Gaussian moments."
        ),
    )
    parser.add_argument(
        "--selection_priority",
        choices=("qk", "conditional_value_bound"),
        default="qk",
    )
    parser.add_argument("--value_leverage_bits", type=int, default=4)
    parser.add_argument("--value_leverage_lambda", type=float, default=1.0)
    parser.add_argument(
        "--value_leverage_space",
        choices=("value", "oproj_per_group"),
        default="value",
        help=(
            "Measure conditional Value residuals directly or after each "
            "GQA query head's o_proj slice."
        ),
    )
    parser.add_argument(
        "--output_group_gain",
        choices=("unit", "spectral"),
        default="unit",
        help=(
            "Weight GQA groups uniformly or by the spectral norm of their "
            "o_proj slice when forming a shared output-error bound."
        ),
    )
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument(
        "--key_profiles",
        default=(
            "fixed200_b48,auto80,fixed400_b80,fixed410_b112,"
            "fixed4221_b208,fixed4421_b240"
        ),
    )
    parser.add_argument(
        "--moment_profiles",
        default="4x1024x4,8x1024x8,16x512x8,32x128x8",
    )
    parser.add_argument(
        "--linear_group_blocks",
        default="1",
        help="Blocks sharing one K-to-V linear map; 0 shares globally.",
    )
    parser.add_argument(
        "--linear_fit_stride",
        type=int,
        default=1,
        help="Use every nth history token when fitting the shared K-to-V map.",
    )
    parser.add_argument(
        "--rate_weights",
        default="0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10",
    )
    parser.add_argument("--maximum_passes", type=int, default=12)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 < args.fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1)")
    if args.linear_fit_stride <= 0:
        raise ValueError("linear_fit_stride must be positive")
    if args.tail_score_sample_count <= 1:
        raise ValueError("tail_score_sample_count must exceed one")
    if args.value_leverage_bits not in (2, 4, 8, 16):
        raise ValueError("value_leverage_bits must be 2, 4, 8, or 16")
    if args.value_leverage_lambda < 0.0:
        raise ValueError("value_leverage_lambda must be non-negative")
    traces = tuple(Path(item) for item in parse_csv(args.traces))
    requested_layers = set(parse_ints(args.layers))
    key_profiles = parse_csv(args.key_profiles)
    moment_profiles = parse_moments(args.moment_profiles)
    linear_group_blocks_values = parse_ints(args.linear_group_blocks)
    rate_weights = parse_floats(args.rate_weights)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_results: list[dict[str, Any]] = []

    for trace_path in traces:
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if layer in requested_layers:
                by_layer[layer].append(record)

        for layer, records in sorted(by_layer.items()):
            records.sort(key=lambda item: int(item.get("step", 0)))
            calibration_records = records[: args.calibration_steps]
            test_records = records[args.calibration_steps :]
            if args.max_test_steps > 0:
                test_records = test_records[: args.max_test_steps]
            if not calibration_records or not test_records:
                raise ValueError(f"layer {layer} lacks calibration or test records")
            evaluation_records = calibration_records + test_records
            state = next(
                record
                for record in records
                if isinstance(record.get("key"), torch.Tensor)
                and isinstance(record.get("value"), torch.Tensor)
            )
            key = state["key"].to(device).float()[0]
            value = state["value"].to(device).float()[0]
            scaling = float(state["scaling"])
            queries = torch.stack(
                [record["query"].to(device).float()[0, :, 0, :] for record in evaluation_records],
                dim=0,
            )
            calibration_queries = queries[: len(calibration_records)]
            steps, query_heads, head_dim = queries.shape
            kv_heads, token_count, _ = key.shape
            if query_heads % kv_heads:
                raise ValueError("query heads must be divisible by KV heads")
            groups = query_heads // kv_heads
            keep = min(token_count, max(1, math.ceil(args.fraction * token_count)))

            o_proj = load_o_proj(args.model_dir, layer).to(device).float()
            if o_proj.shape[1] != query_heads * head_dim:
                raise ValueError(
                    f"o_proj input {o_proj.shape[1]} != {query_heads}*{head_dim}"
                )
            full_heads = torch.empty(
                steps, query_heads, head_dim, device=device, dtype=torch.float32
            )
            exact_sparse_heads = torch.empty_like(full_heads)
            projected_errors: list[dict[str, torch.Tensor]] = []
            candidates: list[dict[str, Candidate]] = []
            tail_slopes: list[float] = []
            tail_intercepts: list[float] = []
            partition_log_errors: list[float] = []
            gaussian_negative_blocks: list[float] = []
            gaussian_mass_deficits: list[float] = []
            gaussian_negative_variances: list[float] = []
            gaussian_repaired_blocks: list[float] = []

            for kv_head in range(kv_heads):
                q_start = kv_head * groups
                q_end = q_start + groups
                head_key = key[kv_head]
                head_value = value[kv_head]
                head_queries = queries[:, q_start:q_end]
                exact_scores = torch.einsum("tgd,nd->tgn", head_queries, head_key)
                exact_scores = exact_scores * scaling
                full_weights = torch.softmax(exact_scores, dim=-1)
                full_group = torch.einsum("tgn,nd->tgd", full_weights, head_value)
                full_heads[:, q_start:q_end] = full_group
                for step in range(steps):
                    for group in range(groups):
                        selected = torch.topk(
                            exact_scores[step, group], k=keep, sorted=False
                        ).indices
                        exact_sparse_heads[step, q_start + group] = (
                            torch.softmax(exact_scores[step, group, selected], dim=0)
                            @ head_value[selected]
                        )

                head_calibration = calibration_queries[:, q_start:q_end].reshape(
                    -1, head_dim
                )
                query_factor, key_factor, _ = qk_balanced_factors(
                    head_key[:: args.sample_stride],
                    head_calibration,
                    args.query_shrinkage,
                )
                raw_coordinates = head_key @ key_factor
                projected_calibration = head_calibration @ query_factor
                bands = quantized_bands(raw_coordinates, projected_calibration)
                head_errors: dict[str, torch.Tensor] = {}
                head_candidates: dict[str, Candidate] = {}
                o_slice = o_proj[:, q_start * head_dim : q_end * head_dim]
                projection_grams = torch.stack(
                    [
                        o_slice[
                            :, group * head_dim : (group + 1) * head_dim
                        ].T
                        @ o_slice[
                            :, group * head_dim : (group + 1) * head_dim
                        ]
                        for group in range(groups)
                    ],
                    dim=0,
                )
                if args.output_group_gain == "spectral":
                    output_group_log_gains = torch.stack(
                        [
                            torch.linalg.matrix_norm(
                                o_slice[
                                    :, group * head_dim : (group + 1) * head_dim
                                ],
                                ord=2,
                            )
                            .clamp_min(1.0e-12)
                            .log()
                            for group in range(groups)
                        ]
                    )
                else:
                    output_group_log_gains = torch.zeros(
                        groups, device=device, dtype=torch.float32
                    )

                for key_name in key_profiles:
                    allocation = profile_allocation(
                        key_name,
                        raw_coordinates,
                        projected_calibration,
                        bands,
                        15,
                    )
                    key_bits = float(GROUP_SIZE * allocation_rate(allocation))
                    proxy_coordinates = reconstruct(bands, allocation).float()
                    active = active_dimensions(allocation, device)
                    conditional_coordinates = proxy_coordinates.index_select(1, active)
                    effective_profiles = sorted(
                        {
                            MomentProfile(
                                min(profile.coordinate_dim, int(active.numel())),
                                profile.block_size,
                                profile.moment_bits,
                            )
                            for profile in moment_profiles
                        }
                    )
                    models = {
                        (profile, linear_group_blocks): fit_block_models(
                            conditional_coordinates[:, : profile.coordinate_dim],
                            head_value,
                            profile.block_size,
                            args.ridge,
                            profile.moment_bits,
                            linear_group_blocks,
                            args.linear_fit_stride,
                        )
                        for profile in effective_profiles
                        for linear_group_blocks in linear_group_blocks_values
                    }
                    tilt_models = {}
                    if args.tail_estimator != "token_exact":
                        covariance_mode = (
                            "diag"
                            if "diag" in args.tail_estimator
                            else "full"
                        )
                        tilt_models = {
                            profile: fit_gaussian_tilt_moments(
                                conditional_coordinates,
                                profile.block_size,
                                profile.moment_bits,
                                covariance_mode,
                            )
                            for profile in effective_profiles
                        }
                    value_leverages: dict[
                        tuple[MomentProfile, int], tuple[torch.Tensor, float]
                    ] = {}
                    for model_key, model in models.items():
                        if args.selection_priority == "conditional_value_bound":
                            value_leverages[model_key] = conditional_value_leverage(
                                conditional_coordinates[
                                    :, : model_key[0].coordinate_dim
                                ],
                                head_value,
                                model,
                                args.value_leverage_bits,
                                (
                                    projection_grams
                                    if args.value_leverage_space
                                    == "oproj_per_group"
                                    else None
                                ),
                            )
                        else:
                            value_leverages[model_key] = (
                                torch.zeros(
                                    token_count,
                                    device=device,
                                    dtype=torch.float32,
                                ),
                                0.0,
                            )
                    candidate_groups = {
                        key: torch.empty_like(full_group) for key in models
                    }
                    projected_queries = (head_queries @ query_factor).reshape(
                        -1, head_dim
                    )
                    proxy_queries = torch.stack(
                        [query_int8(query).float() for query in projected_queries],
                        dim=0,
                    ).reshape(steps, groups, head_dim)
                    proxy_scores = torch.einsum(
                        "tgd,nd->tgn", proxy_queries, proxy_coordinates
                    ) * scaling
                    for step in range(steps):
                        calibrated_step = []
                        calibration_parameters = []
                        for group in range(groups):
                            calibrated_proxy, slope, intercept = (
                                calibrate_proxy_scores(
                                    proxy_scores[step, group],
                                    exact_scores[step, group],
                                    args.tail_score_calibration,
                                    args.tail_score_sample_count,
                                )
                            )
                            calibrated_step.append(calibrated_proxy)
                            calibration_parameters.append((slope, intercept))
                            tail_slopes.append(slope)
                            tail_intercepts.append(intercept)
                        calibrated_scores = torch.stack(
                            calibrated_step, dim=0
                        )
                        log_partitions = torch.stack(
                            [
                                estimate_log_partition(
                                    exact_scores[step, group],
                                    args.tail_score_sample_count,
                                )
                                for group in range(groups)
                            ]
                        )
                        for group in range(groups):
                            partition_log_errors.append(
                                float(
                                    log_partitions[group]
                                    - torch.logsumexp(
                                        exact_scores[step, group].float(), dim=0
                                    )
                                )
                            )
                        shared_selected: dict[
                            tuple[MomentProfile, int], torch.Tensor
                        ] = {}
                        if args.candidate_policy != "per_head":
                            for model_key, (leverage, _) in value_leverages.items():
                                leverage_rows = (
                                    leverage[None, :]
                                    if leverage.dim() == 1
                                    else leverage
                                )
                                if args.candidate_policy == "gqa_shared_max":
                                    priority = (
                                        calibrated_scores
                                        + output_group_log_gains[:, None]
                                        + args.value_leverage_lambda * leverage_rows
                                    ).amax(dim=0)
                                elif (
                                    args.candidate_policy
                                    == "gqa_shared_normalized_max"
                                ):
                                    priority = shared_normalized_max_priority(
                                        calibrated_scores
                                        + output_group_log_gains[:, None],
                                        log_partitions,
                                        leverage_rows,
                                        args.value_leverage_lambda,
                                    )
                                else:
                                    priority = shared_output_bound_priority(
                                        calibrated_scores
                                        + output_group_log_gains[:, None],
                                        log_partitions,
                                        leverage_rows,
                                        args.value_leverage_lambda,
                                    )
                                shared_selected[model_key] = torch.topk(
                                    priority,
                                    k=keep,
                                    sorted=False,
                                ).indices
                        for group in range(groups):
                            calibrated_proxy = calibrated_scores[group]
                            tail_cache: dict[
                                tuple[MomentProfile, int],
                                tuple[torch.Tensor, torch.Tensor],
                            ] = {}
                            for (profile, linear_group_blocks), model in models.items():
                                model_key = (profile, linear_group_blocks)
                                leverage, _ = value_leverages[model_key]
                                group_leverage = (
                                    leverage
                                    if leverage.dim() == 1
                                    else leverage[group]
                                )
                                selected = (
                                    shared_selected[model_key]
                                    if args.candidate_policy != "per_head"
                                    else torch.topk(
                                        calibrated_proxy
                                        + args.value_leverage_lambda
                                        * group_leverage,
                                        k=keep,
                                        sorted=False,
                                    ).indices
                                )
                                cache_key = model_key
                                if cache_key not in tail_cache:
                                    if args.tail_estimator == "token_exact":
                                        denominator, weighted_x, _ = tail_statistics(
                                            calibrated_proxy,
                                            conditional_coordinates[
                                                :, : profile.coordinate_dim
                                            ],
                                            head_value,
                                            selected,
                                            profile.block_size,
                                        )
                                    else:
                                        slope, intercept = calibration_parameters[
                                            group
                                        ]
                                        score_direction = (
                                            float(slope)
                                            * scaling
                                            * proxy_queries[
                                                step, group
                                            ].index_select(0, active)
                                        )
                                        if args.tail_estimator.endswith(
                                            "_conditioned"
                                        ):
                                            denominator, weighted_x, diagnostics = (
                                                gaussian_tilt_tail_statistics_selected_conditioned(
                                                    calibrated_proxy,
                                                    score_direction,
                                                    intercept,
                                                    conditional_coordinates,
                                                    conditional_coordinates[
                                                        :, : profile.coordinate_dim
                                                    ],
                                                    selected,
                                                    tilt_models[profile],
                                                )
                                            )
                                        elif args.tail_estimator.endswith(
                                            "_hybrid"
                                        ):
                                            denominator, weighted_x, diagnostics = (
                                                gaussian_tilt_tail_statistics_hybrid(
                                                    calibrated_proxy,
                                                    score_direction,
                                                    intercept,
                                                    conditional_coordinates,
                                                    conditional_coordinates[
                                                        :, : profile.coordinate_dim
                                                    ],
                                                    selected,
                                                    tilt_models[profile],
                                                )
                                            )
                                        else:
                                            denominator, weighted_x, diagnostics = (
                                                gaussian_tilt_tail_statistics(
                                                    calibrated_proxy,
                                                    score_direction,
                                                    intercept,
                                                    conditional_coordinates[
                                                        :, : profile.coordinate_dim
                                                    ],
                                                    selected,
                                                    tilt_models[profile],
                                                )
                                            )
                                        gaussian_negative_blocks.append(
                                            diagnostics[
                                                "negative_block_fraction"
                                            ]
                                        )
                                        gaussian_mass_deficits.append(
                                            diagnostics[
                                                "selected_mass_deficit_ratio"
                                            ]
                                        )
                                        gaussian_negative_variances.append(
                                            diagnostics.get(
                                                "negative_variance_fraction",
                                                0.0,
                                            )
                                        )
                                        gaussian_repaired_blocks.append(
                                            diagnostics.get(
                                                "repaired_block_fraction",
                                                0.0,
                                            )
                                        )
                                    tail_cache[cache_key] = denominator, weighted_x
                                denominator, weighted_x = tail_cache[cache_key]
                                tail_numerator = conditional_tail_numerator(
                                    denominator, weighted_x, model
                                )
                                candidate_groups[(profile, linear_group_blocks)][
                                    step, group
                                ] = (
                                    combine_selected_and_tail(
                                        exact_scores[step, group],
                                        calibrated_proxy,
                                        head_value,
                                        selected,
                                        tail_numerator,
                                        denominator.sum(),
                                        1.0,
                                    )
                                )
                    for (profile, linear_group_blocks), output in candidate_groups.items():
                        model = models[(profile, linear_group_blocks)]
                        _, leverage_rate = value_leverages[
                            (profile, linear_group_blocks)
                        ]
                        actual_group_blocks = int(model["linear_group_blocks"])
                        label = (
                            f"{key_name}+{profile.label}_g{actual_group_blocks}"
                            f"+{args.selection_priority}"
                        )
                        delta = (output - full_group).reshape(steps, groups * head_dim)
                        projected = delta @ o_slice.T
                        rate = (
                            key_bits
                            + float(model["moment_bits_per_token"])
                            + leverage_rate
                        )
                        if args.tail_estimator != "token_exact":
                            rate += float(
                                tilt_models[profile]["moment_bits_per_token"]
                            )
                        head_errors[label] = projected
                        head_candidates[label] = Candidate(
                            label=label,
                            key_profile=key_name,
                            moment_profile=(
                                f"{profile.label}_g{actual_group_blocks}"
                            ),
                            rate_bits=rate,
                        )
                    del proxy_coordinates, conditional_coordinates, models, tilt_models

                projected_errors.append(head_errors)
                candidates.append(head_candidates)
                del exact_scores, full_weights, full_group, raw_coordinates, bands

            full_projected = full_heads.reshape(steps, -1) @ o_proj.T
            exact_sparse_projected = exact_sparse_heads.reshape(steps, -1) @ o_proj.T
            exact_sparse_error = exact_sparse_projected - full_projected
            calibration_slice = slice(0, len(calibration_records))
            test_slice = slice(len(calibration_records), steps)
            calibration_errors = [
                {label: value[calibration_slice] for label, value in head.items()}
                for head in projected_errors
            ]
            test_errors = [
                {label: value[test_slice] for label, value in head.items()}
                for head in projected_errors
            ]
            calibration_reference = full_projected[calibration_slice]
            test_reference = full_projected[test_slice]

            frontier = build_frontier(
                errors=calibration_errors,
                candidates=candidates,
                reference=calibration_reference,
                rate_weights=rate_weights,
                maximum_passes=args.maximum_passes,
            )
            selections = {
                "one_standard_error": select_one_standard_error(frontier),
                "pareto_knee": select_knee(frontier),
                "minimum_distortion": min(
                    frontier, key=lambda item: item["relative_l2"]
                ),
                "minimum_rate": min(frontier, key=lambda item: item["aux_ratio"]),
            }
            selection_rows: dict[str, Any] = {}
            for rule, calibration_result in selections.items():
                choices = tuple(calibration_result["choices"])
                test_result = configuration_metrics(
                    choices, test_errors, candidates, test_reference
                )
                selection_rows[rule] = {
                    "calibration": calibration_result,
                    "test": test_result,
                    "profile_distribution": profile_description(
                        list(choices), candidates
                    ),
                }

            frontier_rows = []
            for item in frontier:
                choices = tuple(item["choices"])
                test_result = configuration_metrics(
                    choices, test_errors, candidates, test_reference
                )
                frontier_rows.append(
                    {
                        "calibration_relative_l2": item["relative_l2"],
                        "test_relative_l2": test_result["relative_l2"],
                        "aux_ratio": item["aux_ratio"],
                        "rate_bits_mean": item["rate_bits_mean"],
                        "choices": item["choices"],
                    }
                )
            common_labels = sorted(
                set.intersection(*(set(items) for items in candidates))
            )
            uniform_rows = []
            for label in common_labels:
                choices = tuple(label for _ in candidates)
                calibration_result = configuration_metrics(
                    choices,
                    calibration_errors,
                    candidates,
                    calibration_reference,
                )
                test_result = configuration_metrics(
                    choices, test_errors, candidates, test_reference
                )
                uniform_rows.append(
                    {
                        "profile": label,
                        "calibration_relative_l2": calibration_result[
                            "relative_l2"
                        ],
                        "test_relative_l2": test_result["relative_l2"],
                        "aux_ratio": test_result["aux_ratio"],
                        "rate_bits_mean": test_result["rate_bits_mean"],
                    }
                )
            result = {
                "trace": trace_path.stem,
                "layer": layer,
                "token_count": token_count,
                "fraction": args.fraction,
                "selected_tokens": keep,
                "calibration_steps": len(calibration_records),
                "test_steps": len(test_records),
                "linear_fit_stride": args.linear_fit_stride,
                "candidate_policy": args.candidate_policy,
                "tail_score_calibration": args.tail_score_calibration,
                "tail_estimator": args.tail_estimator,
                "selection_priority": args.selection_priority,
                "value_leverage_bits": args.value_leverage_bits,
                "value_leverage_lambda": args.value_leverage_lambda,
                "value_leverage_space": args.value_leverage_space,
                "output_group_gain": args.output_group_gain,
                "tail_score_slope": summarize(tail_slopes),
                "tail_score_intercept": summarize(tail_intercepts),
                "sampled_log_partition_error": summarize(
                    partition_log_errors
                ),
                "gaussian_negative_block_fraction": summarize(
                    gaussian_negative_blocks
                ),
                "gaussian_selected_mass_deficit_ratio": summarize(
                    gaussian_mass_deficits
                ),
                "gaussian_negative_variance_fraction": summarize(
                    gaussian_negative_variances
                ),
                "gaussian_repaired_block_fraction": summarize(
                    gaussian_repaired_blocks
                ),
                "exact_topk": {
                    "calibration_relative_l2": relative_energy(
                        exact_sparse_error[calibration_slice], calibration_reference
                    ),
                    "test_relative_l2": relative_energy(
                        exact_sparse_error[test_slice], test_reference
                    ),
                },
                "frontier": frontier_rows,
                "uniform_profiles": uniform_rows,
                "selections": selection_rows,
            }
            all_results.append(result)
            print(
                json.dumps(
                    {
                        "trace": trace_path.stem,
                        "layer": layer,
                        "frontier_points": len(frontier_rows),
                        "one_se_test": selection_rows["one_standard_error"]["test"][
                            "relative_l2"
                        ],
                        "one_se_aux": selection_rows["one_standard_error"]["test"][
                            "aux_ratio"
                        ],
                    }
                ),
                flush=True,
            )
            del key, value, queries, o_proj, full_heads, exact_sparse_heads
            torch.cuda.empty_cache()

    aggregate: dict[str, Any] = {}
    for rule in (
        "one_standard_error",
        "pareto_knee",
        "minimum_distortion",
        "minimum_rate",
    ):
        aggregate[rule] = {
            "test_relative_l2": summarize(
                item["selections"][rule]["test"]["relative_l2"]
                for item in all_results
            ),
            "calibration_relative_l2": summarize(
                item["selections"][rule]["calibration"]["relative_l2"]
                for item in all_results
            ),
            "aux_ratio": summarize(
                item["selections"][rule]["test"]["aux_ratio"]
                for item in all_results
            ),
        }
    payload = {
        "schema": "qksieve_layerwise_rate_distortion_v1",
        "contract": {
            "selection_scope": "request-local per-layer per-KV-head",
            "selection_input": "dense calibration queries only",
            "selection_target": "real o_proj output distortion",
            "test_queries_used_for_selection": False,
            "task_labels": False,
            "learned_router": False,
            "full_fallback": False,
            "linear_fit_stride": args.linear_fit_stride,
            "candidate_policy": args.candidate_policy,
            "tail_score_calibration": args.tail_score_calibration,
            "tail_score_sample_count": args.tail_score_sample_count,
            "tail_estimator": args.tail_estimator,
            "selection_priority": args.selection_priority,
            "value_leverage_bits": args.value_leverage_bits,
            "value_leverage_lambda": args.value_leverage_lambda,
            "value_leverage_space": args.value_leverage_space,
            "output_group_gain": args.output_group_gain,
        },
        "inputs": [str(path) for path in traces],
        "results": all_results,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
