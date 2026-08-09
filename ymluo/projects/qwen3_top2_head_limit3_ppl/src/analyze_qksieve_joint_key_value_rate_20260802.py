#!/usr/bin/env python
"""Audit joint Key-proxy and blockwise Value-completion rate allocation.

The experiment evaluates held-out real QKV traces.  It asks whether a cheaper
Key selector can recover Full-attention output quality by completing the
omitted Value tail with blockwise K-conditioned moments.  Exact Full outputs
and the empirical proxy-tail oracle are diagnostics only; no Full fallback is
part of any practical profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import (
    FULL_KV_BITS,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import (
    distortion_table_from_bands,
    qk_balanced_factors,
)
from analyze_qk_progressive_refinement_20260727 import (
    allocation_rate,
    quantized_bands,
    reconstruct,
)
from analyze_qksieve_conditional_value_moments_20260802 import (
    combine_selected_and_tail,
    conditional_tail_numerator,
    fit_block_models,
    output_metrics,
    tail_statistics,
)


FIXED_KEY_ALLOCATIONS: dict[str, tuple[int, ...]] = {
    "fixed200_b48": (2, 0, 0, 0, 0, 0, 0, 0),
    "fixed110_b64": (1, 1, 0, 0, 0, 0, 0, 0),
    "fixed210_b80": (2, 1, 0, 0, 0, 0, 0, 0),
    "fixed400_b80": (4, 0, 0, 0, 0, 0, 0, 0),
    "fixed410_b112": (4, 1, 0, 0, 0, 0, 0, 0),
    "fixed4221_b208": (4, 2, 2, 1, 0, 0, 0, 0),
    "fixed4421_b240": (4, 4, 2, 1, 0, 0, 0, 0),
}


@dataclass(frozen=True, order=True)
class MomentProfile:
    coordinate_dim: int
    block_size: int
    moment_bits: int

    @property
    def label(self) -> str:
        return (
            f"d{self.coordinate_dim}_b{self.block_size}_i{self.moment_bits}"
        )


def parse_csv(specification: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(x.strip() for x in specification.split(",") if x.strip()))
    if not values:
        raise ValueError("expected at least one value")
    return values


def parse_ints(specification: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected at least one float")
    return values


def parse_moments(specification: str) -> tuple[MomentProfile, ...]:
    result: list[MomentProfile] = []
    for item in parse_csv(specification):
        fields = item.lower().split("x")
        if len(fields) != 3:
            raise ValueError(f"invalid moment profile {item!r}")
        profile = MomentProfile(*(int(field) for field in fields))
        if profile.coordinate_dim <= 0 or profile.block_size <= 0:
            raise ValueError("moment dimensions and block size must be positive")
        if profile.moment_bits not in (4, 8, 16):
            raise ValueError("moment bits must be 4, 8, or 16")
        result.append(profile)
    return tuple(sorted(set(result)))


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "maximum": float(tensor.max()),
    }


def profile_allocation(
    name: str,
    raw_coordinates: torch.Tensor,
    projected_calibration: torch.Tensor,
    bands: list[torch.Tensor],
    rate_budget: int,
) -> tuple[int, ...]:
    automatic_budgets = {
        "auto80": 5,
        "auto112": 7,
        "auto240": rate_budget,
    }
    if name in automatic_budgets:
        return tuple(
            allocate_bits(
                distortion_table_from_bands(
                    raw_coordinates,
                    projected_calibration,
                    bands,
                ),
                automatic_budgets[name],
                ZERO_BIT_LEVELS,
                include_scale_metadata=True,
            )
        )
    try:
        return FIXED_KEY_ALLOCATIONS[name]
    except KeyError as error:
        raise ValueError(f"unknown Key profile {name!r}") from error


def active_dimensions(
    allocation: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    dimensions = [
        dimension
        for band, bits in enumerate(allocation)
        if bits > 0
        for dimension in range(band * GROUP_SIZE, (band + 1) * GROUP_SIZE)
    ]
    if not dimensions:
        raise RuntimeError("Key profile has no active coordinate")
    return torch.tensor(dimensions, device=device, dtype=torch.long)


def evaluate_key_profile(
    *,
    query: torch.Tensor,
    head_key: torch.Tensor,
    head_value: torch.Tensor,
    query_factor: torch.Tensor,
    proxy_coordinates: torch.Tensor,
    conditional_coordinates: torch.Tensor,
    models: dict[MomentProfile, dict[str, Any]],
    scaling: float,
    fraction: float,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    token_count = head_key.shape[0]
    keep = min(token_count, max(1, math.ceil(fraction * token_count)))
    projected_query = query @ query_factor
    proxy_query = query_int8(projected_query).float()
    exact_scores = head_key @ query * scaling
    proxy_scores = proxy_coordinates @ proxy_query * scaling
    selected = torch.topk(proxy_scores, k=keep, sorted=False).indices
    full_weights = torch.softmax(exact_scores, dim=0)
    full_output = full_weights @ head_value
    sparse_output = torch.softmax(exact_scores[selected], dim=0) @ head_value[selected]
    full_norm = torch.linalg.vector_norm(full_output).clamp_min(1.0e-12)

    tail_cache: dict[
        tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    outputs: dict[MomentProfile, torch.Tensor] = {}
    for profile, model in models.items():
        cache_key = (profile.coordinate_dim, profile.block_size)
        if cache_key not in tail_cache:
            tail_cache[cache_key] = tail_statistics(
                proxy_scores,
                conditional_coordinates[:, : profile.coordinate_dim],
                head_value,
                selected,
                profile.block_size,
            )
        denominator, weighted_x, _ = tail_cache[cache_key]
        tail_numerator = conditional_tail_numerator(
            denominator,
            weighted_x,
            model,
        )
        outputs[profile] = combine_selected_and_tail(
            exact_scores,
            proxy_scores,
            head_value,
            selected,
            tail_numerator,
            denominator.sum(),
            1.0,
        )

    first_tail = next(iter(tail_cache.values()))
    denominator, _, empirical_numerator = first_tail
    oracle_output = combine_selected_and_tail(
        exact_scores,
        proxy_scores,
        head_value,
        selected,
        empirical_numerator,
        denominator.sum(),
        1.0,
    )
    metrics: dict[str, dict[str, float]] = {
        "sparse": output_metrics(sparse_output, full_output),
        "oracle": output_metrics(oracle_output, full_output),
    }
    for profile, output in outputs.items():
        metrics[profile.label] = {
            **output_metrics(output, full_output),
            "oracle_regret_relative_l2": float(
                torch.linalg.vector_norm(output - oracle_output) / full_norm
            ),
        }
    diagnostics = {
        "selected_mass": float(full_weights[selected].sum()),
        "selected_tokens": keep,
    }
    return metrics, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--max_heldout_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--rate_budget", type=int, default=15)
    parser.add_argument("--fractions", default="0.02,0.04")
    parser.add_argument(
        "--key_profiles",
        default=(
            "fixed200_b48,fixed110_b64,auto80,fixed210_b80,fixed400_b80,"
            "fixed410_b112,fixed4221_b208,fixed4421_b240,auto240"
        ),
    )
    parser.add_argument(
        "--moment_profiles",
        default=(
            "4x1024x4,4x1024x8,8x1024x8,16x1024x8,"
            "8x512x8,16x512x8,16x256x8,16x128x8,32x128x8"
        ),
    )
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--layers", default="")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    traces = tuple(Path(x) for x in parse_csv(args.traces))
    key_profiles = parse_csv(args.key_profiles)
    moments = parse_moments(args.moment_profiles)
    fractions = parse_floats(args.fractions)
    requested_layers = set(parse_ints(args.layers)) if args.layers.strip() else None
    if any(not 0.0 < fraction < 1.0 for fraction in fractions):
        raise ValueError("fractions must lie in (0, 1)")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []

    for trace_path in traces:
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if requested_layers is None or layer in requested_layers:
                by_layer[layer].append(record)

        for layer, records in sorted(by_layer.items()):
            records.sort(key=lambda item: int(item.get("step", 0)))
            state = next(
                (
                    record
                    for record in records
                    if isinstance(record.get("key"), torch.Tensor)
                    and isinstance(record.get("value"), torch.Tensor)
                ),
                None,
            )
            if state is None:
                raise ValueError(f"layer {layer} has no materialized K/V")
            calibration_records = records[: min(args.calibration_steps, len(records))]
            heldout_records = records[len(calibration_records) :]
            if not heldout_records:
                raise ValueError(f"layer {layer} has no held-out query")
            if args.max_heldout_steps > 0:
                heldout_records = heldout_records[: args.max_heldout_steps]
            key = state["key"].to(device).float()[0]
            value = state["value"].to(device).float()[0]
            calibration = torch.stack(
                [
                    record["query"].to(device).float()[0, :, 0, :]
                    for record in calibration_records
                ],
                dim=0,
            )
            scaling = float(state["scaling"])
            kv_heads, token_count, head_dim = key.shape
            query_heads = calibration.shape[1]
            if query_heads % kv_heads:
                raise ValueError("query heads must be divisible by KV heads")
            groups = query_heads // kv_heads

            for kv_head in range(kv_heads):
                head_key = key[kv_head]
                head_value = value[kv_head]
                head_calibration = calibration[
                    :, kv_head * groups : (kv_head + 1) * groups
                ].reshape(-1, head_dim)
                query_factor, key_factor, _ = qk_balanced_factors(
                    head_key[:: args.sample_stride],
                    head_calibration,
                    args.query_shrinkage,
                )
                raw_coordinates = head_key @ key_factor
                projected_calibration = head_calibration @ query_factor
                bands = quantized_bands(raw_coordinates, projected_calibration)

                states: dict[str, dict[str, Any]] = {}
                for key_name in key_profiles:
                    allocation = profile_allocation(
                        key_name,
                        raw_coordinates,
                        projected_calibration,
                        bands,
                        args.rate_budget,
                    )
                    proxy_coordinates = reconstruct(bands, allocation).float()
                    active = active_dimensions(allocation, device)
                    conditional_coordinates = proxy_coordinates.index_select(1, active)
                    effective_moments = sorted(
                        {
                            MomentProfile(
                                min(profile.coordinate_dim, int(active.numel())),
                                profile.block_size,
                                profile.moment_bits,
                            )
                            for profile in moments
                        }
                    )
                    models = {
                        profile: fit_block_models(
                            conditional_coordinates[:, : profile.coordinate_dim],
                            head_value,
                            profile.block_size,
                            args.ridge,
                            profile.moment_bits,
                        )
                        for profile in effective_moments
                    }
                    states[key_name] = {
                        "allocation": allocation,
                        "proxy_coordinates": proxy_coordinates,
                        "conditional_coordinates": conditional_coordinates,
                        "models": models,
                        "key_bits": GROUP_SIZE * allocation_rate(allocation),
                    }

                for record in heldout_records:
                    queries = record["query"].to(device).float()[0, :, 0, :]
                    step = int(record.get("step", 0))
                    for group in range(groups):
                        query_head = kv_head * groups + group
                        for key_name, key_state in states.items():
                            metrics, diagnostics = evaluate_key_profile(
                                query=queries[query_head],
                                head_key=head_key,
                                head_value=head_value,
                                query_factor=query_factor,
                                proxy_coordinates=key_state["proxy_coordinates"],
                                conditional_coordinates=(
                                    key_state["conditional_coordinates"]
                                ),
                                models=key_state["models"],
                                scaling=scaling,
                                fraction=fractions[0],
                            )
                            for fraction_index, fraction in enumerate(fractions):
                                if fraction_index:
                                    metrics, diagnostics = evaluate_key_profile(
                                        query=queries[query_head],
                                        head_key=head_key,
                                        head_value=head_value,
                                        query_factor=query_factor,
                                        proxy_coordinates=(
                                            key_state["proxy_coordinates"]
                                        ),
                                        conditional_coordinates=(
                                            key_state["conditional_coordinates"]
                                        ),
                                        models=key_state["models"],
                                        scaling=scaling,
                                        fraction=fraction,
                                    )
                                base = {
                                    "trace": trace_path.stem,
                                    "layer": layer,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "step": step,
                                    "fraction": fraction,
                                    "key_profile": key_name,
                                    "key_allocation": "-".join(
                                        map(str, key_state["allocation"])
                                    ),
                                    "key_bits_per_token": key_state["key_bits"],
                                    **diagnostics,
                                }
                                for method in ("sparse", "oracle"):
                                    rows.append(
                                        {
                                            **base,
                                            "method": method,
                                            "moment_profile": method,
                                            "moment_bits_per_token": 0.0,
                                            "total_aux_ratio_of_full_kv": (
                                                key_state["key_bits"] / FULL_KV_BITS
                                            ),
                                            "oracle_regret_relative_l2": (
                                                0.0 if method == "oracle" else ""
                                            ),
                                            **metrics[method],
                                        }
                                    )
                                for profile, model in key_state["models"].items():
                                    moment_rate = float(model["moment_bits_per_token"])
                                    rows.append(
                                        {
                                            **base,
                                            "method": "conditional_block_moment",
                                            "moment_profile": profile.label,
                                            "coordinate_dim": profile.coordinate_dim,
                                            "block_size": profile.block_size,
                                            "moment_bits": profile.moment_bits,
                                            "moment_bits_per_token": moment_rate,
                                            "total_aux_ratio_of_full_kv": (
                                                key_state["key_bits"] + moment_rate
                                            )
                                            / FULL_KV_BITS,
                                            **metrics[profile.label],
                                        }
                                    )
            print(
                json.dumps(
                    {
                        "trace": trace_path.stem,
                        "layer": layer,
                        "rows": len(rows),
                    }
                ),
                flush=True,
            )
            del key, value
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("no rows were produced")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_query.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({field for row in rows for field in row}),
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["key_profile"],
                row["method"],
                row["moment_profile"],
                row["fraction"],
            )
        ].append(row)
    summary: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        key_name, method, moment_profile, fraction = key
        result: dict[str, Any] = {
            "key_profile": key_name,
            "method": method,
            "moment_profile": moment_profile,
            "fraction": fraction,
            "cases": len(items),
        }
        for metric in (
            "relative_l2",
            "cosine",
            "selected_mass",
            "key_bits_per_token",
            "moment_bits_per_token",
            "total_aux_ratio_of_full_kv",
        ):
            for statistic, value in summarize(float(item[metric]) for item in items).items():
                result[f"{metric}_{statistic}"] = value
        regret_values = [
            float(item["oracle_regret_relative_l2"])
            for item in items
            if item.get("oracle_regret_relative_l2") not in (None, "")
        ]
        if regret_values:
            for statistic, value in summarize(regret_values).items():
                result[f"oracle_regret_relative_l2_{statistic}"] = value
        summary.append(result)

    practical = [
        row for row in summary if row["method"] == "conditional_block_moment"
    ]
    pareto: list[dict[str, Any]] = []
    for candidate in practical:
        dominated = any(
            other["fraction"] == candidate["fraction"]
            and other["relative_l2_mean"] <= candidate["relative_l2_mean"]
            and other["total_aux_ratio_of_full_kv_mean"]
            <= candidate["total_aux_ratio_of_full_kv_mean"]
            and (
                other["relative_l2_mean"] < candidate["relative_l2_mean"]
                or other["total_aux_ratio_of_full_kv_mean"]
                < candidate["total_aux_ratio_of_full_kv_mean"]
            )
            for other in practical
        )
        if not dominated:
            pareto.append(candidate)

    report = {
        "schema": "qksieve_joint_key_value_rate_v1",
        "contract": {
            "calibration_steps": args.calibration_steps,
            "max_heldout_steps": args.max_heldout_steps,
            "heldout_disjoint_from_basis_calibration": True,
            "full_fallback": False,
            "router": False,
            "empirical_proxy_tail_oracle_is_diagnostic_only": True,
        },
        "pareto": sorted(
            pareto,
            key=lambda row: (
                row["fraction"],
                row["total_aux_ratio_of_full_kv_mean"],
            ),
        ),
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
