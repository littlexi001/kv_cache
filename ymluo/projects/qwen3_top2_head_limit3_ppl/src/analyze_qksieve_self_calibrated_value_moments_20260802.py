#!/usr/bin/env python
"""Select the smallest conditional Value-moment profile during dense prefill.

Dense prefill already computes exact attention outputs.  A few prefill Query
rows can therefore measure the error of several block-moment index profiles
without labels, training, task names, or length rules.  For each layer/KV head
we choose the lowest-rate profile whose worst calibration output error is below
a numerical tolerance, then evaluate that frozen choice on later decode Query
rows.  No Full fallback is used when no profile meets the tolerance; the least
erroneous available sparse profile is selected instead.
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


@dataclass(frozen=True, order=True)
class Profile:
    coordinate_dim: int
    block_size: int
    moment_bits: int

    @property
    def label(self) -> str:
        return f"d{self.coordinate_dim}_b{self.block_size}_i{self.moment_bits}"


def parse_ints(specification: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(
        sorted({float(x) for x in specification.split(",") if x.strip()})
    )
    if not values:
        raise ValueError("expected at least one float")
    return values


def parse_profiles(specification: str) -> tuple[Profile, ...]:
    profiles: list[Profile] = []
    for item in specification.split(","):
        if not item.strip():
            continue
        fields = item.lower().split("x")
        if len(fields) != 3:
            raise ValueError(f"invalid profile {item!r}; expected dimxblockxbits")
        profile = Profile(*(int(field) for field in fields))
        if profile.coordinate_dim <= 0 or profile.block_size <= 0:
            raise ValueError("profile dimensions and block size must be positive")
        if profile.moment_bits not in (4, 8, 16):
            raise ValueError("profile moment bits must be 4, 8, or 16")
        profiles.append(profile)
    if not profiles:
        raise ValueError("expected at least one profile")
    return tuple(sorted(set(profiles)))


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "maximum": float(tensor.max()),
    }


def effective_profile(
    profile: Profile,
    available_dimensions: int,
) -> Profile:
    return Profile(
        min(profile.coordinate_dim, available_dimensions),
        profile.block_size,
        profile.moment_bits,
    )


def evaluate_query_profiles(
    query: torch.Tensor,
    head_key: torch.Tensor,
    head_value: torch.Tensor,
    query_factor: torch.Tensor,
    proxy_coordinates: torch.Tensor,
    conditional_coordinates: torch.Tensor,
    models: dict[Profile, dict[str, Any]],
    scaling: float,
    fraction: float,
) -> tuple[dict[Profile, dict[str, float]], dict[str, float]]:
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
    selected_mass = float(full_weights[selected].sum())

    tail_cache: dict[
        tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    output_tensors: dict[Profile, torch.Tensor] = {}
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
        output = combine_selected_and_tail(
            exact_scores,
            proxy_scores,
            head_value,
            selected,
            tail_numerator,
            denominator.sum(),
            1.0,
        )
        output_tensors[profile] = output

    first_tail = next(iter(tail_cache.values()))
    denominator, _, empirical_numerator = first_tail
    empirical_output = combine_selected_and_tail(
        exact_scores,
        proxy_scores,
        head_value,
        selected,
        empirical_numerator,
        denominator.sum(),
        1.0,
    )
    full_norm = torch.linalg.vector_norm(full_output).clamp_min(1.0e-12)
    outputs: dict[Profile, dict[str, float]] = {}
    for profile, output in output_tensors.items():
        outputs[profile] = {
            **output_metrics(output, full_output),
            "oracle_regret_relative_l2": float(
                torch.linalg.vector_norm(output - empirical_output) / full_norm
            ),
        }
    diagnostic = {
        "selected_mass": selected_mass,
        "sparse_relative_l2": output_metrics(
            sparse_output, full_output
        )["relative_l2"],
        "oracle_relative_l2": output_metrics(
            empirical_output, full_output
        )["relative_l2"],
    }
    return outputs, diagnostic


def choose_profile(
    calibration: dict[Profile, list[float]],
    models: dict[Profile, dict[str, Any]],
    tolerance: float,
) -> tuple[Profile, bool, float, float]:
    candidates: list[tuple[float, float, float, Profile]] = []
    for profile, errors in calibration.items():
        rate = float(models[profile]["moment_bits_per_token"])
        maximum = max(errors)
        mean = sum(errors) / len(errors)
        if maximum <= tolerance:
            candidates.append((rate, maximum, mean, profile))
    if candidates:
        rate, maximum, mean, selected = min(candidates)
        return selected, True, maximum, mean
    selected = min(
        calibration,
        key=lambda profile: (
            max(calibration[profile]),
            sum(calibration[profile]) / len(calibration[profile]),
            float(models[profile]["moment_bits_per_token"]),
        ),
    )
    errors = calibration[selected]
    return selected, False, max(errors), sum(errors) / len(errors)


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
    parser.add_argument("--fraction", type=float, default=0.04)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--tolerances", default="0.01,0.02,0.05,0.1")
    parser.add_argument(
        "--selection_mode",
        choices=("absolute", "oracle_regret"),
        default="oracle_regret",
    )
    parser.add_argument(
        "--profiles",
        default=(
            "4x1024x4,4x1024x8,8x1024x8,16x1024x8,"
            "8x512x8,16x512x8,16x256x8,16x128x8,"
            "32x128x8,32x64x8,32x32x8,48x32x8,64x32x8"
        ),
    )
    parser.add_argument("--layers", default="")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    traces = tuple(Path(x) for x in args.traces.split(",") if x.strip())
    profiles = parse_profiles(args.profiles)
    tolerances = parse_floats(args.tolerances)
    requested_layers = set(parse_ints(args.layers)) if args.layers.strip() else None
    if not 0.0 < args.fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1)")
    if any(tolerance <= 0.0 for tolerance in tolerances):
        raise ValueError("tolerances must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    head_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    for trace_path in traces:
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if requested_layers is None or layer in requested_layers:
                by_layer[layer].append(record)

        for layer, records in sorted(by_layer.items()):
            records.sort(key=lambda item: int(item.get("step", 0)))
            if len(records) <= args.calibration_steps:
                raise ValueError(f"layer {layer} has no held-out query")
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
            calibration_records = records[: args.calibration_steps]
            heldout_records = records[args.calibration_steps :]
            if args.max_heldout_steps > 0:
                heldout_records = heldout_records[: args.max_heldout_steps]
            key = state["key"].to(device).float()[0]
            value = state["value"].to(device).float()[0]
            calibration_queries = torch.stack(
                [
                    record["query"].to(device).float()[0, :, 0, :]
                    for record in calibration_records
                ],
                dim=0,
            )
            scaling = float(state["scaling"])
            kv_heads, token_count, head_dim = key.shape
            query_heads = calibration_queries.shape[1]
            if query_heads % kv_heads:
                raise ValueError("query heads must be divisible by KV heads")
            groups = query_heads // kv_heads

            for kv_head in range(kv_heads):
                head_key = key[kv_head]
                head_value = value[kv_head]
                head_calibration = calibration_queries[
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
                allocation = allocate_bits(
                    distortion_table_from_bands(
                        raw_coordinates,
                        projected_calibration,
                        bands,
                    ),
                    args.rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                proxy_coordinates = reconstruct(bands, allocation).float()
                active_dimensions = torch.tensor(
                    [
                        dimension
                        for band, bits in enumerate(allocation)
                        if bits > 0
                        for dimension in range(
                            band * GROUP_SIZE,
                            (band + 1) * GROUP_SIZE,
                        )
                    ],
                    device=device,
                    dtype=torch.long,
                )
                available_dimensions = int(active_dimensions.numel())
                if available_dimensions <= 0:
                    raise RuntimeError("QKSieve allocation has no active coordinate")
                conditional_coordinates = proxy_coordinates.index_select(
                    1, active_dimensions
                )
                effective_profiles = sorted(
                    {
                        effective_profile(profile, available_dimensions)
                        for profile in profiles
                    }
                )
                models: dict[Profile, dict[str, Any]] = {}
                for profile in effective_profiles:
                    models[profile] = fit_block_models(
                        conditional_coordinates[:, : profile.coordinate_dim],
                        head_value,
                        profile.block_size,
                        args.ridge,
                        profile.moment_bits,
                    )

                calibration_errors: dict[Profile, list[float]] = {
                    profile: [] for profile in models
                }
                for record in calibration_records:
                    queries = record["query"].to(device).float()[0, :, 0, :]
                    for group in range(groups):
                        query_head = kv_head * groups + group
                        profile_metrics, _ = evaluate_query_profiles(
                            queries[query_head],
                            head_key,
                            head_value,
                            query_factor,
                            proxy_coordinates,
                            conditional_coordinates,
                            models,
                            scaling,
                            args.fraction,
                        )
                        for profile, metrics in profile_metrics.items():
                            calibration_errors[profile].append(
                                metrics[
                                    "relative_l2"
                                    if args.selection_mode == "absolute"
                                    else "oracle_regret_relative_l2"
                                ]
                            )

                choices: dict[float, Profile] = {}
                for tolerance in tolerances:
                    selected, met, calibration_max, calibration_mean = choose_profile(
                        calibration_errors,
                        models,
                        tolerance,
                    )
                    choices[tolerance] = selected
                    key_bits = GROUP_SIZE * allocation_rate(allocation)
                    moment_rate = float(
                        models[selected]["moment_bits_per_token"]
                    )
                    head_rows.append(
                        {
                            "trace": trace_path.stem,
                            "layer": layer,
                            "kv_head": kv_head,
                            "tolerance": tolerance,
                            "met_tolerance": met,
                            "selected_profile": selected.label,
                            "coordinate_dim": selected.coordinate_dim,
                            "block_size": selected.block_size,
                            "moment_bits": selected.moment_bits,
                            "calibration_error_max": calibration_max,
                            "calibration_error_mean": calibration_mean,
                            "calibration_selection_metric": args.selection_mode,
                            "key_index_bits_per_token": key_bits,
                            "moment_bits_per_token": moment_rate,
                            "total_aux_ratio_of_full_kv": (
                                key_bits + moment_rate
                            )
                            / FULL_KV_BITS,
                        }
                    )

                for record in heldout_records:
                    queries = record["query"].to(device).float()[0, :, 0, :]
                    step = int(record.get("step", 0))
                    for group in range(groups):
                        query_head = kv_head * groups + group
                        profile_metrics, diagnostic = evaluate_query_profiles(
                            queries[query_head],
                            head_key,
                            head_value,
                            query_factor,
                            proxy_coordinates,
                            conditional_coordinates,
                            models,
                            scaling,
                            args.fraction,
                        )
                        for tolerance, selected in choices.items():
                            metrics = profile_metrics[selected]
                            head_row = next(
                                row
                                for row in reversed(head_rows)
                                if row["trace"] == trace_path.stem
                                and row["layer"] == layer
                                and row["kv_head"] == kv_head
                                and row["tolerance"] == tolerance
                            )
                            query_rows.append(
                                {
                                    **head_row,
                                    "query_head": query_head,
                                    "step": step,
                                    "relative_l2": metrics["relative_l2"],
                                    "cosine": metrics["cosine"],
                                    "oracle_regret_relative_l2": metrics[
                                        "oracle_regret_relative_l2"
                                    ],
                                    **diagnostic,
                                }
                            )
            print(
                json.dumps(
                    {
                        "trace": trace_path.stem,
                        "layer": layer,
                        "heads": len(head_rows),
                        "queries": len(query_rows),
                    }
                ),
                flush=True,
            )
            del key, value
            torch.cuda.empty_cache()

    if not head_rows or not query_rows:
        raise RuntimeError("no calibration/evaluation rows were produced")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("heads", head_rows), ("per_query", query_rows)):
        with (args.output_dir / f"{name}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=sorted({key for row in rows for key in row}),
            )
            writer.writeheader()
            writer.writerows(rows)

    summary: list[dict[str, Any]] = []
    for tolerance in tolerances:
        current_queries = [
            row for row in query_rows if row["tolerance"] == tolerance
        ]
        current_heads = [
            row for row in head_rows if row["tolerance"] == tolerance
        ]
        result: dict[str, Any] = {
            "tolerance": tolerance,
            "heads": len(current_heads),
            "heldout_query_heads": len(current_queries),
            "met_tolerance_rate": sum(
                bool(row["met_tolerance"]) for row in current_heads
            )
            / len(current_heads),
            "profile_distribution": dict(
                sorted(
                    {
                        profile: sum(
                            row["selected_profile"] == profile
                            for row in current_heads
                        )
                        for profile in {
                            str(row["selected_profile"])
                            for row in current_heads
                        }
                    }.items()
                )
            ),
        }
        for metric in (
            "relative_l2",
            "cosine",
            "selected_mass",
            "sparse_relative_l2",
            "oracle_relative_l2",
            "oracle_regret_relative_l2",
            "total_aux_ratio_of_full_kv",
        ):
            for statistic, value in summarize(
                float(row[metric]) for row in current_queries
            ).items():
                result[f"{metric}_{statistic}"] = value
        summary.append(result)
    report = {
        "schema": "qksieve_self_calibrated_value_moments_v1",
        "traces": [str(path) for path in traces],
        "contract": {
            "calibration_steps": args.calibration_steps,
            "max_heldout_steps": args.max_heldout_steps,
            "fraction": args.fraction,
            "rate_budget": args.rate_budget,
            "ridge": args.ridge,
            "selection_mode": args.selection_mode,
            "selection_rule": (
                "lowest-rate conditional-moment profile whose maximum dense-"
                "prefill calibration metric is within tolerance"
            ),
            "full_fallback": False,
            "router": False,
            "heldout_is_disjoint_from_calibration": True,
        },
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
