#!/usr/bin/env python
"""Audit GQA-shared candidate selection for query-metric principal codes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from analyze_qaware_binarypc_blockmean_layer0_20260802 import (
    assert_numeric_backend_sane,
    binary_proxy_scores,
    encode_binary_principal,
    evenly_spaced_indices,
    fit_binary_principal_projection,
    quantize_log_error_norms,
    query_metric_factors,
    query_windows,
)
from analyze_qksieve_block_coreset_20260802 import (
    block_coreset_tail_statistics,
    fit_block_coreset,
)
from analyze_qksieve_conditional_value_moments_20260802 import (
    combine_selected_and_tail,
)
from analyze_qksieve_control_variate_layer0_probe_20260802 import (
    load_layer0_activations,
    output_metrics,
)


def gqa_shared_scores(
    proxy_scores: torch.Tensor,
    fraction: float,
    mode: str,
    sample_count: int = 256,
) -> torch.Tensor:
    """Combine per-query-head proxy scores into one KV-head candidate priority."""
    if proxy_scores.ndim != 2:
        raise ValueError("proxy_scores must have shape [GQA heads, history]")
    if mode == "raw_max":
        return proxy_scores.amax(dim=0)
    if mode == "mass_sum":
        normalized = proxy_scores - torch.logsumexp(proxy_scores, dim=1)[:, None]
        return torch.logsumexp(normalized, dim=0)
    if mode != "margin_max":
        raise ValueError(f"unknown GQA sharing mode: {mode}")

    sample_indices = evenly_spaced_indices(
        proxy_scores.shape[1], sample_count, proxy_scores.device
    )
    sample = proxy_scores.index_select(1, sample_indices)
    thresholds = torch.quantile(sample, 1.0 - fraction, dim=1)
    scales = sample.std(dim=1).clamp_min(1.0e-6)
    margins = (proxy_scores - thresholds[:, None]) / scales[:, None]
    return margins.amax(dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--texts", type=Path, nargs="+", required=True)
    parser.add_argument("--history_tokens", type=int, default=8192)
    parser.add_argument("--calibration_tokens", type=int, default=16)
    parser.add_argument("--query_tokens", type=int, default=4)
    parser.add_argument(
        "--calibration_source",
        choices=("post_history", "history_tail"),
        default="post_history",
    )
    parser.add_argument("--heldout_gap", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.06)
    parser.add_argument("--shared_multipliers", default="1.0,1.25,1.5")
    parser.add_argument("--shared_modes", default="raw_max,margin_max,mass_sum")
    parser.add_argument("--binary_bits", type=int, default=64)
    parser.add_argument("--projection_iterations", type=int, default=4)
    parser.add_argument("--projection_sample_stride", type=int, default=8)
    parser.add_argument(
        "--projection_sample_count",
        type=int,
        default=4096,
        help="Length-independent reservoir size; <=0 falls back to stride.",
    )
    parser.add_argument(
        "--projection_initialization",
        choices=("random", "spectral"),
        default="spectral",
    )
    parser.add_argument("--metric_shrinkage", default="oas")
    parser.add_argument(
        "--risk_lambda",
        type=float,
        default=2.0,
        help="One-sided score uncertainty multiplier; <=0 disables risk rows.",
    )
    parser.add_argument("--risk_error_bits", type=int, default=4)
    parser.add_argument("--risk_error_block_size", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--key_mean_bits", type=int, default=8)
    parser.add_argument("--value_mean_bits", type=int, default=4)
    parser.add_argument("--quantile_samples", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate_output(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    selector_coordinates: torch.Tensor,
    selector_query: torch.Tensor,
    full_query: torch.Tensor,
    coreset: Any,
    scale: float,
) -> tuple[dict[str, float], dict[str, float]]:
    full_weights = torch.softmax(exact_scores, dim=0)
    full_output = full_weights @ values
    reference = exact_scores.index_select(0, selected).amin()
    tail_z, tail_y, diagnostics = block_coreset_tail_statistics(
        selector_coordinates,
        values,
        selector_query * scale,
        selected,
        reference,
        coreset,
        selected_conditioned=False,
        full_score_coordinates=selector_coordinates,
        full_score_direction=selector_query * scale,
    )
    tail_output = combine_selected_and_tail(
        exact_scores,
        exact_scores,
        values,
        selected,
        tail_y,
        tail_z,
        1.0,
    )
    metrics = {
        **output_metrics(tail_output, full_output),
        "selected_mass": float(full_weights.index_select(0, selected).sum()),
        "top1_recall": float(
            torch.isin(torch.argmax(exact_scores)[None], selected)[0]
        ),
        "proxy_top1_recall": float(
            torch.isin(torch.argmax(proxy_scores)[None], selected)[0]
        ),
    }
    return metrics, diagnostics


def evaluate_text(
    text_path: Path,
    token_ids: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query, key, value, metadata = load_layer0_activations(args.model, token_ids)
    history_count = args.history_tokens
    history_key = key[:history_count]
    history_value = value[:history_count]
    calibration, heldout = query_windows(
        query,
        history_count,
        args.calibration_tokens,
        args.query_tokens,
        args.calibration_source,
        args.heldout_gap,
    )
    group_size = int(metadata["gqa_groups"])
    head_dim = int(metadata["head_dim"])
    scale = float(head_dim**-0.5)
    modes = args.shared_modes.split(",")
    multipliers = [float(item) for item in args.shared_multipliers.split(",")]
    rows: list[dict[str, Any]] = []

    for kv_head in range(int(metadata["kv_heads"])):
        head_key = history_key[:, kv_head].contiguous()
        head_value = history_value[:, kv_head].contiguous()
        head_calibration = calibration[
            :, kv_head * group_size : (kv_head + 1) * group_size
        ].reshape(-1, head_dim)
        query_factor, key_factor, resolved_shrinkage = query_metric_factors(
            head_calibration, args.metric_shrinkage
        )
        coordinates = head_key @ key_factor
        if args.projection_sample_count > 0:
            projection_indices = evenly_spaced_indices(
                history_count,
                args.projection_sample_count,
                head_key.device,
            )
            projection_samples = coordinates.index_select(0, projection_indices)
        else:
            projection_samples = coordinates[:: args.projection_sample_stride]
        projection = fit_binary_principal_projection(
            projection_samples,
            args.binary_bits,
            args.projection_iterations,
            seed=5000 + kv_head,
            initialization=args.projection_initialization,
        )
        codes, reconstruction_errors = encode_binary_principal(
            coordinates, projection
        )
        quantized_errors, risk_aux_bits = quantize_log_error_norms(
            reconstruction_errors,
            args.risk_error_bits,
            args.risk_error_block_size,
        )
        coreset = fit_block_coreset(
            coordinates,
            head_value,
            args.block_size,
            cluster_count=1,
            moment_bits=2,
            iterations=1,
            full_score_coordinates=coordinates,
            value_moment_bits=args.value_mean_bits,
            full_score_moment_bits=args.key_mean_bits,
        )

        for query_offset in range(args.query_tokens):
            current_queries = heldout[
                query_offset,
                kv_head * group_size : (kv_head + 1) * group_size,
            ]
            metric_queries = current_queries @ query_factor
            exact_matrix = current_queries @ head_key.T * scale
            proxy_matrix = torch.stack(
                [
                    binary_proxy_scores(codes, projection, metric_query, scale)
                    for metric_query in metric_queries
                ]
            )
            proxy_variants = {"base": (proxy_matrix, 0.0)}
            if args.risk_lambda > 0.0:
                uncertainty = (
                    metric_queries.float().norm(dim=1)[:, None]
                    * quantized_errors[None, :]
                    / float(head_dim)
                )
                proxy_variants["ucb95"] = (
                    proxy_matrix + args.risk_lambda * uncertainty,
                    risk_aux_bits,
                )
            independent_keep = max(1, math.ceil(history_count * args.fraction))

            for risk_mode, (current_proxy_matrix, selector_aux_bits) in (
                proxy_variants.items()
            ):
                for group_offset in range(group_size):
                    selected = torch.topk(
                        current_proxy_matrix[group_offset],
                        independent_keep,
                        sorted=False,
                    ).indices
                    metrics, diagnostics = evaluate_output(
                        exact_matrix[group_offset],
                        current_proxy_matrix[group_offset],
                        head_value,
                        selected,
                        coordinates,
                        metric_queries[group_offset],
                        current_queries[group_offset],
                        coreset,
                        scale,
                    )
                    rows.append(
                        {
                            "text": text_path.stem,
                            "kv_head": kv_head,
                            "query_head": kv_head * group_size + group_offset,
                            "query_offset": query_offset,
                            "selection": "independent",
                            "risk_mode": risk_mode,
                            "shared_mode": "independent",
                            "shared_multiplier": 1.0,
                            "fraction": args.fraction,
                            "selected_tokens": independent_keep,
                            "selector_aux_bits_per_token": selector_aux_bits,
                            "resolved_metric_shrinkage": resolved_shrinkage,
                            "tail_bits_per_token": float(
                                diagnostics["bits_per_token"]
                            ),
                            **metrics,
                        }
                    )

                for mode in modes:
                    shared_priority = gqa_shared_scores(
                        current_proxy_matrix,
                        args.fraction,
                        mode,
                        args.quantile_samples,
                    )
                    for multiplier in multipliers:
                        shared_keep = min(
                            history_count,
                            max(1, math.ceil(independent_keep * multiplier)),
                        )
                        shared_selected = torch.topk(
                            shared_priority, shared_keep, sorted=False
                        ).indices
                        for group_offset in range(group_size):
                            metrics, diagnostics = evaluate_output(
                                exact_matrix[group_offset],
                                current_proxy_matrix[group_offset],
                                head_value,
                                shared_selected,
                                coordinates,
                                metric_queries[group_offset],
                                current_queries[group_offset],
                                coreset,
                                scale,
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": (
                                        kv_head * group_size + group_offset
                                    ),
                                    "query_offset": query_offset,
                                    "selection": "gqa_shared",
                                    "risk_mode": risk_mode,
                                    "shared_mode": mode,
                                    "shared_multiplier": multiplier,
                                    "fraction": args.fraction,
                                    "selected_tokens": shared_keep,
                                    "selector_aux_bits_per_token": (
                                        selector_aux_bits
                                    ),
                                    "resolved_metric_shrinkage": (
                                        resolved_shrinkage
                                    ),
                                    "tail_bits_per_token": float(
                                        diagnostics["bits_per_token"]
                                    ),
                                    **metrics,
                                }
                            )
    return rows, metadata


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        groups[
            (
                row["selection"],
                row["risk_mode"],
                row["shared_mode"],
                row["shared_multiplier"],
            )
        ].append(row)
    output = []
    for (selection, risk_mode, mode, multiplier), group in sorted(groups.items()):
        errors = torch.tensor([row["relative_l2"] for row in group])
        output.append(
            {
                "selection": selection,
                "risk_mode": risk_mode,
                "shared_mode": mode,
                "shared_multiplier": multiplier,
                "conditions": len(group),
                "selected_tokens": group[0]["selected_tokens"],
                "relative_l2_mean": float(errors.mean()),
                "relative_l2_p90": float(torch.quantile(errors, 0.9)),
                "relative_l2_worst": float(errors.max()),
                "cosine_mean": sum(row["cosine"] for row in group) / len(group),
                "selected_mass_mean": sum(row["selected_mass"] for row in group)
                / len(group),
                "top1_recall_mean": sum(row["top1_recall"] for row in group)
                / len(group),
                "tail_bits_per_token": group[0]["tail_bits_per_token"],
                "selector_aux_bits_per_token": group[0][
                    "selector_aux_bits_per_token"
                ],
            }
        )
    return output


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    assert_numeric_backend_sane()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    needed = args.history_tokens + args.heldout_gap + args.query_tokens
    if args.calibration_source == "post_history":
        needed += args.calibration_tokens
    rows: list[dict[str, Any]] = []
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
        current_rows, metadata = evaluate_text(text_path, token_ids, args)
        rows.extend(current_rows)
    payload = {
        "schema": "qmetric_gqa_shared_layer0_v1",
        "contract": {
            "scope": "real Qwen3 layer-0 mechanism audit; not end-to-end quality",
            "model": str(args.model),
            "history_tokens": args.history_tokens,
            "calibration_tokens": args.calibration_tokens,
            "calibration_source": args.calibration_source,
            "heldout_gap": args.heldout_gap,
            "query_tokens": args.query_tokens,
            "fraction": args.fraction,
            "binary_bits": args.binary_bits,
            "projection_sample_count": args.projection_sample_count,
            "projection_initialization": args.projection_initialization,
            "metric_shrinkage": args.metric_shrinkage,
            "risk_lambda": args.risk_lambda,
            "model_metadata": metadata,
        },
        "aggregate": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
