from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


CANDIDATE_FRACTIONS = (0.02, 0.03, 0.04, 0.06, 0.08)
THRESHOLDS = tuple(float(value) for value in np.linspace(0.0, 8.0, 33))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a sampled-error calibrated candidate budget."
    )
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--attention_fraction", type=float, default=0.02)
    parser.add_argument("--sample_fraction", type=float, default=0.0025)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "minimum": float(array.min()),
    }


def quantized_pca_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    projection_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_heads = int(query.shape[0])
    kv_heads = int(key.shape[0])
    groups = query_heads // kv_heads
    sampled_key = key[:, ::32]
    second_moment = torch.einsum(
        "hkd,hke->hde", sampled_key, sampled_key
    ) / float(sampled_key.shape[1])
    _, eigenvectors = torch.linalg.eigh(second_moment)
    basis = eigenvectors[..., -projection_dim:]
    projected_key = torch.einsum("hkd,hdm->hkm", key, basis)
    grouped_query = query.reshape(kv_heads, groups, query.shape[-1])
    projected_query = torch.einsum("hgd,hdm->hgm", grouped_query, basis)

    key_scale = projected_key.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    key_codes = torch.round(projected_key / key_scale).clamp(-7, 7)
    query_scale = projected_query.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 127.0
    query_codes = torch.round(projected_query / query_scale).clamp(-127, 127)
    integer_scores = torch.einsum("hgm,hkm->hgk", query_codes, key_codes)
    scores = integer_scores * query_scale * key_scale.squeeze(-1).unsqueeze(1)
    return scores.reshape(query_heads, key.shape[1]), basis


@torch.inference_mode()
def collect_examples(
    trace_path: Path,
    projection_dim: int,
    attention_fraction: float,
    sample_fraction: float,
    device: torch.device,
) -> list[dict[str, object]]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    examples: list[dict[str, object]] = []
    for record in payload["records"]:
        query = record["query"].to(device).float()[0, :, 0, :]
        key = record["key"].to(device).float()[0, :, :-1, :]
        scaling = float(record["scaling"])
        query_heads = int(query.shape[0])
        kv_heads = int(key.shape[0])
        groups = query_heads // kv_heads
        history_count = int(key.shape[1])
        final_count = max(1, math.ceil(attention_fraction * history_count))
        maximum_count = max(
            final_count,
            math.ceil(CANDIDATE_FRACTIONS[-1] * history_count),
        )
        expanded_key = key.repeat_interleave(groups, dim=0)
        exact_scores = torch.einsum("hkd,hd->hk", expanded_key, query) * scaling
        approximate_scores, _ = quantized_pca_scores(
            query, key, projection_dim
        )
        approximate_scores = approximate_scores * scaling
        exact_probabilities = torch.softmax(exact_scores, dim=-1)
        exact_top_scores, exact_top_indices = torch.topk(
            exact_scores, k=final_count, dim=-1, sorted=False
        )
        del exact_top_scores
        exact_top_mass = torch.gather(
            exact_probabilities, dim=-1, index=exact_top_indices
        ).sum(dim=-1)

        proxy_top_scores, proxy_top_indices = torch.topk(
            approximate_scores, k=maximum_count, dim=-1, sorted=True
        )
        sample_count = max(16, math.ceil(sample_fraction * history_count))
        sample_indices = torch.linspace(
            0,
            history_count - 1,
            sample_count,
            device=device,
        ).round().long().unique()
        sampled_error = (
            exact_scores.index_select(-1, sample_indices)
            - approximate_scores.index_select(-1, sample_indices)
        )
        sampled_error = sampled_error - sampled_error.mean(dim=-1, keepdim=True)
        error_sigma = sampled_error.square().mean(dim=-1).sqrt().clamp_min(1.0e-8)

        metrics_by_fraction: dict[float, dict[str, torch.Tensor]] = {}
        reference_score = proxy_top_scores[:, final_count - 1]
        for fraction in CANDIDATE_FRACTIONS:
            candidate_count = max(final_count, math.ceil(fraction * history_count))
            candidate_indices = proxy_top_indices[:, :candidate_count]
            candidate_exact_scores = torch.gather(
                exact_scores, dim=-1, index=candidate_indices
            )
            selected_local = torch.topk(
                candidate_exact_scores,
                k=final_count,
                dim=-1,
                sorted=False,
            ).indices
            selected_indices = torch.gather(
                candidate_indices, dim=-1, index=selected_local
            )
            selected_mass = torch.gather(
                exact_probabilities, dim=-1, index=selected_indices
            ).sum(dim=-1)
            overlap = torch.stack(
                [
                    torch.isin(selected_indices[head], exact_top_indices[head])
                    .float()
                    .mean()
                    for head in range(query_heads)
                ]
            )
            if candidate_count == final_count:
                boundary_score = proxy_top_scores[:, candidate_count]
            else:
                boundary_score = proxy_top_scores[:, candidate_count - 1]
            normalized_buffer = (reference_score - boundary_score) / error_sigma
            metrics_by_fraction[fraction] = {
                "mass_ratio": selected_mass / exact_top_mass,
                "top2_overlap": overlap,
                "normalized_buffer": normalized_buffer,
            }

        for head in range(query_heads):
            continuous_metrics = {}
            for threshold in THRESHOLDS:
                score_cutoff = (
                    proxy_top_scores[head, final_count - 1]
                    - threshold * error_sigma[head]
                )
                candidate_count = int(
                    (proxy_top_scores[head] >= score_cutoff).sum().item()
                )
                candidate_count = min(
                    maximum_count, max(final_count, candidate_count)
                )
                candidate_indices = proxy_top_indices[head, :candidate_count]
                candidate_exact_scores = exact_scores[head].index_select(
                    0, candidate_indices
                )
                selected_local = torch.topk(
                    candidate_exact_scores, k=final_count, sorted=False
                ).indices
                selected_indices = candidate_indices.index_select(
                    0, selected_local
                )
                selected_mass = exact_probabilities[head].index_select(
                    0, selected_indices
                ).sum()
                overlap = torch.isin(
                    selected_indices, exact_top_indices[head]
                ).float().mean()
                continuous_metrics[str(threshold)] = {
                    "candidate_fraction": candidate_count / history_count,
                    "mass_ratio": float(
                        (selected_mass / exact_top_mass[head]).item()
                    ),
                    "top2_overlap": float(overlap.item()),
                }
            examples.append(
                {
                    "trace": trace_path.stem,
                    "layer": int(record["layer"]),
                    "head": head,
                    "error_sigma": float(error_sigma[head].item()),
                    "fractions": {
                        str(fraction): {
                            name: float(values[head].item())
                            for name, values in metrics.items()
                        }
                        for fraction, metrics in metrics_by_fraction.items()
                    },
                    "continuous": continuous_metrics,
                }
            )
        del expanded_key, exact_scores, approximate_scores
        torch.cuda.empty_cache()
    return examples


def evaluate_policy(
    examples: list[dict[str, object]], threshold: float
) -> dict[str, object]:
    chosen_fractions: list[float] = []
    mass_ratios: list[float] = []
    overlaps: list[float] = []
    action_counts = {str(fraction): 0 for fraction in CANDIDATE_FRACTIONS}
    for example in examples:
        fraction_metrics = example["fractions"]
        chosen_fraction = CANDIDATE_FRACTIONS[-1]
        for fraction in CANDIDATE_FRACTIONS:
            metrics = fraction_metrics[str(fraction)]
            if metrics["normalized_buffer"] >= threshold:
                chosen_fraction = fraction
                break
        metrics = fraction_metrics[str(chosen_fraction)]
        chosen_fractions.append(chosen_fraction)
        mass_ratios.append(metrics["mass_ratio"])
        overlaps.append(metrics["top2_overlap"])
        action_counts[str(chosen_fraction)] += 1
    total = len(examples)
    return {
        "threshold": threshold,
        "average_candidate_fraction": float(np.mean(chosen_fractions)),
        "mass_ratio": summarize(mass_ratios),
        "top2_overlap": summarize(overlaps),
        "unsafe_mass_ratio_below_0.995": float(
            np.mean(np.asarray(mass_ratios) < 0.995)
        ),
        "unsafe_mass_ratio_below_0.999": float(
            np.mean(np.asarray(mass_ratios) < 0.999)
        ),
        "action_rates": {
            fraction: count / total for fraction, count in action_counts.items()
        },
    }


def select_threshold(results: list[dict[str, object]]) -> float:
    feasible = [
        result
        for result in results
        if result["mass_ratio"]["mean"] >= 0.999
        and result["mass_ratio"]["p10"] >= 0.995
    ]
    if not feasible:
        return float(results[-1]["threshold"])
    return float(
        min(feasible, key=lambda result: result["average_candidate_fraction"])[
            "threshold"
        ]
    )


def evaluate_continuous_policy(
    examples: list[dict[str, object]], threshold: float
) -> dict[str, object]:
    candidate_fractions = []
    mass_ratios = []
    overlaps = []
    for example in examples:
        metrics = example["continuous"][str(threshold)]
        candidate_fractions.append(metrics["candidate_fraction"])
        mass_ratios.append(metrics["mass_ratio"])
        overlaps.append(metrics["top2_overlap"])
    return {
        "threshold": threshold,
        "average_candidate_fraction": float(np.mean(candidate_fractions)),
        "candidate_fraction": summarize(candidate_fractions),
        "mass_ratio": summarize(mass_ratios),
        "top2_overlap": summarize(overlaps),
        "unsafe_mass_ratio_below_0.995": float(
            np.mean(np.asarray(mass_ratios) < 0.995)
        ),
        "unsafe_mass_ratio_below_0.999": float(
            np.mean(np.asarray(mass_ratios) < 0.999)
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    examples_by_trace = {
        trace_path.stem: collect_examples(
            trace_path,
            args.projection_dim,
            args.attention_fraction,
            args.sample_fraction,
            device,
        )
        for trace_path in args.trace_paths
    }
    sweeps = {
        trace: [evaluate_policy(examples, threshold) for threshold in THRESHOLDS]
        for trace, examples in examples_by_trace.items()
    }
    continuous_sweeps = {
        trace: [
            evaluate_continuous_policy(examples, threshold)
            for threshold in THRESHOLDS
        ]
        for trace, examples in examples_by_trace.items()
    }
    selected_thresholds = {
        trace: select_threshold(results) for trace, results in sweeps.items()
    }
    selected_continuous_thresholds = {
        trace: select_threshold(results)
        for trace, results in continuous_sweeps.items()
    }
    cross_trace = {}
    for training_trace, threshold in selected_thresholds.items():
        cross_trace[training_trace] = {
            test_trace: evaluate_policy(test_examples, threshold)
            for test_trace, test_examples in examples_by_trace.items()
            if test_trace != training_trace
        }
    continuous_cross_trace = {}
    for training_trace, threshold in selected_continuous_thresholds.items():
        continuous_cross_trace[training_trace] = {
            test_trace: evaluate_continuous_policy(test_examples, threshold)
            for test_trace, test_examples in examples_by_trace.items()
            if test_trace != training_trace
        }
    report = {
        "projection_dim": args.projection_dim,
        "attention_fraction": args.attention_fraction,
        "sample_fraction": args.sample_fraction,
        "candidate_fractions": CANDIDATE_FRACTIONS,
        "example_counts": {
            trace: len(examples) for trace, examples in examples_by_trace.items()
        },
        "selected_thresholds": selected_thresholds,
        "selected_continuous_thresholds": selected_continuous_thresholds,
        "cross_trace": cross_trace,
        "continuous_cross_trace": continuous_cross_trace,
        "sweeps": sweeps,
        "continuous_sweeps": continuous_sweeps,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_thresholds": selected_thresholds,
        "selected_continuous_thresholds": selected_continuous_thresholds,
        "cross_trace": cross_trace,
        "continuous_cross_trace": continuous_cross_trace,
    }, indent=2))


if __name__ == "__main__":
    main()
