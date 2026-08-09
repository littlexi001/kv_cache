from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_balanced_pca_int4 import (
    grouped_scores,
    quantize_per_band_logscale_int4,
    quantize_per_token_int8,
    summarize,
)


def systematic_sample_indices(
    steps: int,
    heads: int,
    history_count: int,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    offsets = torch.arange(sample_count, device=device).reshape(1, 1, -1)
    row = torch.arange(steps, device=device).reshape(-1, 1, 1)
    head = torch.arange(heads, device=device).reshape(1, -1, 1)
    stride = max(1, history_count // sample_count)
    if stride % 2 == 0:
        stride += 1
    phase = (row * 997 + head * 131) % history_count
    return (phase + offsets * stride) % history_count


def output_metrics(
    estimate: torch.Tensor, reference: torch.Tensor
) -> tuple[list[float], list[float]]:
    cosine = F.cosine_similarity(estimate, reference, dim=-1)
    relative_l2 = (
        torch.linalg.vector_norm(estimate - reference, dim=-1)
        / torch.linalg.vector_norm(reference, dim=-1).clamp_min(1.0e-12)
    )
    return cosine.flatten().cpu().tolist(), relative_l2.flatten().cpu().tolist()


@torch.inference_mode()
def evaluate_trace(path: Path, device: torch.device) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)

    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for _, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        key_record = next((row for row in records if row.get("key") is not None), None)
        if key_record is None or len(records) < 2:
            continue
        key = key_record["key"].to(device).float()[0]
        value = key_record["value"].to(device).float()[0]
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        value = value[:, :history_count]
        query = torch.stack(
            [row["query"].to(device).float()[0, :, 0] for row in records]
        )
        kv_heads = int(key.shape[0])
        query_heads = int(query.shape[1])
        group_size = query_heads // kv_heads
        keep_count = max(1, math.ceil(0.02 * history_count))
        candidate_count = max(keep_count, math.ceil(0.08 * history_count))
        scaling = 1.0 / math.sqrt(key.shape[-1])
        exact_scores = grouped_scores(key, query, group_size)[1:] * scaling

        sampled_key = key[:, ::32]
        second_moment = torch.einsum("hnd,hne->hde", sampled_key, sampled_key)
        second_moment /= float(sampled_key.shape[1])
        _, basis = torch.linalg.eigh(second_moment)
        basis = basis[..., -64:]
        projected_key = torch.einsum("hnd,hdm->hnm", key, basis)
        grouped_query = query.reshape(
            len(records), kv_heads, group_size, query.shape[-1]
        )
        projected_query = torch.einsum("thgd,hdm->thgm", grouped_query, basis)
        proxy_scores = grouped_scores(
            quantize_per_band_logscale_int4(projected_key, 16, 0.25),
            quantize_per_token_int8(projected_query),
            group_size,
        )[1:]
        candidates = torch.topk(
            proxy_scores, candidate_count, dim=-1, sorted=False
        ).indices
        candidate_exact = torch.gather(exact_scores, -1, candidates)
        local = torch.topk(candidate_exact, keep_count, dim=-1, sorted=False).indices
        selected = torch.gather(candidates, -1, local)
        selected_scores = torch.gather(exact_scores, -1, selected)
        value_q = value.repeat_interleave(group_size, dim=0)

        full_outputs = []
        sparse_outputs = []
        corrected_outputs: dict[float, list[torch.Tensor]] = {
            fraction: [] for fraction in (0.001, 0.0025, 0.005)
        }
        reliable_tail_outputs: list[torch.Tensor] = []
        reliable8_tail_outputs: list[torch.Tensor] = []
        reliable_tail_alphas: list[torch.Tensor] = []
        reliable8_tail_alphas: list[torch.Tensor] = []
        fixed_shrink_outputs: dict[float, list[torch.Tensor]] = {
            alpha: [] for alpha in (0.5, 0.625, 0.75)
        }
        mass_gated_outputs: dict[tuple[float, float], list[torch.Tensor]] = {
            (alpha, threshold): []
            for alpha in (0.5, 1.0)
            for threshold in (0.7, 0.8, 0.9, 0.95)
        }
        mass_gated_active: dict[tuple[float, float], list[torch.Tensor]] = {
            key: [] for key in mass_gated_outputs
        }
        actual_selected_mass = []
        estimated_selected_mass: dict[float, list[torch.Tensor]] = {
            fraction: [] for fraction in corrected_outputs
        }
        sample_indices = {
            fraction: systematic_sample_indices(
                exact_scores.shape[0],
                query_heads,
                history_count,
                max(1, math.ceil(fraction * history_count)),
                device,
            )
            for fraction in corrected_outputs
        }

        for step in range(exact_scores.shape[0]):
            scores = exact_scores[step]
            full_weights = torch.softmax(scores, dim=-1)
            full_outputs.append(torch.einsum("hn,hnd->hd", full_weights, value_q))
            selected_step = selected[step]
            selected_value = torch.gather(
                value_q,
                1,
                selected_step.unsqueeze(-1).expand(-1, -1, value.shape[-1]),
            )
            selected_step_scores = selected_scores[step]
            sparse_weights = torch.softmax(selected_step_scores, dim=-1)
            sparse_outputs.append(
                torch.einsum("hk,hkd->hd", sparse_weights, selected_value)
            )
            actual_selected_mass.append(
                torch.gather(full_weights, -1, selected_step).sum(dim=-1)
            )

            for fraction, all_indices in sample_indices.items():
                sample = all_indices[step]
                sample_scores = torch.gather(scores, -1, sample)
                sample_value = torch.gather(
                    value_q,
                    1,
                    sample.unsqueeze(-1).expand(-1, -1, value.shape[-1]),
                )
                valid = ~(
                    sample.unsqueeze(-1) == selected_step.unsqueeze(-2)
                ).any(dim=-1)
                valid_count = valid.sum(dim=-1).clamp_min(1)
                sample_scores = sample_scores.masked_fill(~valid, -torch.inf)
                max_score = torch.maximum(
                    selected_step_scores.amax(dim=-1),
                    sample_scores.amax(dim=-1),
                )
                selected_exp = torch.exp(selected_step_scores - max_score.unsqueeze(-1))
                sample_exp = torch.exp(sample_scores - max_score.unsqueeze(-1)).masked_fill(
                    ~valid, 0.0
                )
                tail_count = history_count - keep_count
                tail_scale = tail_count / valid_count.float()
                selected_denominator = selected_exp.sum(dim=-1)
                tail_denominator = tail_scale * sample_exp.sum(dim=-1)
                denominator = selected_denominator + tail_denominator
                selected_numerator = torch.einsum(
                    "hk,hkd->hd", selected_exp, selected_value
                )
                sampled_numerator = torch.einsum(
                    "hm,hmd->hd", sample_exp, sample_value
                )
                tail_numerator = tail_scale.unsqueeze(-1) * sampled_numerator
                numerator = selected_numerator + tail_numerator
                corrected_outputs[fraction].append(
                    numerator / denominator.unsqueeze(-1).clamp_min(1.0e-12)
                )
                estimated_selected_mass[fraction].append(
                    selected_denominator / denominator.clamp_min(1.0e-12)
                )
                if fraction == 0.005:
                    # Two disjoint half-samples estimate the same tail correction.
                    # Their agreement gives a training-free reliability shrinkage.
                    even = valid & (
                        torch.arange(sample.shape[-1], device=device) % 2 == 0
                    ).unsqueeze(0)
                    odd = valid & ~(
                        torch.arange(sample.shape[-1], device=device) % 2 == 0
                    ).unsqueeze(0)
                    half_outputs = []
                    for half_valid in (even, odd):
                        half_count = half_valid.sum(dim=-1).clamp_min(1)
                        half_scale = tail_count / half_count.float()
                        half_exp = sample_exp.masked_fill(~half_valid, 0.0)
                        half_denominator = (
                            selected_denominator
                            + half_scale * half_exp.sum(dim=-1)
                        )
                        half_numerator = selected_numerator + half_scale.unsqueeze(
                            -1
                        ) * torch.einsum("hm,hmd->hd", half_exp, sample_value)
                        half_outputs.append(
                            half_numerator
                            / half_denominator.unsqueeze(-1).clamp_min(1.0e-12)
                        )
                    base_output = selected_numerator / selected_denominator.unsqueeze(
                        -1
                    ).clamp_min(1.0e-12)
                    delta_even = half_outputs[0] - base_output
                    delta_odd = half_outputs[1] - base_output
                    signal_power = (delta_even + delta_odd).square().sum(dim=-1)
                    noise_power = (delta_even - delta_odd).square().sum(dim=-1)
                    reliability = signal_power / (
                        signal_power + noise_power + 1.0e-12
                    )
                    reliable_denominator = (
                        selected_denominator + reliability * tail_denominator
                    )
                    reliable_numerator = selected_numerator + reliability.unsqueeze(
                        -1
                    ) * tail_numerator
                    reliable_tail_outputs.append(
                        reliable_numerator
                        / reliable_denominator.unsqueeze(-1).clamp_min(1.0e-12)
                    )
                    reliable_tail_alphas.append(reliability)
                    reliability_dims = (
                        torch.arange(8, device=device) * value.shape[-1] // 8
                    )
                    delta_even8 = delta_even.index_select(-1, reliability_dims)
                    delta_odd8 = delta_odd.index_select(-1, reliability_dims)
                    signal_power8 = (delta_even8 + delta_odd8).square().sum(dim=-1)
                    noise_power8 = (delta_even8 - delta_odd8).square().sum(dim=-1)
                    reliability8 = signal_power8 / (
                        signal_power8 + noise_power8 + 1.0e-12
                    )
                    reliable8_denominator = (
                        selected_denominator + reliability8 * tail_denominator
                    )
                    reliable8_numerator = selected_numerator + reliability8.unsqueeze(
                        -1
                    ) * tail_numerator
                    reliable8_tail_outputs.append(
                        reliable8_numerator
                        / reliable8_denominator.unsqueeze(-1).clamp_min(1.0e-12)
                    )
                    reliable8_tail_alphas.append(reliability8)
                    for alpha, outputs in fixed_shrink_outputs.items():
                        fixed_denominator = (
                            selected_denominator + alpha * tail_denominator
                        )
                        fixed_numerator = selected_numerator + alpha * tail_numerator
                        outputs.append(
                            fixed_numerator
                            / fixed_denominator.unsqueeze(-1).clamp_min(1.0e-12)
                        )
                    estimated_mass = selected_denominator / denominator.clamp_min(
                        1.0e-12
                    )
                    for (alpha, threshold), outputs in mass_gated_outputs.items():
                        active = estimated_mass < threshold
                        gated_denominator = (
                            selected_denominator + alpha * tail_denominator
                        )
                        gated_numerator = selected_numerator + alpha * tail_numerator
                        corrected = gated_numerator / gated_denominator.unsqueeze(
                            -1
                        ).clamp_min(1.0e-12)
                        outputs.append(torch.where(active.unsqueeze(-1), corrected, base_output))
                        mass_gated_active[(alpha, threshold)].append(active.float())

        full_output = torch.stack(full_outputs)
        sparse_output = torch.stack(sparse_outputs)
        cosine, relative_l2 = output_metrics(sparse_output, full_output)
        metrics["pca64_top2"]["full_output_cosine"].extend(cosine)
        metrics["pca64_top2"]["full_output_relative_l2"].extend(relative_l2)
        actual_mass = torch.stack(actual_selected_mass)
        metrics["pca64_top2"]["selected_mass"].extend(
            actual_mass.flatten().cpu().tolist()
        )
        for fraction, outputs in corrected_outputs.items():
            name = f"pca64_top2_tail{100*fraction:g}"
            estimate = torch.stack(outputs)
            cosine, relative_l2 = output_metrics(estimate, full_output)
            metrics[name]["full_output_cosine"].extend(cosine)
            metrics[name]["full_output_relative_l2"].extend(relative_l2)
            mass_estimate = torch.stack(estimated_selected_mass[fraction])
            mass_error = (mass_estimate - actual_mass).abs()
            metrics[name]["selected_mass_absolute_error"].extend(
                mass_error.flatten().cpu().tolist()
            )
        reliable_estimate = torch.stack(reliable_tail_outputs)
        cosine, relative_l2 = output_metrics(reliable_estimate, full_output)
        metrics["pca64_top2_tail0.5_reliability"]["full_output_cosine"].extend(
            cosine
        )
        metrics["pca64_top2_tail0.5_reliability"][
            "full_output_relative_l2"
        ].extend(relative_l2)
        metrics["pca64_top2_tail0.5_reliability"]["reliability_alpha"].extend(
            torch.stack(reliable_tail_alphas).flatten().cpu().tolist()
        )
        reliable8_estimate = torch.stack(reliable8_tail_outputs)
        cosine, relative_l2 = output_metrics(reliable8_estimate, full_output)
        metrics["pca64_top2_tail0.5_reliability8"]["full_output_cosine"].extend(
            cosine
        )
        metrics["pca64_top2_tail0.5_reliability8"][
            "full_output_relative_l2"
        ].extend(relative_l2)
        metrics["pca64_top2_tail0.5_reliability8"]["reliability_alpha"].extend(
            torch.stack(reliable8_tail_alphas).flatten().cpu().tolist()
        )
        for alpha, outputs in fixed_shrink_outputs.items():
            estimate = torch.stack(outputs)
            cosine, relative_l2 = output_metrics(estimate, full_output)
            name = f"pca64_top2_tail0.5_shrink{alpha:g}"
            metrics[name]["full_output_cosine"].extend(cosine)
            metrics[name]["full_output_relative_l2"].extend(relative_l2)
        for (alpha, threshold), outputs in mass_gated_outputs.items():
            estimate = torch.stack(outputs)
            cosine, relative_l2 = output_metrics(estimate, full_output)
            name = f"pca64_top2_tail0.5_a{alpha:g}_masslt{threshold:g}"
            metrics[name]["full_output_cosine"].extend(cosine)
            metrics[name]["full_output_relative_l2"].extend(relative_l2)
            metrics[name]["active_rate"].extend(
                torch.stack(mass_gated_active[(alpha, threshold)])
                .flatten()
                .cpu()
                .tolist()
            )
        del key, value, query, exact_scores
        torch.cuda.empty_cache()

    return {
        "trace": str(path),
        "methods": {
            method: {name: summarize(values) for name, values in values_by_name.items()}
            for method, values_by_name in sorted(metrics.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "method": "sampled tail denominator and value-moment correction",
        "traces": [
            evaluate_trace(path, torch.device(args.device))
            for path in args.trace_paths
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
