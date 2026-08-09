from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluate_pca_coselection_hybrid import grouped_scores, record_candidate_quality, summarize


def select_energy_band(
    residual: torch.Tensor, eigenvalues: torch.Tensor, band_size: int
) -> int:
    if residual.ndim != 2:
        raise ValueError("residual must have shape [group_heads, dimensions]")
    if residual.shape[1] % band_size != 0:
        raise ValueError("dimensions must be divisible by band_size")
    weighted = residual.square() * eigenvalues.unsqueeze(0)
    energies = weighted.reshape(residual.shape[0], -1, band_size).sum(dim=(0, 2))
    return int(torch.argmax(energies).item())


def update_query_state(
    state: torch.Tensor, current: torch.Tensor, band: int, band_size: int
) -> torch.Tensor:
    updated = state.clone()
    start = band * band_size
    updated[:, start : start + band_size] = current[:, start : start + band_size]
    return updated


def residual_score_radius(
    residual: torch.Tensor, projected_key: torch.Tensor, band_size: int
) -> torch.Tensor:
    """Cauchy upper bound for score error, summed over PCA bands."""
    if residual.ndim != 2 or projected_key.ndim != 2:
        raise ValueError("residual and projected_key must both be matrices")
    if residual.shape[1] != projected_key.shape[1]:
        raise ValueError("residual and projected_key dimensions must match")
    if residual.shape[1] % band_size != 0:
        raise ValueError("dimensions must be divisible by band_size")
    residual_norms = residual.reshape(residual.shape[0], -1, band_size).norm(dim=-1)
    key_norms = projected_key.reshape(projected_key.shape[0], -1, band_size).norm(
        dim=-1
    )
    return residual_norms @ key_norms.T


def boundary_rescue_candidates(
    approximate_scores: torch.Tensor,
    radius: torch.Tensor,
    keep_count: int,
    candidate_count: int,
) -> torch.Tensor:
    """Keep proxy top-k, then fill the pool with optimistic-bound candidates."""
    if approximate_scores.shape != radius.shape:
        raise ValueError("approximate scores and radius must have the same shape")
    if not 0 < keep_count <= candidate_count <= approximate_scores.numel():
        raise ValueError("candidate counts must satisfy 0 < keep <= candidate <= N")
    primary = torch.topk(approximate_scores, k=keep_count).indices
    rescue_count = candidate_count - keep_count
    if rescue_count == 0:
        return primary
    optimistic = approximate_scores + radius
    optimistic = optimistic.clone()
    optimistic[primary] = -torch.inf
    rescue = torch.topk(optimistic, k=rescue_count).indices
    return torch.cat((primary, rescue))


def record_top2_output_quality(
    metrics: dict[str, dict[str, list[float]]],
    method: str,
    candidate_indices: torch.Tensor,
    exact_scores: torch.Tensor,
    true_indices: torch.Tensor,
    value: torch.Tensor,
    keep_count: int,
    scaling: float,
) -> None:
    candidate_scores = exact_scores.index_select(0, candidate_indices)
    local = torch.topk(candidate_scores, k=keep_count).indices
    selected = candidate_indices.index_select(0, local)
    selected_weights = torch.softmax(exact_scores[selected] * scaling, dim=-1)
    oracle_weights = torch.softmax(exact_scores[true_indices] * scaling, dim=-1)
    selected_output = selected_weights @ value[selected].float()
    oracle_output = oracle_weights @ value[true_indices].float()
    cosine = torch.nn.functional.cosine_similarity(
        selected_output.unsqueeze(0), oracle_output.unsqueeze(0)
    )
    relative_l2 = torch.linalg.vector_norm(selected_output - oracle_output) / torch.linalg.vector_norm(
        oracle_output
    ).clamp_min(1.0e-12)
    metrics[method]["oracle_top2_output_cosine"].append(float(cosine.item()))
    metrics[method]["oracle_top2_output_relative_l2"].append(float(relative_l2.item()))


def evaluate_trace(
    path: Path,
    *,
    projection_dim: int,
    band_size: int,
    candidate_fraction: float,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)

    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    scheduler_metrics: dict[str, list[float]] = defaultdict(list)
    layer_steps: dict[int, int] = {}
    for layer, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        layer_steps[layer] = len(records)
        if len(records) < 2:
            continue
        key_record = next((record for record in records if record.get("key") is not None), None)
        if key_record is None:
            raise ValueError(f"layer {layer} has no stored key tensor")
        key = key_record["key"].to(device).float()[0]
        value = key_record["value"].to(device).float()[0]
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        value = value[:, :history_count]
        queries = torch.stack(
            [record["query"].to(device).float()[0, :, 0] for record in records]
        )
        kv_heads = int(key.shape[0])
        query_heads = int(queries.shape[1])
        group_size = query_heads // kv_heads
        keep_count = max(1, math.ceil(0.02 * history_count))
        candidate_count = max(keep_count, math.ceil(candidate_fraction * history_count))

        exact_scores = grouped_scores(key, queries, group_size)
        true_indices = torch.topk(exact_scores, k=keep_count, dim=-1).indices
        sampled_key = key[:, ::32]
        second_moment = torch.einsum("hnd,hne->hde", sampled_key, sampled_key) / float(
            sampled_key.shape[1]
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        basis = eigenvectors[..., -projection_dim:]
        retained_eigenvalues = eigenvalues[..., -projection_dim:]
        projected_key = torch.einsum("hnd,hdm->hnm", key, basis)
        grouped_query = queries.reshape(len(records), kv_heads, group_size, queries.shape[-1])
        projected_query = torch.einsum("thgd,hdm->thgm", grouped_query, basis)

        energy_state = projected_query[0].clone()
        round_robin_state = projected_query[0].clone()
        band_count = projection_dim // band_size
        for step in range(1, len(records)):
            energy_scores: list[torch.Tensor] = []
            energy_radii: list[torch.Tensor] = []
            round_robin_scores: list[torch.Tensor] = []
            full_scores: list[torch.Tensor] = []
            for kv_head in range(kv_heads):
                current = projected_query[step, kv_head]
                residual = current - energy_state[kv_head]
                selected_band = select_energy_band(
                    residual, retained_eigenvalues[kv_head], band_size
                )
                energy_state[kv_head] = update_query_state(
                    energy_state[kv_head], current, selected_band, band_size
                )
                round_robin_band = (step - 1) % band_count
                round_robin_state[kv_head] = update_query_state(
                    round_robin_state[kv_head], current, round_robin_band, band_size
                )
                keys = projected_key[kv_head]
                energy_scores.append(energy_state[kv_head] @ keys.T)
                residual_after = current - energy_state[kv_head]
                energy_radii.append(
                    residual_score_radius(residual_after, keys, band_size)
                )
                round_robin_scores.append(round_robin_state[kv_head] @ keys.T)
                full_scores.append(current @ keys.T)
                error_energy = (
                    residual_after.square() * retained_eigenvalues[kv_head].unsqueeze(0)
                ).sum(dim=-1)
                total_energy = (
                    current.square() * retained_eigenvalues[kv_head].unsqueeze(0)
                ).sum(dim=-1).clamp_min(1.0e-12)
                scheduler_metrics["relative_residual_energy"].extend(
                    torch.sqrt(error_energy / total_energy).cpu().tolist()
                )
                scheduler_metrics["selected_band"].append(float(selected_band))

            method_scores = {
                "full_pca64": torch.cat(full_scores, dim=0),
                "round_robin_error_feedback": torch.cat(round_robin_scores, dim=0),
                "energy_error_feedback": torch.cat(energy_scores, dim=0),
            }
            concatenated_radius = torch.cat(energy_radii, dim=0)
            for head in range(query_heads):
                energy_head_scores = method_scores["energy_error_feedback"][head]
                radius = concatenated_radius[head]
                optimistic = energy_head_scores + radius
                lower = energy_head_scores - radius
                lower_threshold = torch.topk(lower, k=keep_count).values[-1]
                certified_count = int((optimistic >= lower_threshold).sum().item())
                scheduler_metrics["certified_candidate_fraction"].append(
                    certified_count / float(history_count)
                )
                scheduler_metrics["radius_to_score_std"].append(
                    float((radius.mean() / energy_head_scores.std().clamp_min(1.0e-12)).item())
                )
                method_candidates = {
                    method: torch.topk(scores[head], k=candidate_count).indices
                    for method, scores in method_scores.items()
                }
                method_candidates["energy_error_feedback_ucb"] = torch.topk(
                    optimistic, k=candidate_count
                ).indices
                method_candidates["energy_error_feedback_boundary_rescue"] = (
                    boundary_rescue_candidates(
                        energy_head_scores,
                        radius,
                        keep_count,
                        candidate_count,
                    )
                )
                for method, candidate in method_candidates.items():
                    record_candidate_quality(
                        metrics,
                        method,
                        candidate,
                        exact_scores[step, head],
                        true_indices[step, head],
                        keep_count,
                    )
                    kv_head = head // group_size
                    record_top2_output_quality(
                        metrics,
                        method,
                        candidate,
                        exact_scores[step, head],
                        true_indices[step, head],
                        value[kv_head],
                        keep_count,
                        scaling=1.0 / math.sqrt(key.shape[-1]),
                    )

        del key, value, queries, exact_scores, true_indices, sampled_key, second_moment
        del eigenvalues, eigenvectors, basis, projected_key, projected_query
        if device.type == "cuda":
            torch.cuda.empty_cache()

    quality = {
        method: {name: summarize(values) for name, values in values_by_metric.items()}
        for method, values_by_metric in metrics.items()
    }
    return {
        "path": str(path),
        "projection_dim": projection_dim,
        "band_size": band_size,
        "candidate_fraction": candidate_fraction,
        "layers": len(records_by_layer),
        "layer_steps": layer_steps,
        "scan_dimensions_after_initialization": {
            "full_pca64": projection_dim,
            "round_robin_error_feedback": band_size,
            "energy_error_feedback": band_size,
            "energy_error_feedback_ucb": band_size,
            "energy_error_feedback_boundary_rescue": band_size,
        },
        "certificate_storage": "one precomputed norm per token and PCA band",
        "test_contract": "causal state starts at step 0; quality is measured on later steps only",
        "quality": quality,
        "scheduler": {name: summarize(values) for name, values in scheduler_metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate contiguous-band spectral error feedback.")
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--band_size", type=int, default=16)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.projection_dim % args.band_size != 0:
        raise ValueError("projection_dim must be divisible by band_size")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    report = {
        "method": "covariance-weighted contiguous-band error feedback",
        "traces": [
            evaluate_trace(
                path,
                projection_dim=args.projection_dim,
                band_size=args.band_size,
                candidate_fraction=args.candidate_fraction,
                device=device,
            )
            for path in args.trace_paths
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for trace in report["traces"]:
        print(trace["path"])
        for method, metrics in trace["quality"].items():
            print(
                method,
                f"recall={100.0 * metrics['top2_position_recall']['mean']:.2f}%",
                f"mass={100.0 * metrics['attention_mass']['mean']:.2f}%",
                f"output_cos={metrics['oracle_top2_output_cosine']['mean']:.5f}",
                f"output_rel_l2={metrics['oracle_top2_output_relative_l2']['mean']:.5f}",
            )
        residual = trace["scheduler"]["relative_residual_energy"]
        print(f"residual_energy={100.0 * residual['mean']:.2f}%")


if __name__ == "__main__":
    main()
