from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def grouped_scores(
    key: torch.Tensor,
    query: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    kv_heads = int(key.shape[0])
    grouped_query = query.reshape(kv_heads, groups, query.shape[-1])
    return torch.einsum("hnd,hgd->hgn", key, grouped_query).reshape(
        kv_heads * groups, key.shape[1]
    )


def aggregate_signal(
    signal: torch.Tensor, mode: str, groups: int
) -> torch.Tensor:
    if mode == "max":
        return signal.max()
    if mode == "p90":
        return torch.quantile(signal, 0.90)
    if mode == "mean":
        return signal.mean()
    grouped = signal.reshape(-1, groups)
    if mode == "kvmax":
        return grouped.max(dim=-1).values
    if mode == "kvmean":
        return grouped.mean(dim=-1)
    raise ValueError(f"unsupported aggregation: {mode}")


def record_quality(
    metrics: dict[str, dict[str, list[float]]],
    method: str,
    approximate_scores: torch.Tensor,
    current_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    candidate_count: int,
    keep_count: int,
) -> None:
    probabilities = torch.softmax(exact_scores, dim=-1)
    true_indices = torch.topk(exact_scores, k=keep_count, dim=-1).indices
    candidate_indices = torch.topk(
        approximate_scores, k=candidate_count, dim=-1
    ).indices
    candidate_exact = torch.gather(exact_scores, -1, candidate_indices)
    local_keep = torch.topk(candidate_exact, k=keep_count, dim=-1).indices
    selected_indices = torch.gather(candidate_indices, -1, local_keep)
    candidate_hit = torch.gather(
        torch.zeros_like(exact_scores, dtype=torch.bool).scatter_(
            -1, candidate_indices, True
        ),
        -1,
        true_indices,
    ).float().mean(dim=-1)
    selected_hit = torch.gather(
        torch.zeros_like(exact_scores, dtype=torch.bool).scatter_(
            -1, selected_indices, True
        ),
        -1,
        true_indices,
    ).float().mean(dim=-1)
    retained_mass = torch.gather(
        probabilities, -1, selected_indices
    ).sum(dim=-1)
    score_scale = current_scores.std(dim=-1).clamp_min(1.0e-8)
    normalized_rmse = (
        (approximate_scores - current_scores).square().mean(dim=-1).sqrt()
        / score_scale
    )
    for value in candidate_hit.tolist():
        metrics[method]["candidate_recall_exact_topk"].append(float(value))
    for value in selected_hit.tolist():
        metrics[method]["reranked_topk_recall"].append(float(value))
    for value in retained_mass.tolist():
        metrics[method]["retained_attention_mass"].append(float(value))
    for value in normalized_rmse.tolist():
        metrics[method]["normalized_score_rmse"].append(float(value))


def evaluate_trace(
    trace_path: Path,
    device: torch.device,
    projection_dim: int,
    update_rank: int,
    thresholds: tuple[float, ...],
    aggregations: tuple[str, ...],
    periodic_intervals: tuple[int, ...],
    topk_refresh_counts: tuple[int, ...],
    relative_risk_ratios: tuple[float, ...],
    relative_max_refresh: int,
    candidate_fraction: float,
    keep_fraction: float,
    metrics: dict[str, dict[str, list[float]]],
    refreshes: dict[str, list[float]],
    gate_signals: dict[str, list[float]],
) -> dict[str, object]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)
    layer_steps: dict[int, int] = {}

    for layer, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        layer_steps[layer] = len(records)
        if len(records) < 2:
            continue
        first_key = records[0].get("key")
        if first_key is None:
            raise RuntimeError("first layer record must contain K")
        key = first_key.to(device).float()[0]
        scaling = float(records[0]["scaling"])
        queries = torch.stack(
            [record["query"].to(device).float()[0, :, 0] for record in records]
        )
        query_heads = int(queries.shape[1])
        kv_heads = int(key.shape[0])
        groups = query_heads // kv_heads
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        candidate_count = max(1, math.ceil(candidate_fraction * history_count))
        keep_count = max(1, math.ceil(keep_fraction * history_count))

        sampled_key = key[:, ::32]
        second_moment = torch.einsum(
            "hnd,hne->hde", sampled_key, sampled_key
        ) / float(sampled_key.shape[1])
        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        basis = eigenvectors[..., -projection_dim:]
        spectral_weights = eigenvalues[..., -projection_dim:].clamp_min(1.0e-12)
        projected_key = torch.einsum("hnd,hdm->hnm", key, basis)
        projected_queries = torch.einsum(
            "tqd,qdm->tqm",
            queries,
            basis.repeat_interleave(groups, dim=0),
        )
        query_weights = spectral_weights.repeat_interleave(groups, dim=0)
        initial_scores = grouped_scores(
            projected_key, projected_queries[0], groups
        )

        adaptive_states = {}
        for aggregation in aggregations:
            for threshold in thresholds:
                method = f"spectral_{aggregation}_tau{threshold:g}"
                adaptive_states[method] = {
                    "scores": initial_scores.clone(),
                    "residual": torch.zeros_like(projected_queries[0]),
                    "aggregation": aggregation,
                    "threshold": threshold,
                }
        periodic_states = {
            interval: initial_scores.clone() for interval in periodic_intervals
        }
        topk_states = {
            count: {
                "scores": initial_scores.clone(),
                "residual": torch.zeros_like(projected_queries[0]),
            }
            for count in topk_refresh_counts
        }
        relative_states = {
            ratio: {
                "scores": initial_scores.clone(),
                "residual": torch.zeros_like(projected_queries[0]),
            }
            for ratio in relative_risk_ratios
        }

        for step in range(1, len(records)):
            current_query = projected_queries[step]
            delta = current_query - projected_queries[step - 1]
            tail_delta = torch.zeros_like(delta)
            tail_delta[..., -update_rank:] = delta[..., -update_rank:]
            omitted_delta = delta - tail_delta
            tail_correction = grouped_scores(projected_key, tail_delta, groups)
            current_scores = grouped_scores(projected_key, current_query, groups)
            exact_scores = grouped_scores(key, queries[step], groups) * scaling

            record_quality(
                metrics,
                "current_pca64",
                current_scores,
                current_scores,
                exact_scores,
                candidate_count,
                keep_count,
            )
            for interval, cached_scores in periodic_states.items():
                method = f"periodic_r{interval}"
                if step % interval == 0:
                    cached_scores.copy_(current_scores)
                    refreshes[method].append(1.0)
                else:
                    cached_scores.add_(tail_correction)
                    refreshes[method].append(0.0)
                record_quality(
                    metrics,
                    method,
                    cached_scores,
                    current_scores,
                    exact_scores,
                    candidate_count,
                    keep_count,
                )

            current_energy = (
                current_query.square() * query_weights
            ).sum(dim=-1).clamp_min(1.0e-12)
            for method, state in adaptive_states.items():
                residual = state["residual"]
                residual.add_(omitted_delta)
                residual_energy = (residual.square() * query_weights).sum(dim=-1)
                relative_drift = torch.sqrt(residual_energy / current_energy)
                signal = aggregate_signal(
                    relative_drift, str(state["aggregation"]), groups
                )
                refresh_mask = signal > float(state["threshold"])
                if signal.ndim == 0:
                    gate_signals[method].append(float(signal.item()))
                    should_refresh = bool(refresh_mask)
                    if should_refresh:
                        state["scores"].copy_(current_scores)
                        residual.zero_()
                        refreshes[method].append(1.0)
                    else:
                        state["scores"].add_(tail_correction)
                        refreshes[method].append(0.0)
                else:
                    gate_signals[method].extend(
                        float(value) for value in signal.tolist()
                    )
                    query_refresh_mask = refresh_mask.repeat_interleave(groups)
                    state["scores"][query_refresh_mask] = current_scores[
                        query_refresh_mask
                    ]
                    state["scores"][~query_refresh_mask] += tail_correction[
                        ~query_refresh_mask
                    ]
                    residual[query_refresh_mask] = 0.0
                    refreshes[method].extend(
                        float(value) for value in refresh_mask.float().tolist()
                    )
                record_quality(
                    metrics,
                    method,
                    state["scores"],
                    current_scores,
                    exact_scores,
                    candidate_count,
                    keep_count,
                )

            for refresh_count, state in topk_states.items():
                method = f"spectral_kv_top{refresh_count}"
                residual = state["residual"]
                residual.add_(omitted_delta)
                residual_energy = (residual.square() * query_weights).sum(dim=-1)
                relative_drift = torch.sqrt(residual_energy / current_energy)
                kv_risk = relative_drift.reshape(kv_heads, groups).mean(dim=-1)
                refresh_indices = torch.topk(
                    kv_risk, k=min(refresh_count, kv_heads)
                ).indices
                refresh_mask = torch.zeros(
                    kv_heads, dtype=torch.bool, device=device
                )
                refresh_mask[refresh_indices] = True
                query_refresh_mask = refresh_mask.repeat_interleave(groups)
                state["scores"][query_refresh_mask] = current_scores[
                    query_refresh_mask
                ]
                state["scores"][~query_refresh_mask] = (
                    state["scores"][~query_refresh_mask]
                    + tail_correction[~query_refresh_mask]
                )
                residual[query_refresh_mask] = 0.0
                refreshes[method].extend(
                    float(value) for value in refresh_mask.float().tolist()
                )
                gate_signals[method].extend(
                    float(value) for value in kv_risk.tolist()
                )
                record_quality(
                    metrics,
                    method,
                    state["scores"],
                    current_scores,
                    exact_scores,
                    candidate_count,
                    keep_count,
                )

            for ratio, state in relative_states.items():
                method = f"spectral_relative_r{ratio:g}_cap{relative_max_refresh}"
                residual = state["residual"]
                residual.add_(omitted_delta)
                residual_energy = (residual.square() * query_weights).sum(dim=-1)
                relative_drift = torch.sqrt(residual_energy / current_energy)
                kv_risk = relative_drift.reshape(kv_heads, groups).mean(dim=-1)
                refresh_mask = kv_risk >= ratio * kv_risk.max()
                if int(refresh_mask.sum()) > relative_max_refresh:
                    refresh_indices = torch.topk(
                        kv_risk, k=min(relative_max_refresh, kv_heads)
                    ).indices
                    refresh_mask.zero_()
                    refresh_mask[refresh_indices] = True
                query_refresh_mask = refresh_mask.repeat_interleave(groups)
                state["scores"][query_refresh_mask] = current_scores[
                    query_refresh_mask
                ]
                state["scores"][~query_refresh_mask] += tail_correction[
                    ~query_refresh_mask
                ]
                residual[query_refresh_mask] = 0.0
                refreshes[method].extend(
                    float(value) for value in refresh_mask.float().tolist()
                )
                gate_signals[method].extend(
                    float(value) for value in kv_risk.tolist()
                )
                record_quality(
                    metrics,
                    method,
                    state["scores"],
                    current_scores,
                    exact_scores,
                    candidate_count,
                    keep_count,
                )

    return {"path": str(trace_path), "layer_steps": layer_steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--update_rank", type=int, default=16)
    parser.add_argument("--thresholds", default="0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.2")
    parser.add_argument("--aggregations", default="max,p90,mean")
    parser.add_argument("--periodic_intervals", default="2,4,8")
    parser.add_argument("--topk_refresh_counts", default="1,2,3")
    parser.add_argument("--relative_risk_ratios", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--relative_max_refresh", type=int, default=2)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--keep_fraction", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    thresholds = tuple(float(item) for item in args.thresholds.split(",") if item)
    aggregations = tuple(item for item in args.aggregations.split(",") if item)
    periodic_intervals = tuple(
        int(item) for item in args.periodic_intervals.split(",") if item
    )
    topk_refresh_counts = tuple(
        int(item) for item in args.topk_refresh_counts.split(",") if item
    )
    relative_risk_ratios = tuple(
        float(item) for item in args.relative_risk_ratios.split(",") if item
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    refreshes: dict[str, list[float]] = defaultdict(list)
    gate_signals: dict[str, list[float]] = defaultdict(list)
    traces = [
        evaluate_trace(
            path,
            device,
            args.projection_dim,
            args.update_rank,
            thresholds,
            aggregations,
            periodic_intervals,
            topk_refresh_counts,
            relative_risk_ratios,
            args.relative_max_refresh,
            args.candidate_fraction,
            args.keep_fraction,
            metrics,
            refreshes,
            gate_signals,
        )
        for path in args.trace_paths
    ]
    method_names = sorted(metrics)
    report = {
        "traces": traces,
        "projection_dim": args.projection_dim,
        "update_rank": args.update_rank,
        "thresholds": thresholds,
        "aggregations": aggregations,
        "periodic_intervals": periodic_intervals,
        "topk_refresh_counts": topk_refresh_counts,
        "relative_risk_ratios": relative_risk_ratios,
        "relative_max_refresh": args.relative_max_refresh,
        "candidate_fraction": args.candidate_fraction,
        "keep_fraction": args.keep_fraction,
        "note": (
            "FP32 PCA trace simulation. The gate accumulates omitted low-variance "
            "query drift and normalizes it by the PCA covariance metric."
        ),
        "metrics": {
            method: {
                metric: summarize(values)
                for metric, values in sorted(metrics[method].items())
            }
            for method in method_names
        },
        "refresh": {
            method: {
                "rate": float(np.mean(refreshes[method]))
                if refreshes[method]
                else 1.0,
                "average_scan_dimensions": (
                    args.projection_dim * float(np.mean(refreshes[method]))
                    + args.update_rank
                    * (1.0 - float(np.mean(refreshes[method])))
                )
                if refreshes[method]
                else float(args.projection_dim),
                "gate_signal": summarize(gate_signals[method])
                if gate_signals[method]
                else None,
            }
            for method in method_names
            if method != "current_pca64"
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
