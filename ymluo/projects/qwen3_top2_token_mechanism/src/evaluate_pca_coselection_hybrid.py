from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluate_coselection_expansion import (
    build_affinity_graph,
    expand_from_seeds,
    frequency_from_seeds,
)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "count": int(array.size),
    }


def grouped_scores(key: torch.Tensor, query: torch.Tensor, groups: int) -> torch.Tensor:
    grouped_query = query.reshape(query.shape[0], key.shape[0], groups, query.shape[-1])
    return torch.einsum("hnd,thgd->thgn", key, grouped_query).reshape(
        query.shape[0], key.shape[0] * groups, key.shape[1]
    )


def record_candidate_quality(
    metrics: dict[str, dict[str, list[float]]],
    method: str,
    candidate_indices: torch.Tensor,
    exact_scores: torch.Tensor,
    true_indices: torch.Tensor,
    keep_count: int,
) -> None:
    candidate_exact = exact_scores.index_select(0, candidate_indices)
    local = torch.topk(candidate_exact, k=keep_count).indices
    selected = candidate_indices.index_select(0, local)
    selected_mask = torch.zeros_like(exact_scores, dtype=torch.bool)
    selected_mask[selected] = True
    recall = selected_mask[true_indices].float().mean()
    probabilities = torch.softmax(exact_scores, dim=-1)
    mass = probabilities.index_select(0, selected).sum()
    metrics[method]["top2_position_recall"].append(float(recall.item()))
    metrics[method]["attention_mass"].append(float(mass.item()))


def gumbel_sample_without_replacement(
    scores: torch.Tensor,
    count: int,
    *,
    temperature: float,
    generator: torch.Generator,
    excluded: torch.Tensor | None = None,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    normalized = (scores.float() - scores.float().mean()) / scores.float().std().clamp_min(1.0e-6)
    uniform = torch.rand(
        normalized.shape,
        device=normalized.device,
        dtype=normalized.dtype,
        generator=generator,
    ).clamp_(1.0e-6, 1.0 - 1.0e-6)
    priority = normalized / temperature - torch.log(-torch.log(uniform))
    if excluded is not None and excluded.numel() > 0:
        priority[excluded] = -torch.inf
    return torch.topk(priority, k=count).indices


def select_seeds(
    strategy: str,
    scores: torch.Tensor,
    ranked: torch.Tensor,
    seed_count: int,
    *,
    generator: torch.Generator,
    hybrid_top_fraction: float,
    exploration_fraction: float,
    band_count: int,
) -> torch.Tensor:
    if strategy == "top":
        return ranked[:seed_count]
    if strategy == "uniform":
        return torch.randperm(
            scores.numel(), device=scores.device, generator=generator
        )[:seed_count]
    if strategy in {"score_t0p5", "score_t1"}:
        temperature = 0.5 if strategy == "score_t0p5" else 1.0
        return gumbel_sample_without_replacement(
            scores, seed_count, temperature=temperature, generator=generator
        )
    if strategy in {"hybrid_score", "hybrid_band"}:
        top_count = min(seed_count, max(1, int(math.ceil(hybrid_top_fraction * seed_count))))
        deterministic = ranked[:top_count]
        sample_count = seed_count - top_count
        if sample_count == 0:
            return deterministic
        if strategy == "hybrid_score":
            sampled = gumbel_sample_without_replacement(
                scores,
                sample_count,
                temperature=1.0,
                generator=generator,
                excluded=deterministic,
            )
        else:
            pool = ranked[top_count:band_count]
            if pool.numel() < sample_count:
                raise ValueError("uncertainty band is smaller than the requested sample")
            order = torch.randperm(pool.numel(), device=pool.device, generator=generator)
            sampled = pool.index_select(0, order[:sample_count])
        return torch.cat((deterministic, sampled))
    if strategy in {"top_plus_next", "top_plus_score", "top_plus_band"}:
        deterministic = ranked[:seed_count]
        sample_count = max(1, int(math.ceil(exploration_fraction * seed_count)))
        if strategy == "top_plus_next":
            sampled = ranked[seed_count : seed_count + sample_count]
        elif strategy == "top_plus_score":
            sampled = gumbel_sample_without_replacement(
                scores,
                sample_count,
                temperature=1.0,
                generator=generator,
                excluded=deterministic,
            )
        else:
            pool = ranked[seed_count:band_count]
            if pool.numel() < sample_count:
                raise ValueError("uncertainty band is smaller than the requested exploration sample")
            order = torch.randperm(pool.numel(), device=pool.device, generator=generator)
            sampled = pool.index_select(0, order[:sample_count])
        return torch.cat((deterministic, sampled))
    raise ValueError(f"unsupported seed strategy: {strategy}")


def evaluate_trace(
    trace_path: Path,
    *,
    device: torch.device,
    train_steps: int,
    projection_dim: int,
    seed_fraction: float,
    seed_strategies: tuple[str, ...],
    candidate_fractions: tuple[float, ...],
    neighbor_count: int,
    hybrid_top_fraction: float,
    exploration_fraction: float,
    sampling_seed: int,
    metrics: dict[str, dict[str, list[float]]],
) -> dict[str, object]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)
    layer_steps: dict[int, int] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(sampling_seed)

    for layer, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        layer_steps[layer] = len(records)
        if len(records) <= train_steps:
            continue
        key = records[0]["key"].to(device).float()[0]
        queries = torch.stack(
            [record["query"].to(device).float()[0, :, 0] for record in records]
        )
        query_heads = int(queries.shape[1])
        kv_heads = int(key.shape[0])
        groups = query_heads // kv_heads
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        keep_count = max(1, math.ceil(0.02 * history_count))
        seed_count = max(1, math.ceil(seed_fraction * history_count))
        candidate_counts = {
            fraction: max(keep_count, math.ceil(fraction * history_count))
            for fraction in candidate_fractions
        }

        exact_scores = grouped_scores(key, queries, groups)
        true_indices = torch.topk(exact_scores, k=keep_count, dim=-1).indices

        sampled_key = key[:, ::32]
        second_moment = torch.einsum("hnd,hne->hde", sampled_key, sampled_key) / float(
            sampled_key.shape[1]
        )
        _, eigenvectors = torch.linalg.eigh(second_moment)
        basis = eigenvectors[..., -projection_dim:]
        projected_key = torch.einsum("hnd,hdm->hnm", key, basis)
        grouped_query = queries.reshape(len(records), kv_heads, groups, queries.shape[-1])
        projected_query = torch.einsum("thgd,hdm->thgm", grouped_query, basis)
        pca64_scores = torch.einsum("hnm,thgm->thgn", projected_key, projected_query).reshape(
            len(records), query_heads, history_count
        )
        pca16_scores = torch.einsum(
            "hnm,thgm->thgn", projected_key[..., -16:], projected_query[..., -16:]
        ).reshape(len(records), query_heads, history_count)

        for head in range(query_heads):
            graph = build_affinity_graph(
                true_indices[:train_steps, head].cpu().numpy(),
                history_count,
                neighbor_count=neighbor_count,
            )
            for step in range(train_steps, len(records)):
                target = true_indices[step, head]
                exact = exact_scores[step, head]
                for score_name, approximate in (
                    ("pca16", pca16_scores[step, head]),
                    ("pca64", pca64_scores[step, head]),
                ):
                    maximum_count = max(candidate_counts.values())
                    direct = torch.topk(approximate, k=maximum_count).indices
                    for fraction, candidate_count in candidate_counts.items():
                        suffix = f"c{100.0 * fraction:g}"
                        record_candidate_quality(
                            metrics,
                            f"{score_name}_direct_{suffix}",
                            direct[:candidate_count],
                            exact,
                            target,
                            keep_count,
                        )
                    band_count = min(maximum_count, max(seed_count, math.ceil(0.04 * history_count)))
                    for strategy in seed_strategies:
                        seeds = select_seeds(
                            strategy,
                            approximate,
                            direct,
                            seed_count,
                            generator=generator,
                            hybrid_top_fraction=hybrid_top_fraction,
                            exploration_fraction=exploration_fraction,
                            band_count=band_count,
                        )
                        graph_candidate = expand_from_seeds(
                            seeds.cpu().numpy(), graph, maximum_count
                        )
                        frequency_candidate = frequency_from_seeds(
                            seeds.cpu().numpy(), graph, maximum_count
                        )
                        graph_tensor = torch.from_numpy(graph_candidate).to(
                            device=device, dtype=torch.long
                        )
                        frequency_tensor = torch.from_numpy(frequency_candidate).to(
                            device=device, dtype=torch.long
                        )
                        for fraction, candidate_count in candidate_counts.items():
                            suffix = f"c{100.0 * fraction:g}"
                            record_candidate_quality(
                                metrics,
                                f"{score_name}_{strategy}_frequency_{suffix}",
                                frequency_tensor[:candidate_count],
                                exact,
                                target,
                                keep_count,
                            )
                            record_candidate_quality(
                                metrics,
                                f"{score_name}_{strategy}_graph_{suffix}",
                                graph_tensor[:candidate_count],
                                exact,
                                target,
                                keep_count,
                            )

        del exact_scores, true_indices, projected_key, pca64_scores, pca16_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"path": str(trace_path), "layer_steps": layer_steps}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PCA seed plus held-out co-selection expansion.")
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--train_steps", type=int, default=8)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--seed_fraction", type=float, default=0.005)
    parser.add_argument(
        "--seed_strategies",
        default="top,uniform,score_t0p5,score_t1,hybrid_score,hybrid_band",
    )
    parser.add_argument("--candidate_fractions", default="0.02,0.04,0.08")
    parser.add_argument("--neighbor_count", type=int, default=8)
    parser.add_argument("--hybrid_top_fraction", type=float, default=0.7)
    parser.add_argument("--exploration_fraction", type=float, default=0.2)
    parser.add_argument("--sampling_seed", type=int, default=20260718)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    candidate_fractions = tuple(
        float(item) for item in args.candidate_fractions.split(",") if item
    )
    seed_strategies = tuple(item for item in args.seed_strategies.split(",") if item)
    if not 0.0 < args.seed_fraction <= min(candidate_fractions):
        raise ValueError("seed fraction must be positive and fit within every candidate budget")
    if not 0.0 < args.hybrid_top_fraction <= 1.0:
        raise ValueError("hybrid_top_fraction must be in (0, 1]")
    if not 0.0 < args.exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be in (0, 1]")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    traces = [
        evaluate_trace(
            path,
            device=device,
            train_steps=args.train_steps,
            projection_dim=args.projection_dim,
            seed_fraction=args.seed_fraction,
            seed_strategies=seed_strategies,
            candidate_fractions=candidate_fractions,
            neighbor_count=args.neighbor_count,
            hybrid_top_fraction=args.hybrid_top_fraction,
            exploration_fraction=args.exploration_fraction,
            sampling_seed=args.sampling_seed,
            metrics=metrics,
        )
        for path in args.trace_paths
    ]
    report = {
        "traces": traces,
        "train_steps": args.train_steps,
        "test_contract": "all steps after train_steps are held out",
        "projection_dim": args.projection_dim,
        "seed_fraction": args.seed_fraction,
        "seed_strategies": seed_strategies,
        "candidate_fractions": candidate_fractions,
        "neighbor_count": args.neighbor_count,
        "hybrid_top_fraction": args.hybrid_top_fraction,
        "exploration_fraction": args.exploration_fraction,
        "sampling_seed": args.sampling_seed,
        "metrics": {
            method: {name: summarize(values) for name, values in method_metrics.items()}
            for method, method_metrics in sorted(metrics.items())
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for method, values in report["metrics"].items():
        print(
            method,
            f"recall={100.0 * values['top2_position_recall']['mean']:.2f}%",
            f"mass={100.0 * values['attention_mass']['mean']:.2f}%",
        )


if __name__ == "__main__":
    main()
