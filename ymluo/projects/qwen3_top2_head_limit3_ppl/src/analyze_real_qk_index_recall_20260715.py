from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


BUDGETS = (0.005, 0.01, 0.02)
CANDIDATE_FRACTION = 0.07


def projection_matrix(
    head_dim: int, projection_dim: int, layer: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260715 + 1009 * layer + projection_dim)
    signs = torch.randint(
        0,
        2,
        (head_dim, projection_dim),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    return (2.0 * signs.float() - 1.0) / math.sqrt(projection_dim)


def orthogonal_projection_matrix(
    head_dim: int, projection_dim: int, layer: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260716 + 1013 * layer)
    matrix = torch.randn(
        (head_dim, head_dim), generator=generator, device=device
    )
    orthogonal, _ = torch.linalg.qr(matrix)
    return orthogonal[:, :projection_dim]


def quantize_dequantize(values: torch.Tensor, bits: int) -> torch.Tensor:
    maximum = 2 ** (bits - 1) - 1
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / maximum
    return torch.round(values / scale).clamp(-maximum, maximum) * scale


def index_fraction(projection_dim: int, bits: int) -> float:
    return (projection_dim * bits + 16) / (2 * 128 * 16)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "minimum": float(array.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    storage = {"qabs_top16_fp16_full_k": 0.5}

    for record in payload["records"]:
        layer = int(record["layer"])
        query = record["query"].to(device).float()[0, :, 0, :]
        key = record["key"].to(device).float()[0]
        scaling = float(record["scaling"])
        query_heads = int(query.shape[0])
        kv_heads = int(key.shape[0])
        groups = query_heads // kv_heads
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]

        score_sets: dict[str, torch.Tensor] = {}
        qabs_scores = []
        for head in range(query_heads):
            kv_head = head // groups
            dimensions = torch.topk(query[head].abs(), k=16).indices
            qabs_scores.append(
                torch.matmul(
                    key[kv_head].index_select(-1, dimensions),
                    query[head].index_select(-1, dimensions),
                )
            )
        score_sets["qabs_top16_fp16_full_k"] = torch.stack(qabs_scores)

        for projection_dim in (16, 32, 64, 96):
            projection = projection_matrix(
                int(query.shape[-1]), projection_dim, layer, device
            )
            projected_query = torch.matmul(query, projection)
            projected_key = torch.matmul(key, projection)
            expanded_projected_key = projected_key.repeat_interleave(groups, dim=0)
            score_sets[f"rp_fp16_m{projection_dim}"] = torch.einsum(
                "hkd,hd->hk", expanded_projected_key, projected_query
            )
            storage[f"rp_fp16_m{projection_dim}"] = index_fraction(
                projection_dim, 16
            )
            for bits in (4, 8):
                quantized_key = quantize_dequantize(expanded_projected_key, bits)
                method = f"rp_int{bits}_m{projection_dim}"
                score_sets[method] = torch.einsum(
                    "hkd,hd->hk", quantized_key, projected_query
                )
                storage[method] = index_fraction(projection_dim, bits)

        sampled_key = key[:, ::32]
        second_moment = torch.einsum(
            "hkd,hke->hde", sampled_key, sampled_key
        ) / sampled_key.shape[1]
        _, pca_vectors = torch.linalg.eigh(second_moment)
        for projection_dim in (32, 48, 64, 96):
            orthogonal = orthogonal_projection_matrix(
                int(query.shape[-1]), projection_dim, layer, device
            )
            projected_query = torch.matmul(query, orthogonal)
            projected_key = torch.matmul(key, orthogonal)
            expanded_projected_key = projected_key.repeat_interleave(groups, dim=0)
            for bits in (4, 8, 16):
                method = f"ortho_{'fp16' if bits == 16 else f'int{bits}'}_m{projection_dim}"
                indexed_key = (
                    expanded_projected_key
                    if bits == 16
                    else quantize_dequantize(expanded_projected_key, bits)
                )
                score_sets[method] = torch.einsum(
                    "hkd,hd->hk", indexed_key, projected_query
                )
                storage[method] = index_fraction(projection_dim, bits)

            per_head_projection = pca_vectors[..., -projection_dim:]
            pca_key = torch.einsum("hkd,hdm->hkm", key, per_head_projection)
            pca_query = torch.stack(
                [
                    torch.matmul(query[head], per_head_projection[head // groups])
                    for head in range(query_heads)
                ]
            )
            expanded_pca_key = pca_key.repeat_interleave(groups, dim=0)
            for bits in (4, 8, 16):
                method = f"pca_{'fp16' if bits == 16 else f'int{bits}'}_m{projection_dim}"
                indexed_key = (
                    expanded_pca_key
                    if bits == 16
                    else quantize_dequantize(expanded_pca_key, bits)
                )
                score_sets[method] = torch.einsum(
                    "hkd,hd->hk", indexed_key, pca_query
                )
                storage[method] = index_fraction(projection_dim, bits)

        for head in range(query_heads):
            kv_head = head // groups
            full_scores = torch.matmul(key[kv_head], query[head]) * scaling
            full_probabilities = torch.softmax(full_scores, dim=-1)
            true_indices = {
                budget: torch.topk(
                    full_scores,
                    k=max(1, math.ceil(budget * history_count)),
                ).indices
                for budget in BUDGETS
            }
            candidate_count = max(1, math.ceil(CANDIDATE_FRACTION * history_count))
            for method, approximate_scores in score_sets.items():
                candidate_indices = torch.topk(
                    approximate_scores[head], k=candidate_count
                ).indices
                candidate_set = set(candidate_indices.cpu().tolist())
                for budget in BUDGETS:
                    keep_count = max(1, math.ceil(budget * history_count))
                    selected = torch.topk(
                        approximate_scores[head], k=keep_count
                    ).indices
                    retained_mass = float(full_probabilities[selected].sum().item())
                    selected_set = set(selected.cpu().tolist())
                    true_set = set(true_indices[budget].cpu().tolist())
                    metrics[method][f"mass_at_{budget:g}"].append(retained_mass)
                    metrics[method][f"topk_overlap_at_{budget:g}"].append(
                        len(selected_set & true_set) / keep_count
                    )
                    metrics[method][f"candidate7_recall_true_{budget:g}"].append(
                        len(candidate_set & true_set) / keep_count
                    )

    report = {
        "trace": str(args.trace_path),
        "layers": [int(record["layer"]) for record in payload["records"]],
        "candidate_fraction": CANDIDATE_FRACTION,
        "methods": {
            method: {
                "logical_index_fraction_of_full_kv": storage[method],
                **{name: summarize(values) for name, values in method_metrics.items()},
            }
            for method, method_metrics in sorted(metrics.items())
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
