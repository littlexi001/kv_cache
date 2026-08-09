from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure intrinsic rank and train-only prototype coverage of state-pointer Q "
            "directions on held-out queries."
        )
    )
    parser.add_argument("--step_profile", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--selected_heads", required=True)
    parser.add_argument("--train_splits", default="train,dev")
    parser.add_argument("--test_splits", default="test")
    parser.add_argument("--prototypes", type=int, default=128)
    parser.add_argument("--fit_sample", type=int, default=4096)
    parser.add_argument("--kmeans_iterations", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def parse_set(spec: str) -> set[str]:
    return {item.strip() for item in spec.split(",") if item.strip()}


def parse_heads(spec: str) -> list[tuple[int, int]]:
    return [
        tuple(int(value) for value in item.strip().split(":"))
        for item in spec.split(",")
        if item.strip()
    ]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def all_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    if not phrase:
        return []
    output = []
    start = 0
    while True:
        index = text.find(phrase, start)
        if index < 0:
            return output
        output.append((index, index + len(phrase)))
        start = index + 1


def overlaps(span: tuple[int, int], targets: Sequence[tuple[int, int]]) -> bool:
    return any(span[1] > target[0] and span[0] < target[1] for target in targets)


def pointer_token_indices(
    *,
    tokenizer: AutoTokenizer,
    step: dict,
    token_positions: Sequence[int],
) -> list[int]:
    compact = [str(item) for item in step["compact_state_before"]]
    state = " ".join(
        [
            str(step.get("lookup_key", "")),
            str(step["step_question"]),
            str(step["question"]),
            *compact,
        ]
    )
    prompt = f"\nCurrent reasoning state: {state}\nRetrieve evidence for the next step:"
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if int(step["step_index"]) == 0:
        phrases = [str(step.get("lookup_key", ""))]
    else:
        phrases = []
        for item in compact:
            phrases.extend([item, item.split(":", maxsplit=1)[-1].strip()])
    spans = [span for phrase in phrases for span in all_spans(prompt, phrase)]
    selected = []
    for token_index, position in enumerate(token_positions):
        token_span = tuple(int(item) for item in encoded["offset_mapping"][position])
        if overlaps(token_span, spans):
            selected.append(token_index)
    return selected or list(range(len(token_positions)))


def spectral_metrics(matrix: torch.Tensor) -> dict[str, float | int]:
    centered = matrix.float() - matrix.float().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered
    values = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0.0)
    probability = values / values.sum().clamp_min(1.0e-30)
    cumulative = torch.cumsum(probability, dim=0)
    entropy = -(probability * torch.log(probability.clamp_min(1.0e-30))).sum()
    return {
        "rank90": int(torch.searchsorted(cumulative, 0.90).item() + 1),
        "rank95": int(torch.searchsorted(cumulative, 0.95).item() + 1),
        "effective_rank": float(torch.exp(entropy).item()),
        "rank8_energy": float(cumulative[min(7, len(cumulative) - 1)].item()),
        "rank16_energy": float(cumulative[min(15, len(cumulative) - 1)].item()),
    }


def spherical_kmeans(
    matrix: torch.Tensor,
    *,
    clusters: int,
    fit_sample: int,
    iterations: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(matrix) > fit_sample:
        indices = torch.randperm(len(matrix), generator=generator)[:fit_sample]
        fit = matrix.index_select(0, indices)
    else:
        fit = matrix
    fit = F.normalize(fit.float(), dim=-1).to(device)
    if len(fit) < clusters:
        clusters = len(fit)
    initial = torch.randperm(len(fit), generator=generator)[:clusters]
    centers = fit.index_select(0, initial.to(device)).clone()
    for _ in range(iterations):
        assignment = torch.argmax(fit @ centers.T, dim=1)
        updated = torch.zeros_like(centers)
        updated.index_add_(0, assignment, fit)
        counts = torch.bincount(assignment, minlength=clusters)
        empty = counts == 0
        if bool(empty.any()):
            replacements = torch.randperm(len(fit), generator=generator)[: int(empty.sum())]
            updated[empty] = fit.index_select(0, replacements.to(device))
        centers = F.normalize(updated, dim=-1)
    return centers.cpu()


def nearest_cosines(
    matrix: torch.Tensor,
    centers: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    output = []
    for start in range(0, len(matrix), batch_size):
        query = matrix[start : start + batch_size].float()
        output.append((query @ centers.T).amax(dim=1))
    return torch.cat(output)


def main() -> None:
    args = parse_args()
    payload = torch.load(args.step_profile, map_location="cpu", weights_only=False)
    train_splits = parse_set(args.train_splits)
    test_splits = parse_set(args.test_splits)
    selected_heads = parse_heads(args.selected_heads)
    layers = [int(item) for item in payload["layers"]]
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    device = torch.device(args.device)

    token_indices = [
        pointer_token_indices(
            tokenizer=tokenizer,
            step=step,
            token_positions=payload["token_positions"][step_index],
        )
        for step_index, step in enumerate(payload["steps"])
    ]
    rows = []
    prototype_centers = []
    for head_index, (layer, head) in enumerate(selected_heads):
        if layer not in layer_to_index:
            raise ValueError(f"profile has no selected layer {layer}")
        train_vectors = []
        test_vectors = []
        for step_index, step in enumerate(payload["steps"]):
            vectors = payload["svd_q"][
                step_index,
                token_indices[step_index],
                layer_to_index[layer],
                head,
            ].float()
            split = str(step["split"])
            if split in train_splits:
                train_vectors.append(vectors)
            if split in test_splits:
                test_vectors.append(vectors)
        train = F.normalize(torch.cat(train_vectors), dim=-1)
        test = F.normalize(torch.cat(test_vectors), dim=-1)
        centers = spherical_kmeans(
            train,
            clusters=args.prototypes,
            fit_sample=args.fit_sample,
            iterations=args.kmeans_iterations,
            seed=args.seed + head_index,
            device=device,
        )
        prototype_centers.append(centers.to(dtype=torch.float16))
        nearest = nearest_cosines(test, centers, batch_size=args.batch_size)
        row = {
            "layer": layer,
            "query_head": head,
            "train_pointer_vectors": len(train),
            "test_pointer_vectors": len(test),
            "prototypes": len(centers),
            "spectrum": spectral_metrics(train),
            "test_nearest_prototype_cosine_mean": float(nearest.mean().item()),
            "test_nearest_prototype_cosine_p05": float(
                torch.quantile(nearest, 0.05).item()
            ),
            "test_coverage_cos_ge_0p8": float((nearest >= 0.8).float().mean().item()),
            "test_coverage_cos_ge_0p9": float((nearest >= 0.9).float().mean().item()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    summary = {
        "source": "train-only state-pointer Q manifold and held-out prototype coverage",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "step_profile": args.step_profile,
        "train_splits": sorted(train_splits),
        "test_splits": sorted(test_splits),
        "heads": len(rows),
        "prototypes_per_head": args.prototypes,
        "macro": {
            "rank90_mean": mean(row["spectrum"]["rank90"] for row in rows),
            "effective_rank_mean": mean(
                row["spectrum"]["effective_rank"] for row in rows
            ),
            "test_nearest_prototype_cosine_mean": mean(
                row["test_nearest_prototype_cosine_mean"] for row in rows
            ),
            "test_nearest_prototype_cosine_p05_mean": mean(
                row["test_nearest_prototype_cosine_p05"] for row in rows
            ),
            "test_coverage_cos_ge_0p8_mean": mean(
                row["test_coverage_cos_ge_0p8"] for row in rows
            ),
            "test_coverage_cos_ge_0p9_mean": mean(
                row["test_coverage_cos_ge_0p9"] for row in rows
            ),
        },
        "rows": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prototype_path = output_dir / "prototypes.pt"
    torch.save(
        {
            "centers": torch.stack(prototype_centers),
            "selected_heads": selected_heads,
            "train_splits": sorted(train_splits),
            "test_splits": sorted(test_splits),
            "normalized": True,
            "source": "train-only spherical state-pointer Q prototypes",
        },
        prototype_path,
    )
    summary["prototype_path"] = str(prototype_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
