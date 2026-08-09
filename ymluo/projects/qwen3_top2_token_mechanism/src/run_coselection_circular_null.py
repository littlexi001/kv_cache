from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test Top-2% pair synchrony against independent circular shifts of each token's "
            "selection time series. The null preserves token marginals and temporal runs."
        )
    )
    parser.add_argument("--selection_indices", required=True)
    parser.add_argument("--head_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--representative_heads", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def read_head_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_heads(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["cluster_score"]))
    requested = min(max(0, count), len(ordered))
    low_count = requested // 2
    high_count = requested - low_count
    return ordered[:low_count] + list(reversed(ordered[-high_count:]))


def incidence_from_indices(indices: np.ndarray, token_count: int) -> np.ndarray:
    observations, budget = indices.shape
    incidence = np.zeros((observations, token_count), dtype=np.uint8)
    row = np.repeat(np.arange(observations), budget)
    incidence[row, indices.reshape(-1)] = 1
    return incidence


def excess_fraction_batch(
    incidence: np.ndarray,
    selection_count: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    values = torch.from_numpy(incidence).to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    cooccurrence = torch.bmm(values.transpose(1, 2), values)
    counts = torch.from_numpy(selection_count).to(device=device, dtype=torch.float32)
    expected = counts[:, None] * counts[None, :] / values.shape[1]
    upper_mask = torch.triu(
        torch.ones((values.shape[2], values.shape[2]), dtype=torch.bool, device=device),
        diagonal=1,
    )
    observed = cooccurrence.float().masked_fill(~upper_mask, 0.0)
    positive_excess = torch.relu(observed - expected).masked_fill(~upper_mask, 0.0)
    denominator = observed.sum(dim=(1, 2))
    result = torch.where(
        denominator > 0,
        positive_excess.sum(dim=(1, 2)) / denominator,
        torch.zeros_like(denominator),
    ).cpu().numpy()
    del values, cooccurrence, counts, expected, upper_mask, observed, positive_excess, denominator
    return result


def circular_shift_batch(
    incidence: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    observations, token_count = incidence.shape
    output = np.empty((batch_size, observations, token_count), dtype=np.uint8)
    token_columns = np.arange(token_count)[None, :]
    observation_rows = np.arange(observations)[:, None]
    for index in range(batch_size):
        offsets = rng.integers(0, observations, size=token_count)
        source_rows = (observation_rows - offsets[None, :]) % observations
        output[index] = incidence[source_rows, token_columns]
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.permutations <= 0 or args.batch_size <= 0:
        raise ValueError("permutations and batch_size must be positive.")
    archive = np.load(args.selection_indices)
    indices = archive["indices"]
    layers = archive["selected_layers"].astype(int).tolist()
    heads = archive["selected_heads"].astype(int).tolist()
    token_count = int(archive["context_token_ids"].size)
    observations = int(indices.shape[2])
    head_rows = read_head_rows(Path(args.head_summary))
    representatives = choose_heads(head_rows, args.representative_heads)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    result_rows: list[dict[str, Any]] = []
    for representative_index, head_row in enumerate(representatives, start=1):
        layer = int(head_row["layer"])
        head = int(head_row["head"])
        print(
            f"null head {representative_index}/{len(representatives)}: L{layer} H{head}",
            flush=True,
        )
        layer_slot = layers.index(layer)
        head_slot = heads.index(head)
        incidence = incidence_from_indices(indices[layer_slot, head_slot], token_count)
        selection_count = incidence.sum(axis=0, dtype=np.int64)
        observed_statistic = float(excess_fraction_batch(incidence, selection_count, device)[0])

        null_statistics: list[float] = []
        remaining = args.permutations
        while remaining > 0:
            current_batch = min(args.batch_size, remaining)
            shifted = circular_shift_batch(incidence, current_batch, rng)
            null_statistics.extend(
                excess_fraction_batch(shifted, selection_count, device).astype(float).tolist()
            )
            remaining -= current_batch
        null_values = np.asarray(null_statistics, dtype=np.float64)
        null_mean = float(null_values.mean())
        null_std = float(null_values.std(ddof=1)) if null_values.size > 1 else 0.0
        empirical_p = float((1 + np.count_nonzero(null_values >= observed_statistic)) / (null_values.size + 1))
        z_score = float((observed_statistic - null_mean) / null_std) if null_std > 0 else 0.0
        result_rows.append(
            {
                "rank_group": "low" if representative_index <= len(representatives) // 2 else "high",
                "layer": layer,
                "head": head,
                "observations": observations,
                "token_count": token_count,
                "budget": int(indices.shape[-1]),
                "cluster_score": float(head_row["cluster_score"]),
                "significant_positive_pairs": int(float(head_row["significant_positive_pairs"])),
                "observed_excess_fraction": observed_statistic,
                "circular_null_mean": null_mean,
                "circular_null_std": null_std,
                "observed_minus_null": observed_statistic - null_mean,
                "null_z_score": z_score,
                "empirical_p_value": empirical_p,
                "permutations": args.permutations,
            }
        )

    low = [row for row in result_rows if row["rank_group"] == "low"]
    high = [row for row in result_rows if row["rank_group"] == "high"]
    summary = {
        "heads_tested": len(result_rows),
        "permutations_per_head": args.permutations,
        "minimum_empirical_p": 1.0 / (args.permutations + 1),
        "significant_at_0p05": sum(float(row["empirical_p_value"]) <= 0.05 for row in result_rows),
        "low_group": {
            "heads": len(low),
            "median_observed_minus_null": float(np.median([row["observed_minus_null"] for row in low])) if low else 0.0,
            "median_z": float(np.median([row["null_z_score"] for row in low])) if low else 0.0,
        },
        "high_group": {
            "heads": len(high),
            "median_observed_minus_null": float(np.median([row["observed_minus_null"] for row in high])) if high else 0.0,
            "median_z": float(np.median([row["null_z_score"] for row in high])) if high else 0.0,
        },
        "null_definition": (
            "Each token selection column is circularly shifted independently across queries. "
            "This preserves every token's marginal selection count and temporal run structure, "
            "while destroying same-query phase alignment between token pairs."
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "circular_shift_null_by_head.csv", result_rows)
    (output_dir / "circular_shift_null_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
