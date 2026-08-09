from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_hierarchical_spectral_quantization_20260727 import (
    covariance_basis,
    quantize_groupwise,
    query_int8,
)


FULL_KV_BITS = 2 * 128 * 16
HIERARCHICAL_CODE_BITS = 16 * 8 + 32 * 4 + 80 * 2
HIERARCHICAL_METADATA_BITS = 8 * 16


def parse_floats(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one floating-point value")
    return result


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def systematic_sample_indices(
    history_count: int,
    sample_count: int,
    offset_seed: int,
    device: torch.device,
) -> torch.Tensor:
    sample_count = min(history_count, max(1, sample_count))
    stride = max(1, history_count // sample_count)
    offset = offset_seed % stride
    indices = offset + torch.arange(sample_count, device=device) * stride
    return indices.clamp_max(history_count - 1).unique()


def mean_and_standard_error(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=values.device)
        return zero, torch.full_like(zero, torch.inf)
    mean = values.float().mean()
    if values.numel() == 1:
        return mean, torch.full_like(mean, torch.inf)
    standard_error = values.float().var(unbiased=True).div(values.numel()).sqrt()
    return mean, standard_error


def estimate_mass_ladders(
    proxy_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    keep_counts: list[int],
    sample_indices: torch.Tensor,
    z_values: list[float],
) -> dict[str, torch.Tensor]:
    center = torch.maximum(
        proxy_scores.max(), exact_scores.index_select(0, sample_indices).max()
    )
    proxy_weights = torch.exp(proxy_scores.float() - center)
    exact_weights = torch.exp(exact_scores.float() - center)
    proxy_prefix = proxy_weights.index_select(0, candidate_indices).cumsum(0)
    proxy_total = proxy_weights.sum()
    sample_difference = (
        exact_weights.index_select(0, sample_indices)
        - proxy_weights.index_select(0, sample_indices)
    )
    global_mean, global_se = mean_and_standard_error(sample_difference)

    output: dict[str, list[torch.Tensor]] = defaultdict(list)
    for keep_count in keep_counts:
        selected = candidate_indices[:keep_count]
        selected_proxy = proxy_prefix[keep_count - 1]
        proxy_tail = (proxy_total - selected_proxy).clamp_min(0.0)
        proxy_mass = selected_proxy / proxy_total.clamp_min(1.0e-30)
        output["proxy"].append(proxy_mass)

        selected_mask = torch.zeros(
            proxy_scores.numel(), dtype=torch.bool, device=proxy_scores.device
        )
        selected_mask[selected] = True
        sample_inside = selected_mask.index_select(0, sample_indices)
        inside_difference = sample_difference[sample_inside]
        outside_difference = sample_difference[~sample_inside]
        inside_mean, inside_se = mean_and_standard_error(inside_difference)
        outside_mean, outside_se = mean_and_standard_error(outside_difference)

        for z_value in z_values:
            global_tail = (
                proxy_tail
                + (proxy_scores.numel() - keep_count)
                * (global_mean + z_value * global_se)
            ).clamp_min(0.0)
            global_mass = selected_proxy / (
                selected_proxy + global_tail
            ).clamp_min(1.0e-30)
            output[f"global_z{z_value:g}"].append(global_mass)

            if torch.isfinite(inside_se) and torch.isfinite(outside_se):
                selected_lower = (
                    selected_proxy
                    + keep_count * (inside_mean - z_value * inside_se)
                ).clamp_min(0.0)
                tail_upper = (
                    proxy_tail
                    + (proxy_scores.numel() - keep_count)
                    * (outside_mean + z_value * outside_se)
                ).clamp_min(0.0)
                stratified_mass = selected_lower / (
                    selected_lower + tail_upper
                ).clamp_min(1.0e-30)
            else:
                stratified_mass = torch.zeros_like(selected_proxy)
            output[f"stratified_z{z_value:g}"].append(stratified_mass)

    return {name: torch.stack(values) for name, values in output.items()}


def choose_rung(values: torch.Tensor, target: float) -> int:
    reached = values >= target
    if bool(reached.any()):
        return int(reached.to(torch.int64).argmax().item())
    return int(values.numel() - 1)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    output: list[dict[str, Any]] = []
    for policy, items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "policy": policy,
            "cases": len(items),
            "basis_tokens": int(items[0]["basis_tokens"]),
            "sample_count": int(items[0]["sample_count"]),
            "index_code_bits": HIERARCHICAL_CODE_BITS,
            "index_metadata_bits": HIERARCHICAL_METADATA_BITS,
            "index_total_bits": (
                HIERARCHICAL_CODE_BITS + HIERARCHICAL_METADATA_BITS
            ),
            "index_ratio_of_full_kv": (
                HIERARCHICAL_CODE_BITS + HIERARCHICAL_METADATA_BITS
            )
            / FULL_KV_BITS,
            "exact_sample_ratio_mean": sum(
                float(item["exact_sample_ratio"]) for item in items
            )
            / len(items),
        }
        for field in (
            "selected_fraction",
            "actual_attention_mass",
            "top2_recall",
            "estimated_attention_mass",
            "mass_overestimate",
        ):
            stats = summarize(float(item[field]) for item in items)
            result.update(
                {f"{field}_{name}": value for name, value in stats.items()}
            )
        result["actual_mass_below_0p80_rate"] = sum(
            float(item["actual_attention_mass"]) < 0.80 for item in items
        ) / len(items)
        result["actual_mass_below_0p90_rate"] = sum(
            float(item["actual_attention_mass"]) < 0.90 for item in items
        ) / len(items)
        result["overestimate_above_0p05_rate"] = sum(
            float(item["mass_overestimate"]) > 0.05 for item in items
        ) / len(items)
        rung_counts = Counter(float(item["selected_fraction"]) for item in items)
        result["rung_distribution"] = json.dumps(
            {
                f"{fraction:.6g}": count / len(items)
                for fraction, count in sorted(rung_counts.items())
            },
            sort_keys=True,
        )
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate training-free numerical budget selection on the "
            "hierarchical 8/4/2 spectral index."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--basis_tokens", type=int, default=0)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument(
        "--fractions", default="0.005,0.01,0.02,0.03,0.04,0.06,0.08"
    )
    parser.add_argument("--mass_targets", default="0.85,0.90,0.92,0.94,0.95")
    parser.add_argument("--z_values", default="0,1")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--max_records", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    fractions = parse_floats(args.fractions)
    targets = parse_floats(args.mass_targets)
    z_values = parse_floats(args.z_values)
    if fractions[0] <= 0.0 or fractions[-1] > 1.0:
        raise ValueError("fractions must be in (0, 1]")
    if targets[0] <= 0.0 or targets[-1] > 1.0:
        raise ValueError("mass targets must be in (0, 1]")
    if args.sample_count <= 0 or args.sample_stride <= 0:
        raise ValueError("sample count and sample stride must be positive")

    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    states: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        layer = int(record["layer"])
        raw_key = record.get("key")
        if layer not in states:
            if raw_key is None:
                raise RuntimeError(f"layer {layer} is missing initial key state")
            all_key = raw_key.to(device).float()[0]
            history_count = int(all_key.shape[1]) - 1
            key = all_key[:, :history_count]
            basis_limit = (
                history_count
                if args.basis_tokens <= 0
                else min(history_count, args.basis_tokens)
            )
            prepared = []
            for kv_head in range(key.shape[0]):
                head_key = key[kv_head]
                basis, _ = covariance_basis(
                    head_key[:basis_limit: args.sample_stride]
                )
                coefficients = head_key @ basis
                reconstructed = torch.cat(
                    (
                        quantize_groupwise(coefficients[:, :16], 8),
                        quantize_groupwise(coefficients[:, 16:48], 4),
                        quantize_groupwise(coefficients[:, 48:], 2),
                    ),
                    dim=-1,
                )
                prepared.append(
                    {
                        "head_key": head_key,
                        "basis": basis,
                        "reconstructed": reconstructed,
                    }
                )
            states[layer] = {
                "history_count": history_count,
                "kv_heads": int(key.shape[0]),
                "prepared": prepared,
            }

        state = states[layer]
        history_count = int(state["history_count"])
        keep_counts = [
            min(history_count, max(1, math.ceil(fraction * history_count)))
            for fraction in fractions
        ]
        max_keep = keep_counts[-1]
        query = record["query"].to(device).float()[0, :, 0, :]
        kv_heads = int(state["kv_heads"])
        groups = int(query.shape[0]) // kv_heads
        scaling = float(record["scaling"])
        top_count = max(1, math.ceil(args.top_fraction * history_count))

        for kv_head in range(kv_heads):
            prepared = state["prepared"][kv_head]
            for group in range(groups):
                query_head = kv_head * groups + group
                head_query = query[query_head]
                projected_query = query_int8(head_query @ prepared["basis"])
                exact_scores = prepared["head_key"] @ head_query * scaling
                proxy_scores = (
                    prepared["reconstructed"] @ projected_query
                ) * scaling
                exact_attention = torch.softmax(exact_scores, dim=-1)
                true_top = torch.topk(exact_scores, k=top_count).indices
                candidates = torch.topk(
                    proxy_scores, k=max_keep, sorted=True
                ).indices
                selected_mask = torch.zeros(
                    history_count, dtype=torch.bool, device=device
                )
                actual_masses = []
                recalls = []
                for keep_count in keep_counts:
                    selected = candidates[:keep_count]
                    selected_mask.zero_()
                    selected_mask[selected] = True
                    actual_masses.append(exact_attention[selected].sum())
                    recalls.append(selected_mask[true_top].float().mean())
                actual_mass_tensor = torch.stack(actual_masses)
                recall_tensor = torch.stack(recalls)
                sample_indices = systematic_sample_indices(
                    history_count,
                    args.sample_count,
                    37 * record_index + 17 * query_head + layer,
                    device,
                )
                estimated = estimate_mass_ladders(
                    proxy_scores,
                    exact_scores,
                    candidates,
                    keep_counts,
                    sample_indices,
                    z_values,
                )
                estimated["oracle"] = actual_mass_tensor

                for rung, fraction in enumerate(fractions):
                    rows.append(
                        {
                            "label": args.label,
                            "record": record_index,
                            "layer": layer,
                            "kv_head": kv_head,
                            "query_head": query_head,
                            "policy": f"fixed_f{fraction:g}",
                            "basis_tokens": args.basis_tokens,
                            "sample_count": sample_indices.numel(),
                            "exact_sample_ratio": (
                                sample_indices.numel() / history_count
                            ),
                            "selected_fraction": (
                                keep_counts[rung] / history_count
                            ),
                            "actual_attention_mass": float(
                                actual_mass_tensor[rung].item()
                            ),
                            "top2_recall": float(recall_tensor[rung].item()),
                            "estimated_attention_mass": float(
                                estimated["proxy"][rung].item()
                            ),
                            "mass_overestimate": float(
                                (
                                    estimated["proxy"][rung]
                                    - actual_mass_tensor[rung]
                                ).item()
                            ),
                        }
                    )

                for estimator, ladder in estimated.items():
                    for target in targets:
                        rung = choose_rung(ladder, target)
                        policy = f"{estimator}_target{target:g}"
                        rows.append(
                            {
                                "label": args.label,
                                "record": record_index,
                                "layer": layer,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "policy": policy,
                                "basis_tokens": args.basis_tokens,
                                "sample_count": sample_indices.numel(),
                                "exact_sample_ratio": (
                                    sample_indices.numel() / history_count
                                ),
                                "selected_fraction": (
                                    keep_counts[rung] / history_count
                                ),
                                "actual_attention_mass": float(
                                    actual_mass_tensor[rung].item()
                                ),
                                "top2_recall": float(recall_tensor[rung].item()),
                                "estimated_attention_mass": float(
                                    ladder[rung].item()
                                ),
                                "mass_overestimate": float(
                                    (
                                        ladder[rung]
                                        - actual_mass_tensor[rung]
                                    ).item()
                                ),
                            }
                        )

        print(
            json.dumps(
                {
                    "label": args.label,
                    "record": record_index + 1,
                    "records": len(records),
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    output = {
        "config": {
            "trace_path": str(args.trace_path),
            "label": args.label,
            "basis_tokens": args.basis_tokens,
            "sample_count": args.sample_count,
            "fractions": fractions,
            "mass_targets": targets,
            "z_values": z_values,
        },
        "policies": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
