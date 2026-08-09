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
from analyze_lowfreq_quantization_sweep import fixed_clip_int2
from analyze_lowfreq_temporal_reuse import merge_cached_rescue
from analyze_rope_free_candidate_rescue import (
    inverse_rope,
    record_candidates,
)


def mask_recent_fraction(scores: torch.Tensor, oldest_fraction: float) -> torch.Tensor:
    history_count = scores.shape[-1]
    scan_count = max(1, math.ceil(oldest_fraction * history_count))
    if scan_count >= history_count:
        return scores
    masked = scores.clone()
    masked[..., scan_count:] = -torch.inf
    return masked


def binary_sign(value: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(value.shape[-1])
    return torch.where(value >= 0, scale, -scale)


@torch.inference_mode()
def evaluate_trace(path: Path, theta: float, device: torch.device) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)

    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rescued_distance: dict[str, list[float]] = defaultdict(list)
    rescued_counts: dict[str, float] = defaultdict(float)

    for _, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        key_record = next((row for row in records if row.get("key") is not None), None)
        if key_record is None or len(records) < 2:
            continue

        post_key = key_record["key"].to(device).float()[0]
        value = key_record["value"].to(device).float()[0]
        history_count = int(post_key.shape[1]) - 1
        post_key = post_key[:, :history_count]
        value = value[:, :history_count]
        post_query = torch.stack(
            [row["query"].to(device).float()[0, :, 0] for row in records]
        )
        kv_heads = int(post_key.shape[0])
        query_heads = int(post_query.shape[1])
        group_size = query_heads // kv_heads
        keep_count = max(1, math.ceil(0.02 * history_count))
        candidate_count = max(keep_count, math.ceil(0.08 * history_count))
        rescue_count = max(1, math.ceil(0.005 * history_count))
        scaling = 1.0 / math.sqrt(post_key.shape[-1])

        exact_scores = grouped_scores(post_key, post_query, group_size)[1:]
        exact_top = torch.topk(exact_scores, keep_count, dim=-1, sorted=False).indices

        sampled_key = post_key[:, ::32]
        second_moment = torch.einsum("hnd,hne->hde", sampled_key, sampled_key)
        second_moment /= float(sampled_key.shape[1])
        _, basis = torch.linalg.eigh(second_moment)
        basis = basis[..., -64:]
        projected_key = torch.einsum("hnd,hdm->hnm", post_key, basis)
        grouped_query = post_query.reshape(
            len(records), kv_heads, group_size, post_query.shape[-1]
        )
        projected_query = torch.einsum("thgd,hdm->thgm", grouped_query, basis)
        base_scores = grouped_scores(
            quantize_per_band_logscale_int4(projected_key, 16, 0.25),
            quantize_per_token_int8(projected_query),
            group_size,
        )[1:]
        base_candidates = torch.topk(
            base_scores, candidate_count, dim=-1, sorted=False
        ).indices
        record_candidates(
            metrics,
            "base_candidate8",
            base_candidates,
            exact_scores,
            exact_top,
            value,
            keep_count,
            group_size,
            scaling,
        )

        key_positions = torch.arange(history_count, device=device)
        query_positions = torch.arange(
            history_count, history_count + len(records), device=device
        )
        pre_key = inverse_rope(post_key, key_positions, theta)
        pre_query = inverse_rope(
            post_query.transpose(0, 1), query_positions, theta
        ).transpose(0, 1)
        half = post_key.shape[-1] // 2
        low_indices = torch.cat(
            (
                torch.arange(half - 16, half, device=device),
                torch.arange(2 * half - 16, 2 * half, device=device),
            )
        )
        low_key = F.normalize(pre_key.index_select(-1, low_indices), dim=-1)
        low_query = F.normalize(pre_query.index_select(-1, low_indices), dim=-1)
        grouped_low_query = low_query.reshape(
            len(records), kv_heads, group_size, low_key.shape[-1]
        )
        low_scores_by_precision = {
            "int2": grouped_scores(
                fixed_clip_int2(low_key, 1.5 / math.sqrt(32)),
                quantize_per_token_int8(grouped_low_query),
                group_size,
            )[1:],
            "binary": grouped_scores(
                binary_sign(low_key), binary_sign(grouped_low_query), group_size
            )[1:],
        }

        base_recall_mask = (
            base_candidates.unsqueeze(-1) == exact_top.unsqueeze(-2)
        ).any(dim=-2)
        for precision, low_scores in low_scores_by_precision.items():
            for oldest_fraction in (0.25, 0.5, 0.75, 1.0):
                label = f"{precision}_oldest{int(100 * oldest_fraction)}"
                gated_scores = mask_recent_fraction(low_scores, oldest_fraction)
                rescue = torch.topk(
                    gated_scores, rescue_count, dim=-1, sorted=False
                ).indices
                candidates = merge_cached_rescue(
                    base_scores, rescue, base_count=candidate_count
                )
                record_candidates(
                    metrics,
                    f"{label}_union8p5",
                    candidates,
                    exact_scores,
                    exact_top,
                    value,
                    keep_count,
                    group_size,
                    scaling,
                )

                rescue_mask = (
                    rescue.unsqueeze(-1) == exact_top.unsqueeze(-2)
                ).any(dim=-2)
                newly_rescued = (~base_recall_mask) & rescue_mask
                distances = 1.0 - exact_top.float() / float(history_count)
                rescued_distance[label].extend(
                    distances[newly_rescued].cpu().tolist()
                )
                rescued_counts[label] += float(newly_rescued.sum().item())

        del post_key, value, post_query, exact_scores, exact_top, pre_key, pre_query
        torch.cuda.empty_cache()

    return {
        "trace": str(path),
        "methods": {
            method: {name: summarize(values) for name, values in values_by_name.items()}
            for method, values_by_name in sorted(metrics.items())
        },
        "rescued_exact_top2_distance": {
            name: summarize(values) for name, values in sorted(rescued_distance.items())
        },
        "rescued_exact_top2_count": dict(sorted(rescued_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--rope_theta", type=float, default=5_000_000.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "method": "distance-gated scale-free low-frequency INT2 rescue",
        "traces": [
            evaluate_trace(path, args.rope_theta, torch.device(args.device))
            for path in args.trace_paths
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
