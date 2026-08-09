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
    quantize_per_token_int4,
    quantize_per_token_int8,
    summarize,
)
from analyze_rope_free_candidate_rescue import (
    inverse_rope,
    quota_union,
    record_candidates,
)


def fixed_clip_int4(value: torch.Tensor, clip: torch.Tensor | float) -> torch.Tensor:
    scale = torch.as_tensor(clip, dtype=torch.float32, device=value.device) / 7.0
    return torch.round(value.float() / scale).clamp(-7, 7) * scale


def fixed_clip_int2(value: torch.Tensor, clip: torch.Tensor | float) -> torch.Tensor:
    scale = torch.as_tensor(clip, dtype=torch.float32, device=value.device)
    normalized = (value.float() / scale).clamp(-1.0, 1.0)
    codes = torch.round((normalized + 1.0) * 1.5).clamp(0, 3)
    return (codes / 1.5 - 1.0) * scale


def fixed_clip_ternary(value: torch.Tensor, clip: torch.Tensor | float) -> torch.Tensor:
    scale = torch.as_tensor(clip, dtype=torch.float32, device=value.device)
    return torch.round(value.float() / scale).clamp(-1, 1) * scale


@torch.inference_mode()
def evaluate_trace(
    path: Path,
    candidate_fractions: tuple[float, ...],
    rescue_fractions: tuple[float, ...],
    theta: float,
    device: torch.device,
) -> dict[str, object]:
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

        positions = torch.arange(history_count, device=device)
        query_positions = torch.arange(
            history_count, history_count + len(records), device=device
        )
        pre_key = inverse_rope(post_key, positions, theta)
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
        query_int8 = quantize_per_token_int8(grouped_low_query)

        rescue_scores: dict[str, torch.Tensor] = {
            "int4_token_scale": grouped_scores(
                quantize_per_token_int4(low_key), query_int8, group_size
            )[1:]
        }
        dimensions = low_key.shape[-1]
        for alpha in (1.5, 2.0, 2.5, 3.0):
            clip = alpha / math.sqrt(dimensions)
            label = str(alpha).replace(".", "p")
            rescue_scores[f"int4_fixed_a{label}"] = grouped_scores(
                fixed_clip_int4(low_key, clip), query_int8, group_size
            )[1:]
            rescue_scores[f"int2_fixed_a{label}"] = grouped_scores(
                fixed_clip_int2(low_key, clip), query_int8, group_size
            )[1:]
            rescue_scores[f"ternary_fixed_a{label}"] = grouped_scores(
                fixed_clip_ternary(low_key, clip), query_int8, group_size
            )[1:]
        for percentile in (0.99, 0.995, 0.999):
            clip = torch.quantile(
                low_key.abs().flatten(1), percentile, dim=-1
            ).reshape(kv_heads, 1, 1)
            label = str(percentile).replace(".", "p")
            rescue_scores[f"int4_head_q{label}"] = grouped_scores(
                fixed_clip_int4(low_key, clip), query_int8, group_size
            )[1:]
            rescue_scores[f"int2_head_q{label}"] = grouped_scores(
                fixed_clip_int2(low_key, clip), query_int8, group_size
            )[1:]

        for candidate_fraction in candidate_fractions:
            candidate_count = max(
                keep_count, math.ceil(candidate_fraction * history_count)
            )
            base_candidates = torch.topk(
                base_scores, candidate_count, dim=-1, sorted=False
            ).indices
            prefix = f"candidate{100*candidate_fraction:g}"
            record_candidates(
                metrics,
                f"{prefix}_base",
                base_candidates,
                exact_scores,
                exact_top,
                value,
                keep_count,
                group_size,
                scaling,
            )
            for rescue_fraction in rescue_fractions:
                rescue_count = min(
                    candidate_count - 1,
                    max(1, math.ceil(rescue_fraction * history_count)),
                )
                for rescue_name, scores in rescue_scores.items():
                    candidates = quota_union(
                        base_scores, scores, candidate_count, rescue_count
                    )
                    record_candidates(
                        metrics,
                        f"{prefix}_rescue{100*rescue_fraction:g}_{rescue_name}",
                        candidates,
                        exact_scores,
                        exact_top,
                        value,
                        keep_count,
                        group_size,
                        scaling,
                    )
        del post_key, value, post_query, exact_scores, exact_top, pre_key, pre_query
        torch.cuda.empty_cache()

    return {
        "trace": str(path),
        "methods": {
            method: {name: summarize(values) for name, values in values_by_name.items()}
            for method, values_by_name in sorted(metrics.items())
        },
    }


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(sorted({float(item) for item in value.split(",")}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate_fractions", default="0.03,0.04,0.05,0.06,0.08")
    parser.add_argument("--rescue_fractions", default="0.0025,0.005")
    parser.add_argument("--rope_theta", type=float, default=5_000_000.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "method": "scale-free low-frequency rescue quantization sweep",
        "candidate_fractions": parse_floats(args.candidate_fractions),
        "rescue_fractions": parse_floats(args.rescue_fractions),
        "traces": [
            evaluate_trace(
                path,
                parse_floats(args.candidate_fractions),
                parse_floats(args.rescue_fractions),
                args.rope_theta,
                torch.device(args.device),
            )
            for path in args.trace_paths
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
