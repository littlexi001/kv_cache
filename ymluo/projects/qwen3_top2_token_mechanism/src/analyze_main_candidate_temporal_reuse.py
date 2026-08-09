from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from analyze_balanced_pca_int4 import (
    grouped_scores,
    quantize_per_band_logscale_int4,
    quantize_per_token_int8,
    summarize,
)
from analyze_lowfreq_temporal_reuse import cached_indices
from analyze_rope_free_candidate_rescue import record_candidates


@torch.inference_mode()
def evaluate_trace(path: Path, device: torch.device) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)

    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    stability: dict[str, list[float]] = defaultdict(list)

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

        exact_scores = grouped_scores(key, query, group_size)[1:]
        exact_top = torch.topk(
            exact_scores, keep_count, dim=-1, sorted=False
        ).indices
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
        current = torch.topk(
            proxy_scores, candidate_count, dim=-1, sorted=False
        ).indices

        for interval in (1, 2, 4, 8):
            reused = cached_indices(current, interval)
            overlap = (
                current.unsqueeze(-1) == reused.unsqueeze(-2)
            ).any(dim=-1).float().mean(dim=-1)
            stability[f"refresh{interval}"].extend(overlap.flatten().cpu().tolist())
            record_candidates(
                metrics,
                f"main_refresh{interval}",
                reused,
                exact_scores,
                exact_top,
                value,
                keep_count,
                group_size,
                scaling,
            )
        del key, value, query, exact_scores, exact_top
        torch.cuda.empty_cache()

    return {
        "trace": str(path),
        "methods": {
            method: {name: summarize(values) for name, values in values_by_name.items()}
            for method, values_by_name in sorted(metrics.items())
        },
        "candidate_stability": {
            name: summarize(values) for name, values in sorted(stability.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "method": "training-free temporal reuse of main PCA64 candidates",
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
