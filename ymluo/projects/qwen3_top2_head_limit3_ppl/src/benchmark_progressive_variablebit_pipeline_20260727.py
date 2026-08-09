from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import progressive_variablebit_cuda_20260727 as progressive
import variablebit_spectral_cuda_20260727 as variablebit


FIXED_4421 = (4, 4, 2, 1, 0, 0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark two-pass progressive reads of a packed 4-4-2-1 "
            "QK-balanced index."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lengths", default="32768,65536,131072")
    parser.add_argument("--candidate_ratios", default="0.38,0.51")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def parse_values(specification: str, caster) -> tuple:
    values = tuple(
        caster(item) for item in specification.split(",") if item.strip()
    )
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def full_scores(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    index: dict[str, object],
    history_count: int,
) -> torch.Tensor:
    return variablebit.scores(
        query_codes,
        query_scales,
        index["packed_codes"],
        index["key_scales"],
        index["bit_allocations"],
        index["code_offsets"],
        index["scale_offsets"],
        index["code_bases"],
        index["scale_bases"],
        index["code_strides"],
        index["scale_strides"],
        history_count,
        score_bias=index["score_bias"],
    )


def elapsed_ms(
    function: Callable[[], object],
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) / repeats)


def target_count(history_count: int) -> int:
    return min(
        history_count,
        1280,
        max(256, math.ceil(0.06 * history_count)),
    )


@torch.inference_mode()
def benchmark_length(
    history_count: int,
    candidate_ratios: tuple[float, ...],
    warmup: int,
    repeats: int,
    generator: torch.Generator,
) -> dict[str, object]:
    device = torch.device("cuda")
    allocation = torch.tensor(
        FIXED_4421,
        dtype=torch.int8,
        device=device,
    ).view(1, 1, 8).expand(1, 8, 8).contiguous()
    index = variablebit.allocate_packed_index(
        allocation,
        history_count,
        torch.float16,
    )
    projected_keys = torch.randn(
        1,
        8,
        history_count,
        128,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    variablebit.encode_projected_keys_into(projected_keys, index, 0)
    del projected_keys
    query = torch.randn(
        1,
        8,
        4,
        128,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    query_codes, query_scales = variablebit.quantize_projected_query(query)
    base_mask = torch.zeros_like(allocation)
    base_mask[..., 0] = 1
    residual_mask = (allocation > 0).to(torch.int8) - base_mask
    target = target_count(history_count)

    reference = full_scores(
        query_codes,
        query_scales,
        index,
        history_count,
    )
    base_reference = progressive.masked_scores(
        query_codes,
        query_scales,
        index,
        base_mask,
        history_count,
    )
    check_count = min(257, history_count)
    check_indices = torch.randint(
        0,
        history_count,
        (1, 32, check_count),
        device=device,
        generator=generator,
    )
    residual_check = progressive.candidate_scores(
        query_codes,
        query_scales,
        index,
        residual_mask,
        check_indices,
    )
    reconstructed_check = (
        torch.gather(base_reference, -1, check_indices) + residual_check
    )
    reference_check = torch.gather(reference, -1, check_indices)
    check_mask = torch.zeros_like(reference, dtype=torch.uint8)
    check_mask.scatter_(-1, check_indices, 1)
    residual_check_coalesced = progressive.token_masked_scores(
        query_codes,
        query_scales,
        index,
        residual_mask,
        check_mask,
    )
    reconstructed_check_coalesced = torch.gather(
        base_reference + residual_check_coalesced,
        -1,
        check_indices,
    )
    score_max_abs_error = float(
        max(
            (reconstructed_check - reference_check).abs().max().item(),
            (
                reconstructed_check_coalesced - reference_check
            ).abs().max().item(),
        )
    )
    if score_max_abs_error > 5.0e-4:
        raise AssertionError(
            f"progressive score decomposition error: {score_max_abs_error}"
        )

    full_scan_ms = elapsed_ms(
        lambda: full_scores(
            query_codes,
            query_scales,
            index,
            history_count,
        ),
        warmup,
        repeats,
    )
    full_pipeline_ms = elapsed_ms(
        lambda: torch.topk(
            full_scores(
                query_codes,
                query_scales,
                index,
                history_count,
            ),
            k=target,
            dim=-1,
        ),
        warmup,
        repeats,
    )
    base_scan_ms = elapsed_ms(
        lambda: progressive.masked_scores(
            query_codes,
            query_scales,
            index,
            base_mask,
            history_count,
        ),
        warmup,
        repeats,
    )

    sample_count = min(256, history_count)
    sample_indices = (
        torch.arange(sample_count, device=device, dtype=torch.int64)
        * max(1, history_count // sample_count)
    ).remainder(history_count)
    sample_indices = sample_indices.view(1, 1, -1).expand(
        1,
        32,
        -1,
    ).contiguous()
    rows = []
    for candidate_ratio in candidate_ratios:
        if not 0.0 < candidate_ratio <= 1.0:
            raise ValueError("candidate ratios must be in (0, 1]")
        candidate_count = min(
            history_count,
            max(target, math.ceil(candidate_ratio * history_count)),
        )

        def progressive_pipeline():
            base = progressive.masked_scores(
                query_codes,
                query_scales,
                index,
                base_mask,
                history_count,
            )
            sampled_residual = progressive.candidate_scores(
                query_codes,
                query_scales,
                index,
                residual_mask,
                sample_indices,
            )
            torch.kthvalue(
                sampled_residual.abs(),
                k=max(1, math.ceil(0.95 * sample_count)),
                dim=-1,
            )
            candidates = torch.topk(
                base,
                k=candidate_count,
                dim=-1,
            ).indices
            residual = progressive.candidate_scores(
                query_codes,
                query_scales,
                index,
                residual_mask,
                candidates,
            )
            refined = torch.gather(base, -1, candidates) + residual
            return torch.topk(refined, k=target, dim=-1)

        gather_pipeline_ms = elapsed_ms(
            progressive_pipeline,
            warmup,
            repeats,
        )
        def coalesced_progressive_pipeline():
            base = progressive.masked_scores(
                query_codes,
                query_scales,
                index,
                base_mask,
                history_count,
            )
            sampled_residual = progressive.candidate_scores(
                query_codes,
                query_scales,
                index,
                residual_mask,
                sample_indices,
            )
            torch.kthvalue(
                sampled_residual.abs(),
                k=max(1, math.ceil(0.95 * sample_count)),
                dim=-1,
            )
            candidates = torch.topk(
                base,
                k=candidate_count,
                dim=-1,
            ).indices
            candidate_mask = torch.zeros_like(base, dtype=torch.uint8)
            candidate_mask.scatter_(-1, candidates, 1)
            residual = progressive.token_masked_scores(
                query_codes,
                query_scales,
                index,
                residual_mask,
                candidate_mask,
            )
            refined = (base + residual).masked_fill_(
                candidate_mask == 0,
                -torch.inf,
            )
            return torch.topk(refined, k=target, dim=-1)

        coalesced_pipeline_ms = elapsed_ms(
            coalesced_progressive_pipeline,
            warmup,
            repeats,
        )
        precomputed_candidates = torch.topk(
            base_reference,
            k=candidate_count,
            dim=-1,
        ).indices
        precomputed_mask = torch.zeros_like(
            base_reference,
            dtype=torch.uint8,
        )
        precomputed_mask.scatter_(-1, precomputed_candidates, 1)
        residual_only_ms = elapsed_ms(
            lambda: progressive.candidate_scores(
                query_codes,
                query_scales,
                index,
                residual_mask,
                precomputed_candidates,
            ),
            warmup,
            repeats,
        )
        coalesced_residual_only_ms = elapsed_ms(
            lambda: progressive.token_masked_scores(
                query_codes,
                query_scales,
                index,
                residual_mask,
                precomputed_mask,
            ),
            warmup,
            repeats,
        )
        conservative_access_ratio = (
            5.0
            + 10.0
            * min(1.0, candidate_ratio + sample_count / history_count)
        ) / 15.0
        rows.append(
            {
                "candidate_ratio": candidate_ratio,
                "candidate_count": candidate_count,
                "conservative_index_access_ratio": (
                    conservative_access_ratio
                ),
                "gather_residual_candidate_read_ms": residual_only_ms,
                "gather_progressive_pipeline_ms": gather_pipeline_ms,
                "gather_pipeline_speedup_vs_full": (
                    full_pipeline_ms / gather_pipeline_ms
                ),
                "coalesced_residual_candidate_read_ms": (
                    coalesced_residual_only_ms
                ),
                "coalesced_progressive_pipeline_ms": (
                    coalesced_pipeline_ms
                ),
                "coalesced_pipeline_speedup_vs_full": (
                    full_pipeline_ms / coalesced_pipeline_ms
                ),
            }
        )

    return {
        "history_count": history_count,
        "target_count": target,
        "score_max_abs_error": score_max_abs_error,
        "full_scan_ms": full_scan_ms,
        "base_scan_ms": base_scan_ms,
        "base_scan_speedup": full_scan_ms / base_scan_ms,
        "full_pipeline_ms": full_pipeline_ms,
        "candidate_replays": rows,
        "note": (
            "Candidate ratios replay trace observations; torch.topk(base, C) "
            "is a conservative stand-in for fused interval compaction."
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    lengths = parse_values(args.lengths, int)
    candidate_ratios = parse_values(args.candidate_ratios, float)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    results = [
        benchmark_length(
            length,
            candidate_ratios,
            args.warmup,
            args.repeats,
            generator,
        )
        for length in lengths
    ]
    payload = {
        "schema": "progressive_variablebit_pipeline_v1",
        "device": torch.cuda.get_device_name(),
        "allocation": list(FIXED_4421),
        "base_allocation": [4, 0, 0, 0, 0, 0, 0, 0],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
