from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import benchmark_variablebit_spectral_attention_20260727 as varbit_bench
import qabs_cuda_kernels as sparse_cuda
import qksieve_query_cuda_20260728 as qfused_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item) for item in spec.split(",") if item.strip()})
    if not values or values[0] <= 0:
        raise ValueError("lengths must contain positive integers")
    return values


def parse_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError("dtype must be float16 or bfloat16")


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def target_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def measure_ms(
    function: Callable[[], object],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def paired_measure_ms(
    baseline: Callable[[], object],
    candidate: Callable[[], object],
    warmup: int,
    iterations: int,
    repeats: int,
    order_seed: int,
) -> tuple[float, float]:
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    for repeat in range(repeats):
        candidate_first = (order_seed + repeat) % 2 == 1
        if candidate_first:
            candidate_times.append(
                measure_ms(candidate, warmup, iterations)
            )
            baseline_times.append(
                measure_ms(baseline, warmup, iterations)
            )
        else:
            baseline_times.append(
                measure_ms(baseline, warmup, iterations)
            )
            candidate_times.append(
                measure_ms(candidate, warmup, iterations)
            )
    return (
        float(statistics.median(baseline_times)),
        float(statistics.median(candidate_times)),
    )


def balanced_bases(
    generator: torch.Generator,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.randn(
        1,
        8,
        128,
        128,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    orthogonal, _ = torch.linalg.qr(raw)
    sigma = torch.logspace(
        -0.6,
        0.6,
        128,
        dtype=torch.float32,
        device="cuda",
    )
    query_basis = orthogonal * sigma.sqrt().view(1, 1, 1, 128)
    key_basis = orthogonal / sigma.sqrt().view(1, 1, 1, 128)
    return query_basis.to(dtype), key_basis.to(dtype)


def exact_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    candidate_indices: torch.Tensor,
    selected_count: int,
) -> torch.Tensor:
    counts = torch.full(
        candidate_indices.shape[:2],
        selected_count,
        dtype=torch.long,
        device=query.device,
    )
    scaling = 128.0**-0.5
    if selected_count >= 1280:
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            candidate_indices.contiguous(),
            counts,
            scaling,
            4,
        )
    if selected_count >= 900:
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            candidate_indices.contiguous(),
            counts,
            scaling,
            2,
        )
    return sparse_cuda.final_attention_ragged_self(
        query,
        key,
        value,
        candidate_indices.contiguous(),
        counts,
        scaling,
    )


def rowwise_topk_recall(
    reference_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    sorted_reference = reference_indices.sort(dim=-1).values
    insertion = torch.searchsorted(sorted_reference, candidate_indices)
    insertion = insertion.clamp_max(sorted_reference.shape[-1] - 1)
    matches = sorted_reference.gather(-1, insertion) == candidate_indices
    return matches.float().mean(dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the experimental fused QKSieve query preparation kernel "
            "against the frozen einsum-plus-INT8 path."
        )
    )
    parser.add_argument("--lengths", default="4096,32768")
    parser.add_argument("--group_count", type=int, default=4)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timing_repeats", type=int, default=5)
    parser.add_argument("--min_code_match", type=float, default=0.90)
    parser.add_argument("--max_scale_relative_p99", type=float, default=0.01)
    parser.add_argument("--max_score_nrmse", type=float, default=0.01)
    parser.add_argument("--min_topk_recall", type=float, default=0.995)
    parser.add_argument("--min_output_cosine", type=float, default=0.999)
    parser.add_argument("--max_output_rmse", type=float, default=0.01)
    parser.add_argument(
        "--min_query_prepare_speedup",
        type=float,
        default=1.05,
    )
    parser.add_argument(
        "--min_selection_speedup",
        type=float,
        default=1.00,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if (
        args.trials <= 0
        or args.warmup < 0
        or args.iterations <= 0
        or args.timing_repeats <= 0
    ):
        raise ValueError("invalid trial or timing counts")
    if not 1 <= args.group_count <= 16:
        raise ValueError("group_count must be in [1, 16]")

    torch.manual_seed(20260728)
    dtype = parse_dtype(args.dtype)
    allocation = varbit_bench.ALLOCATION_PROFILES[
        "qmse_total_b15"
    ].unsqueeze(0).cuda()
    rows: list[dict[str, float | int | bool]] = []

    for history_tokens in parse_ints(args.lengths):
        selected_count = target_count(history_tokens)
        for trial in range(args.trials):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(
                20260728
                + 1_000_003 * args.group_count
                + 1009 * history_tokens
                + 97 * (1 if dtype == torch.bfloat16 else 0)
                + trial
            )
            query_basis, key_basis = balanced_bases(generator, dtype)
            grouped_query = torch.randn(
                1,
                8,
                args.group_count,
                128,
                generator=generator,
                dtype=dtype,
                device="cuda",
            )
            query = grouped_query.reshape(
                1,
                8 * args.group_count,
                128,
            )
            key = torch.randn(
                1,
                8,
                history_tokens,
                128,
                generator=generator,
                dtype=dtype,
                device="cuda",
            )
            value = torch.randn(
                key.shape,
                generator=generator,
                dtype=dtype,
                device="cuda",
            )
            packed_index = varbit_cuda.allocate_packed_index(
                allocation,
                history_tokens,
                dtype,
            )
            projected_key = torch.einsum(
                "bhnd,bhdm->bhnm",
                key,
                key_basis,
            )
            varbit_cuda.encode_projected_keys_into(
                projected_key.contiguous(),
                packed_index,
                0,
            )

            def unfused_prepare() -> tuple[torch.Tensor, torch.Tensor]:
                projected = torch.einsum(
                    "bhgd,bhdm->bhgm",
                    grouped_query,
                    query_basis,
                )
                return varbit_cuda.quantize_projected_query(projected)

            def fused_prepare() -> tuple[torch.Tensor, torch.Tensor]:
                return qfused_cuda.project_quantize(
                    grouped_query,
                    query_basis,
                )

            reference_codes, reference_scales = unfused_prepare()
            fused_codes, fused_scales = fused_prepare()

            def score(
                codes: torch.Tensor,
                scales: torch.Tensor,
            ) -> torch.Tensor:
                query_head_count = grouped_query.shape[1] * args.group_count
                return varbit_cuda.scores(
                    codes,
                    scales,
                    packed_index["packed_codes"],
                    packed_index["key_scales"],
                    packed_index["bit_allocations"],
                    packed_index["code_offsets"],
                    packed_index["scale_offsets"],
                    packed_index["code_bases"],
                    packed_index["scale_bases"],
                    packed_index["code_strides"],
                    packed_index["scale_strides"],
                    history_tokens,
                    score_bias=packed_index.get("score_bias"),
                ).reshape(1, query_head_count, history_tokens)

            reference_scores = score(reference_codes, reference_scales)
            fused_scores = score(fused_codes, fused_scales)
            reference_indices = torch.topk(
                reference_scores,
                selected_count,
                dim=-1,
                sorted=False,
            ).indices
            fused_indices = torch.topk(
                fused_scores,
                selected_count,
                dim=-1,
                sorted=False,
            ).indices
            reference_output = exact_sparse_attention(
                query,
                key,
                value,
                reference_indices,
                selected_count,
            )
            fused_output = exact_sparse_attention(
                query,
                key,
                value,
                fused_indices,
                selected_count,
            )

            code_match_by_head = (
                (reference_codes == fused_codes).float().mean(dim=-1)
            )
            scale_relative = (
                (reference_scales.float() - fused_scales.float()).abs()
                / reference_scales.float().abs().clamp_min(1.0e-8)
            )
            scale_relative_p99_by_head = torch.quantile(
                scale_relative,
                0.99,
                dim=-1,
            )
            score_error = (reference_scores - fused_scores).float()
            score_rmse_by_head = (
                score_error.square().mean(dim=-1).sqrt()
            )
            score_nrmse_by_head = (
                score_rmse_by_head
                / reference_scores.float().std(dim=-1).clamp_min(1.0e-8)
            )
            topk_recall_by_head = rowwise_topk_recall(
                reference_indices,
                fused_indices,
            )
            output_cosine_by_head = F.cosine_similarity(
                reference_output.float(),
                fused_output.float(),
                dim=-1,
            )
            output_rmse_by_head = (
                (reference_output - fused_output)
                .float()
                .square()
                .mean(dim=-1)
                .sqrt()
            )
            code_match = float(code_match_by_head.min())
            scale_relative_p99 = float(scale_relative_p99_by_head.max())
            score_nrmse = float(score_nrmse_by_head.max())
            topk_recall = float(topk_recall_by_head.min())
            output_cosine = float(output_cosine_by_head.min())
            output_rmse = float(output_rmse_by_head.max())

            def unfused_selection() -> torch.Tensor:
                codes, scales = unfused_prepare()
                return torch.topk(
                    score(codes, scales),
                    selected_count,
                    dim=-1,
                    sorted=False,
                ).indices

            def fused_selection() -> torch.Tensor:
                codes, scales = fused_prepare()
                return torch.topk(
                    score(codes, scales),
                    selected_count,
                    dim=-1,
                    sorted=False,
                ).indices

            unfused_prepare_ms, fused_prepare_ms = paired_measure_ms(
                unfused_prepare,
                fused_prepare,
                args.warmup,
                args.iterations,
                args.timing_repeats,
                order_seed=trial,
            )
            unfused_selection_ms, fused_selection_ms = paired_measure_ms(
                unfused_selection,
                fused_selection,
                args.warmup,
                args.iterations,
                args.timing_repeats,
                order_seed=trial + 1,
            )
            passed = (
                code_match >= args.min_code_match
                and scale_relative_p99 <= args.max_scale_relative_p99
                and score_nrmse <= args.max_score_nrmse
                and topk_recall >= args.min_topk_recall
                and output_cosine >= args.min_output_cosine
                and output_rmse <= args.max_output_rmse
                and unfused_prepare_ms / fused_prepare_ms
                >= args.min_query_prepare_speedup
                and unfused_selection_ms / fused_selection_ms
                >= args.min_selection_speedup
            )
            rows.append(
                {
                    "dtype": dtype_name(dtype),
                    "group_count": args.group_count,
                    "history_tokens": history_tokens,
                    "selected_tokens": selected_count,
                    "trial": trial,
                    "code_exact_match": code_match,
                    "code_exact_match_mean": float(
                        code_match_by_head.mean()
                    ),
                    "scale_relative_p99": scale_relative_p99,
                    "score_nrmse": score_nrmse,
                    "score_nrmse_mean": float(score_nrmse_by_head.mean()),
                    "topk_recall": topk_recall,
                    "topk_recall_mean": float(topk_recall_by_head.mean()),
                    "output_cosine": output_cosine,
                    "output_cosine_mean": float(
                        output_cosine_by_head.mean()
                    ),
                    "output_rmse": output_rmse,
                    "output_rmse_mean": float(output_rmse_by_head.mean()),
                    "unfused_query_prepare_ms": unfused_prepare_ms,
                    "fused_query_prepare_ms": fused_prepare_ms,
                    "query_prepare_speedup": (
                        unfused_prepare_ms / fused_prepare_ms
                    ),
                    "unfused_selection_ms": unfused_selection_ms,
                    "fused_selection_ms": fused_selection_ms,
                    "selection_speedup": (
                        unfused_selection_ms / fused_selection_ms
                    ),
                    "passed": passed,
                }
            )

    report = {
        "schema": "qksieve_qfused_correctness_v2",
        "baseline": (
            "einsum query projection followed by bandwise INT8 quantization"
        ),
        "candidate": "fused CUDA query projection and bandwise INT8",
        "kernel_layout": "coalesced-output-dimension-gqa-basis-reuse-v2",
        "dtype": dtype_name(dtype),
        "group_count": args.group_count,
        "projection_rounding": (
            "FP32 accumulate then model-dtype rounding before INT8"
        ),
        "timing_protocol": (
            "median of alternating baseline/candidate measurement order"
        ),
        "timing_repeats": args.timing_repeats,
        "correctness_reduction": (
            "minimum agreement or maximum error over query heads"
        ),
        "acceptance": {
            "min_code_match": args.min_code_match,
            "max_scale_relative_p99": args.max_scale_relative_p99,
            "max_score_nrmse": args.max_score_nrmse,
            "min_topk_recall": args.min_topk_recall,
            "min_output_cosine": args.min_output_cosine,
            "max_output_rmse": args.max_output_rmse,
            "min_query_prepare_speedup": args.min_query_prepare_speedup,
            "min_selection_speedup": args.min_selection_speedup,
        },
        "all_passed": all(bool(row["passed"]) for row in rows),
        "rows": rows,
    }
    source_root = Path(__file__).resolve().parent
    report["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            source_root / "qksieve_query_cuda_20260728.py",
            source_root / "variablebit_spectral_cuda_20260727.py",
            source_root / "qabs_cuda_kernels.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
