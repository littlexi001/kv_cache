from __future__ import annotations

import json

import torch

import variablebit_spectral_cuda_20260727 as variablebit


ALLOCATION = torch.tensor(
    [
        [8, 4, 2, 1, 0, 0, 0, 0],
        [4, 4, 2, 2, 1, 1, 0, 0],
    ],
    dtype=torch.int8,
).unsqueeze(0)


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(20260727)
    token_count = 257
    capacity = 320
    query_groups = 4
    projected_keys = torch.randn(
        1,
        2,
        token_count,
        128,
        dtype=torch.float16,
        device="cuda",
    )
    calibration_queries = torch.randn(
        1,
        2,
        23,
        128,
        dtype=torch.float16,
        device="cuda",
    )
    calibration_codes, calibration_scales = (
        variablebit.quantize_projected_query(calibration_queries)
    )
    reconstructed_calibration = (
        calibration_codes.float()
        * calibration_scales.float().repeat_interleave(16, dim=-1)
    )
    exact_query_mean = calibration_queries.float().mean(dim=2)
    proxy_query_mean = reconstructed_calibration.mean(dim=2)
    query_bands = calibration_queries.float().reshape(1, 2, 23, 8, 16)
    scale_metrics = torch.einsum(
        "bhqgd,bhqge->bhgde",
        query_bands,
        query_bands,
    ) / query_bands.shape[2]

    biased = variablebit.allocate_packed_index(
        ALLOCATION.cuda(),
        capacity,
        torch.float16,
        enable_score_bias=True,
    )
    biased["packed_codes"].zero_()
    biased["key_scales"].zero_()
    biased["score_bias"].zero_()
    variablebit.encode_projected_keys_into(
        projected_keys,
        biased,
        0,
        scale_metrics=scale_metrics,
        exact_query_mean=exact_query_mean,
        proxy_query_mean=proxy_query_mean,
        bias_shrinkage=0.5,
    )
    baseline = variablebit.allocate_packed_index(
        ALLOCATION.cuda(),
        capacity,
        torch.float16,
    )
    baseline["packed_codes"].zero_()
    baseline["key_scales"].zero_()
    variablebit.encode_projected_keys_into(
        projected_keys,
        baseline,
        0,
        scale_metrics=scale_metrics,
    )

    cpu_reference = variablebit.allocate_packed_index(
        ALLOCATION,
        capacity,
        torch.float16,
        enable_score_bias=True,
    )
    cpu_reference["packed_codes"].zero_()
    cpu_reference["key_scales"].zero_()
    cpu_reference["score_bias"].zero_()
    variablebit.encode_projected_keys_into(
        projected_keys.cpu(),
        cpu_reference,
        0,
        scale_metrics=scale_metrics.cpu(),
        exact_query_mean=exact_query_mean.cpu(),
        proxy_query_mean=proxy_query_mean.cpu(),
        bias_shrinkage=0.5,
    )

    future_queries = torch.randn(
        1,
        2,
        query_groups,
        128,
        dtype=torch.float16,
        device="cuda",
    )
    query_codes, query_scales = variablebit.quantize_projected_query(
        future_queries
    )
    common_arguments = (
        query_codes,
        query_scales,
        biased["packed_codes"],
        biased["key_scales"],
        biased["bit_allocations"],
        biased["code_offsets"],
        biased["scale_offsets"],
        biased["code_bases"],
        biased["scale_bases"],
        biased["code_strides"],
        biased["scale_strides"],
        token_count,
    )
    scores_without_bias = variablebit.scores(*common_arguments)
    scores_with_bias = variablebit.scores(
        *common_arguments,
        score_bias=biased["score_bias"],
    )
    expected_delta = biased["score_bias"][:, :, :token_count].repeat_interleave(
        query_groups,
        dim=1,
    )
    result = {
        "extension_version": "v13",
        "code_equal_to_unbiased": torch.equal(
            biased["packed_codes"],
            baseline["packed_codes"],
        ),
        "scale_max_abs_error_to_unbiased": float(
            (biased["key_scales"] - baseline["key_scales"])
            .float()
            .abs()
            .max()
            .item()
        ),
        "bias_max_abs_error_vs_cpu": float(
            (
                biased["score_bias"][:, :, :token_count].cpu().float()
                - cpu_reference["score_bias"][:, :, :token_count].float()
            )
            .abs()
            .max()
            .item()
        ),
        "score_delta_max_abs_error": float(
            (
                scores_with_bias
                - scores_without_bias
                - expected_delta.float()
            )
            .abs()
            .max()
            .item()
        ),
        "bias_mean_abs": float(
            biased["score_bias"][:, :, :token_count]
            .float()
            .abs()
            .mean()
            .item()
        ),
    }
    print(json.dumps(result, indent=2))
    if (
        not result["code_equal_to_unbiased"]
        or result["scale_max_abs_error_to_unbiased"] > 2.0e-3
        or result["bias_max_abs_error_vs_cpu"] > 2.0e-2
        or result["score_delta_max_abs_error"] > 2.0e-3
    ):
        raise AssertionError(result)


if __name__ == "__main__":
    main()
