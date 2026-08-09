from __future__ import annotations

import argparse
import json

import torch

import qabs_cuda_kernels as qabs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active_tokens", type=int, default=511)
    parser.add_argument("--capacity_tokens", type=int, default=640)
    parser.add_argument("--candidate_tokens", type=int, default=32)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.capacity_tokens <= args.active_tokens:
        raise ValueError("capacity_tokens must exceed active_tokens")

    torch.manual_seed(20260724)
    device = torch.device("cuda")
    dtype = torch.float16
    batch_count = 1
    query_head_count = 32
    kv_head_count = 8
    head_dim = 128

    query = torch.randn(
        batch_count,
        query_head_count,
        head_dim,
        dtype=dtype,
        device=device,
    )
    key_buffer = torch.randn(
        batch_count,
        kv_head_count,
        args.capacity_tokens,
        head_dim,
        dtype=dtype,
        device=device,
    )
    value_buffer = torch.randn_like(key_buffer)
    key = key_buffer[..., : args.active_tokens, :]
    value = value_buffer[..., : args.active_tokens, :]
    indices = torch.randint(
        0,
        args.active_tokens - 1,
        (
            batch_count,
            query_head_count,
            args.candidate_tokens,
        ),
        dtype=torch.long,
        device=device,
    )
    counts = torch.randint(
        max(1, args.candidate_tokens // 2),
        args.candidate_tokens + 1,
        (batch_count, query_head_count),
        dtype=torch.long,
        device=device,
    )
    scaling = head_dim**-0.5

    contiguous_output = qabs.final_attention_ragged_self(
        query,
        key.contiguous(),
        value.contiguous(),
        indices,
        counts,
        scaling,
    )
    strided_output = qabs.final_attention_ragged_self(
        query,
        key,
        value,
        indices,
        counts,
        scaling,
    )
    contiguous_split_output = qabs.final_attention_ragged_self_split(
        query,
        key.contiguous(),
        value.contiguous(),
        indices,
        counts,
        scaling,
        split_count=4,
    )
    strided_split_output = qabs.final_attention_ragged_self_split(
        query,
        key,
        value,
        indices,
        counts,
        scaling,
        split_count=4,
    )
    torch.cuda.synchronize()
    max_abs_error = float(
        (contiguous_output.float() - strided_output.float()).abs().max().item()
    )
    split_max_abs_error = float(
        (
            contiguous_split_output.float()
            - strided_split_output.float()
        ).abs().max().item()
    )
    result = {
        "active_tokens": args.active_tokens,
        "capacity_tokens": args.capacity_tokens,
        "key_is_contiguous": bool(key.is_contiguous()),
        "key_stride": list(key.stride()),
        "contiguous_key_stride": list(key.contiguous().stride()),
        "max_abs_error": max_abs_error,
        "exact_match": bool(torch.equal(contiguous_output, strided_output)),
        "split4_max_abs_error": split_max_abs_error,
        "split4_exact_match": bool(
            torch.equal(contiguous_split_output, strided_split_output)
        ),
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    if max_abs_error != 0.0 or split_max_abs_error != 0.0:
        raise RuntimeError(
            "strided attention mismatch: "
            f"single={max_abs_error}, split4={split_max_abs_error}"
        )


if __name__ == "__main__":
    main()
