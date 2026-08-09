#!/usr/bin/env python
"""Audit the released FIER end-to-end benchmark with synchronized timing.

This script imports the upstream FIER model implementation directly.  It keeps
Full and FIER on the same paged-KV backend so both paths reuse the prefilled KV
cache.  It does not claim that the released selector is numerically faithful;
the upstream release currently creates random packed selector tensors during
decode, which is recorded in the output contract.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fier_root", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("full_same_backend", "fier_release"), required=True
    )
    parser.add_argument("--context_len", type=int, required=True)
    parser.add_argument("--decode_len", type=int, default=256)
    parser.add_argument("--token_budget", type=int, default=1280)
    parser.add_argument("--page_size_argument", type=int, default=1)
    parser.add_argument("--extra_cache_tokens", type=int, default=512)
    parser.add_argument(
        "--prefill_chunk_size",
        type=int,
        default=0,
        help="Feed prefill in cache-appending chunks; 0 keeps one-shot prefill.",
    )
    parser.add_argument(
        "--materialize_unused_dense_causal_mask",
        action="store_true",
        help=(
            "Preserve the released model's O(N^2) causal-mask allocation. "
            "FierAttention does not consume this mask, so the default skips it."
        ),
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use identical NF4 model weights for Full and FIER capacity runs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    root = args.fier_root.resolve()
    sys.path.insert(0, str(root))
    from fier import LlamaForCausalLM  # pylint: disable=import-outside-toplevel

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16
    load_kwargs = {
        "device_map": {"": str(device)},
        "torch_dtype": dtype,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig  # pylint: disable=import-outside-toplevel

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    with device:
        model = LlamaForCausalLM.from_pretrained(
            str(args.model_path),
            **load_kwargs,
        )
    model.eval()
    model_allocated_bytes = int(torch.cuda.memory_allocated(device))

    if not args.materialize_unused_dense_causal_mask:
        # FierAttention is causal inside the Quest prefill kernel and ignores the
        # dense mask argument. Avoid allocating an unused O(N^2) tensor.
        model.model._prepare_decoder_attention_mask = lambda *_args, **_kwargs: None

    max_seq_len = args.context_len + args.decode_len + args.extra_cache_tokens
    requested_budget = (
        max_seq_len if args.method == "full_same_backend" else args.token_budget
    )
    model.fier_init(
        page_size=args.page_size_argument,
        max_seq_len=max_seq_len,
        token_budget=requested_budget,
        dtype=dtype,
        device=device,
    )
    post_cache_allocated_bytes = int(torch.cuda.memory_allocated(device))
    controller = model.model.iController
    controller_page_size = int(controller.page_size)
    configured_page_budget = int(model.model._quest_page_budget)
    effective_token_budget = configured_page_budget * controller_page_size
    hidden_size = int(model._config.hidden_size)

    prefill_ms: list[float] = []
    decode_ms_per_token: list[float] = []
    for iteration in range(args.iterations):
        generator = torch.Generator(device=device).manual_seed(args.seed + iteration)
        prefill = torch.randn(
            1,
            args.context_len,
            hidden_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        decode_inputs = torch.randn(
            args.decode_len,
            1,
            1,
            hidden_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        chunk_size = args.prefill_chunk_size or args.context_len
        for chunk_start in range(0, args.context_len, chunk_size):
            model(
                inputs_embeds=prefill[
                    :, chunk_start : chunk_start + chunk_size, :
                ]
            )
        torch.cuda.synchronize(device)
        prefill_ms.append(1000.0 * (time.perf_counter() - start))

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for step in range(args.decode_len):
            model(inputs_embeds=decode_inputs[step])
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        decode_ms_per_token.append(1000.0 * elapsed / args.decode_len)
        model.fier_clear()
        del prefill, decode_inputs
        torch.cuda.empty_cache()

    result = {
        "schema": "fier_official_release_audit_v1",
        "fier_commit": "e0b34153591dd7a55171f09f30abee35b0f08356",
        "method": args.method,
        "model_path": str(args.model_path),
        "context_len": args.context_len,
        "decode_len": args.decode_len,
        "iterations": args.iterations,
        "requested_token_budget": requested_budget,
        "page_size_argument": args.page_size_argument,
        "extra_cache_tokens": args.extra_cache_tokens,
        "prefill_chunk_size": args.prefill_chunk_size or args.context_len,
        "unused_dense_causal_mask_materialized": (
            args.materialize_unused_dense_causal_mask
        ),
        "controller_page_size": controller_page_size,
        "configured_page_budget": configured_page_budget,
        "effective_token_budget_upper_bound": effective_token_budget,
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": decode_ms_per_token,
        "decode_median_ms_per_token": statistics.median(decode_ms_per_token),
        "decode_p10_ms_per_token": percentile(decode_ms_per_token, 0.10),
        "decode_p90_ms_per_token": percentile(decode_ms_per_token, 0.90),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "model_weight_mode": "nf4_double_quant" if args.load_in_4bit else "fp16",
        "model_allocated_bytes_after_load": model_allocated_bytes,
        "allocated_bytes_after_cache_init": post_cache_allocated_bytes,
        "claim_boundary": [
            "Uses the released FIER model and CUDA backend with synchronized timing.",
            "Uses synthetic hidden states, matching the released speed protocol.",
            "Full uses the same paged-KV backend and reuses the prefilled cache.",
            "The unused O(N^2) dense causal mask is skipped unless explicitly requested; Quest's prefill kernel remains causal.",
            "The released FIER selector currently creates random 2-bit packed tensors during decode; this run is a systems-path audit, not a joint quality-speed reproduction.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
