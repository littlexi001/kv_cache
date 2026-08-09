from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

import run_controlled_public_kv_benchmark_v1 as lb
from offloaded_prefill_cache_20260716 import (
    QuantizedOffloadedExactPrefillCache,
)
from run_direct_countcap_denseprompt_ppl_20260725 import (
    MIXED_TOPIC_POOLS,
    encode_evaluation_stream,
)
from run_head_top2_targeted_ppl_20260714 import (
    install_llama_head_top_fraction_patch,
    load_model,
    prefill_query_tail_mode,
)
from run_multitopic_lpcm_ppl_20260714 import TOPICS, make_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--topic",
        choices=sorted(set(TOPICS) | set(MIXED_TOPIC_POOLS)),
        default="mixed_a",
    )
    parser.add_argument("--history_tokens", type=int, default=32000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--query_tail_tokens", type=int, default=8)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--top_fraction", type=float, default=0.01)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=4096)
    parser.add_argument(
        "--dataset_cache_dir",
        default="/home/fdong/ymluo/datasets/sklearn",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def grouped_kv_queries(
    query: torch.Tensor,
    kv_heads: int,
) -> torch.Tensor:
    batch, query_heads, query_tokens, head_dim = query.shape
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    return (
        query.reshape(
            batch,
            kv_heads,
            groups,
            query_tokens,
            head_dim,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, kv_heads, query_tokens * groups, head_dim)
    )


def empty_metric_accumulator() -> dict[str, float]:
    return {
        "key_squared_error": 0.0,
        "key_squared_norm": 0.0,
        "value_squared_error": 0.0,
        "value_squared_norm": 0.0,
        "score_squared_error": 0.0,
        "score_squared_norm": 0.0,
        "output_squared_error": 0.0,
        "output_squared_norm": 0.0,
        "score_sum_x": 0.0,
        "score_sum_y": 0.0,
        "score_sum_x2": 0.0,
        "score_sum_y2": 0.0,
        "score_sum_xy": 0.0,
        "score_count": 0.0,
        "topk_recall_sum": 0.0,
        "selected_mass_sum": 0.0,
        "query_count": 0.0,
    }


def add_tensor_error(
    accumulator: dict[str, float],
    prefix: str,
    exact: torch.Tensor,
    approximate: torch.Tensor,
) -> None:
    accumulator[f"{prefix}_squared_error"] += float(
        torch.sum((approximate.float() - exact.float()) ** 2).item()
    )
    accumulator[f"{prefix}_squared_norm"] += float(
        torch.sum(exact.float() ** 2).item()
    )


def add_score_statistics(
    accumulator: dict[str, float],
    exact: torch.Tensor,
    approximate: torch.Tensor,
) -> None:
    x = exact.float()
    y = approximate.float()
    accumulator["score_sum_x"] += float(x.sum().item())
    accumulator["score_sum_y"] += float(y.sum().item())
    accumulator["score_sum_x2"] += float((x * x).sum().item())
    accumulator["score_sum_y2"] += float((y * y).sum().item())
    accumulator["score_sum_xy"] += float((x * y).sum().item())
    accumulator["score_count"] += float(x.numel())


def finalize_metrics(
    accumulator: dict[str, float],
    *,
    key_group_size: int,
    value_group_size: int,
) -> dict[str, float]:
    count = accumulator["score_count"]
    covariance = (
        accumulator["score_sum_xy"]
        - accumulator["score_sum_x"]
        * accumulator["score_sum_y"]
        / count
    )
    variance_x = (
        accumulator["score_sum_x2"]
        - accumulator["score_sum_x"] ** 2 / count
    )
    variance_y = (
        accumulator["score_sum_y2"]
        - accumulator["score_sum_y"] ** 2 / count
    )
    return {
        "key_group_size": key_group_size,
        "value_group_size": value_group_size,
        "effective_bits_per_kv_coordinate": (
            4.0
            + 8.0 / key_group_size
            + 8.0 / value_group_size
        ),
        "key_nmse": (
            accumulator["key_squared_error"]
            / accumulator["key_squared_norm"]
        ),
        "value_nmse": (
            accumulator["value_squared_error"]
            / accumulator["value_squared_norm"]
        ),
        "score_nmse": (
            accumulator["score_squared_error"]
            / accumulator["score_squared_norm"]
        ),
        "score_pearson": covariance
        / math.sqrt(max(variance_x * variance_y, 1.0e-30)),
        "top1_percent_recall": (
            accumulator["topk_recall_sum"]
            / accumulator["query_count"]
        ),
        "selected_exact_attention_mass": (
            accumulator["selected_mass_sum"]
            / accumulator["query_count"]
        ),
        "attention_output_relative_mse": (
            accumulator["output_squared_error"]
            / accumulator["output_squared_norm"]
        ),
        "evaluated_queries": int(accumulator["query_count"]),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0 < args.query_tokens < args.history_tokens:
        raise ValueError("query_tokens must be in (0, history_tokens)")
    if args.sample_stride <= 0 or args.group_size <= 0:
        raise ValueError("sample stride and group size must be positive")
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top fraction must be in (0, 1]")

    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(
        SimpleNamespace(
            model_name_or_path=args.model_name_or_path,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
        )
    )
    stream = encode_evaluation_stream(
        tokenizer,
        args.topic,
        args.history_tokens + 8,
        args.dataset_cache_dir,
        args.seed,
    )
    history = stream[: args.history_tokens]
    remote_ids = history[: -args.query_tokens]
    query_ids = history[-args.query_tokens :]
    bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)
    cache = QuantizedOffloadedExactPrefillCache(
        capacity=args.history_tokens + 8,
        bits=4,
        group_size=args.group_size,
    )
    cache, _, _ = lb.run_token_segment(
        model,
        bundle.input_ids[:, : bundle.query_start],
        cache,
        0,
        input_device,
        args.prefill_chunk_tokens,
    )
    query_tensor = torch.tensor(
        [query_ids],
        dtype=torch.long,
        device=input_device,
    )
    with prefill_query_tail_mode(args.query_tail_tokens) as captured_queries:
        cache, _, _ = lb.run_token_segment(
            model,
            query_tensor,
            cache,
            len(remote_ids),
            input_device,
            args.prefill_chunk_tokens,
        )
    cache.synchronize()

    per_head_quantizer = QuantizedOffloadedExactPrefillCache(
        capacity=1,
        bits=4,
    )
    group_quantizer = QuantizedOffloadedExactPrefillCache(
        capacity=1,
        bits=4,
        group_size=args.group_size,
    )
    quantizers = {
        "int4_per_head": (per_head_quantizer, per_head_quantizer),
        f"int4_group{args.group_size}": (
            group_quantizer,
            group_quantizer,
        ),
        f"key_group{args.group_size}_value_per_head": (
            group_quantizer,
            per_head_quantizer,
        ),
        f"key_per_head_value_group{args.group_size}": (
            per_head_quantizer,
            group_quantizer,
        ),
    }
    accumulators = {
        name: empty_metric_accumulator() for name in quantizers
    }
    sampled_tokens = None
    head_dim = None
    for layer, state in enumerate(cache.completed_states()):
        exact_key = state.host_kv[
            0, ..., : state.length : args.sample_stride, :
        ].to(state.device)
        exact_value = state.host_kv[
            1, ..., : state.length : args.sample_stride, :
        ].to(state.device)
        sampled_tokens = int(exact_key.shape[-2])
        head_dim = int(exact_key.shape[-1])
        query = captured_queries[layer].to(state.device)
        grouped_query = grouped_kv_queries(
            query,
            kv_heads=int(exact_key.shape[1]),
        )
        exact_scores = torch.einsum(
            "bhqd,bhkd->bhqk",
            grouped_query.float(),
            exact_key.float(),
        ) / math.sqrt(head_dim)
        exact_attention = torch.softmax(exact_scores, dim=-1)
        exact_output = torch.einsum(
            "bhqk,bhkd->bhqd",
            exact_attention,
            exact_value.float(),
        )
        selected_count = max(
            1,
            int(math.ceil(args.top_fraction * sampled_tokens)),
        )
        exact_top = torch.topk(
            exact_scores,
            k=selected_count,
            dim=-1,
        ).indices

        for name, (key_quantizer, value_quantizer) in quantizers.items():
            key_codes, key_scales = key_quantizer._quantize(exact_key)
            value_codes, value_scales = value_quantizer._quantize(
                exact_value
            )
            approximate_key = key_quantizer._dequantize(
                key_codes,
                key_scales,
                exact_key.dtype,
                head_dim,
            )
            approximate_value = value_quantizer._dequantize(
                value_codes,
                value_scales,
                exact_value.dtype,
                head_dim,
            )
            approximate_scores = torch.einsum(
                "bhqd,bhkd->bhqk",
                grouped_query.float(),
                approximate_key.float(),
            ) / math.sqrt(head_dim)
            approximate_attention = torch.softmax(
                approximate_scores,
                dim=-1,
            )
            approximate_output = torch.einsum(
                "bhqk,bhkd->bhqd",
                approximate_attention,
                approximate_value.float(),
            )
            accumulator = accumulators[name]
            add_tensor_error(
                accumulator,
                "key",
                exact_key,
                approximate_key,
            )
            add_tensor_error(
                accumulator,
                "value",
                exact_value,
                approximate_value,
            )
            add_tensor_error(
                accumulator,
                "score",
                exact_scores,
                approximate_scores,
            )
            add_tensor_error(
                accumulator,
                "output",
                exact_output,
                approximate_output,
            )
            add_score_statistics(
                accumulator,
                exact_scores,
                approximate_scores,
            )
            approximate_top = torch.topk(
                approximate_scores,
                k=selected_count,
                dim=-1,
            ).indices
            recall = (
                (
                    approximate_top.unsqueeze(-1)
                    == exact_top.unsqueeze(-2)
                )
                .any(dim=-1)
                .float()
                .mean(dim=-1)
            )
            selected_mass = exact_attention.gather(
                dim=-1,
                index=approximate_top,
            ).sum(dim=-1)
            accumulator["topk_recall_sum"] += float(recall.sum().item())
            accumulator["selected_mass_sum"] += float(
                selected_mass.sum().item()
            )
            accumulator["query_count"] += float(recall.numel())
            del (
                key_codes,
                key_scales,
                value_codes,
                value_scales,
                approximate_key,
                approximate_value,
                approximate_scores,
                approximate_attention,
                approximate_output,
                approximate_top,
                recall,
                selected_mass,
            )
        del (
            exact_key,
            exact_value,
            query,
            grouped_query,
            exact_scores,
            exact_attention,
            exact_output,
            exact_top,
        )

    assert sampled_tokens is not None and head_dim is not None
    methods = {
        name: finalize_metrics(
            accumulator,
            key_group_size=(
                args.group_size
                if name.startswith(f"int4_group{args.group_size}")
                or name.startswith(f"key_group{args.group_size}")
                else head_dim
            ),
            value_group_size=(
                args.group_size
                if name.startswith(f"int4_group{args.group_size}")
                or name.endswith(f"value_group{args.group_size}")
                else head_dim
            ),
        )
        for name, accumulator in accumulators.items()
    }
    group_name = f"int4_group{args.group_size}"
    key_group_name = f"key_group{args.group_size}_value_per_head"
    value_group_name = f"key_per_head_value_group{args.group_size}"
    summary = {
        "schema": "groupwise_prefill_quantization_mechanism_v1",
        "model_name_or_path": args.model_name_or_path,
        "topic": args.topic,
        "history_tokens": args.history_tokens,
        "sample_stride": args.sample_stride,
        "sampled_tokens_per_layer": sampled_tokens,
        "query_tail_tokens": args.query_tail_tokens,
        "top_fraction": args.top_fraction,
        "methods": methods,
        "groupwise_over_per_head": {
            "key_nmse_ratio": (
                methods[group_name]["key_nmse"]
                / methods["int4_per_head"]["key_nmse"]
            ),
            "value_nmse_ratio": (
                methods[group_name]["value_nmse"]
                / methods["int4_per_head"]["value_nmse"]
            ),
            "score_nmse_ratio": (
                methods[group_name]["score_nmse"]
                / methods["int4_per_head"]["score_nmse"]
            ),
            "output_relative_mse_ratio": (
                methods[group_name]["attention_output_relative_mse"]
                / methods["int4_per_head"][
                    "attention_output_relative_mse"
                ]
            ),
            "top1_recall_gain_points": 100.0
            * (
                methods[group_name]["top1_percent_recall"]
                - methods["int4_per_head"]["top1_percent_recall"]
            ),
            "selected_mass_gain_points": 100.0
            * (
                methods[group_name]["selected_exact_attention_mass"]
                - methods["int4_per_head"][
                    "selected_exact_attention_mass"
                ]
            ),
        },
        "asymmetric_diagnosis": {
            "key_group_value_per_head_output_mse_ratio_vs_both_group": (
                methods[key_group_name]["attention_output_relative_mse"]
                / methods[group_name]["attention_output_relative_mse"]
            ),
            "key_per_head_value_group_output_mse_ratio_vs_both_group": (
                methods[value_group_name]["attention_output_relative_mse"]
                / methods[group_name]["attention_output_relative_mse"]
            ),
            "key_group_value_per_head_top1_recall": methods[
                key_group_name
            ]["top1_percent_recall"],
            "key_per_head_value_group_top1_recall": methods[
                value_group_name
            ]["top1_percent_recall"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
