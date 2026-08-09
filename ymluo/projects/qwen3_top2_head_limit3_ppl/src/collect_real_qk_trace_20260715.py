from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

import run_controlled_public_kv_benchmark_v1 as lb
from run_critical_position_budget_probe_20260715 import load_model, run_one_token
from run_head_top2_targeted_ppl_20260714 import (
    capture_qk_trace,
    head_adaptive_mass_mode,
    head_qabs_sampled_mass_mode,
    head_top_fraction_mode,
    install_llama_head_top_fraction_patch,
    prefill_query_tail_mode,
    set_attention_implementation,
)
from run_multitopic_lpcm_ppl_20260714 import (
    TOPICS,
    encode_topic_stream,
    make_bundle,
)


def encode_jsonl_stream(
    tokenizer: Any,
    source_path: Path,
    source_field: str,
    required_tokens: int,
    seed: int,
    *,
    repeat_documents: bool,
) -> list[int]:
    documents: list[str] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(source_field)
            if value is None:
                raise KeyError(
                    f"{source_path}:{line_number} has no field {source_field!r}"
                )
            text = str(value).strip()
            if text:
                documents.append(text)
    if not documents:
        raise RuntimeError(f"{source_path} contains no non-empty documents")

    stream: list[int] = []
    separator = "\n\n---\n\n"
    cycle = 0
    while len(stream) < required_tokens:
        cycle_documents = list(documents)
        random.Random(seed + cycle).shuffle(cycle_documents)
        for document in cycle_documents:
            stream.extend(
                tokenizer(
                    separator + document,
                    add_special_tokens=False,
                )["input_ids"]
            )
            if len(stream) >= required_tokens:
                break
        if not repeat_documents or not cycle_documents:
            break
        cycle += 1
    if len(stream) < required_tokens:
        raise RuntimeError(
            f"{source_path} has only {len(stream)} usable tokens, "
            f"need {required_tokens}; pass --repeat_source_documents to cycle it"
        )
    return stream[:required_tokens]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--topic", choices=sorted(TOPICS), default="sports")
    parser.add_argument("--history_tokens", type=int, default=32000)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--record_steps",
        default="",
        help=(
            "Optional comma-separated zero-based decode steps to capture. "
            "All steps are captured when omitted."
        ),
    )
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument(
        "--prefill_query_tail_tokens",
        type=int,
        default=0,
        help=(
            "Retain this many final prefill Queries per layer for request-local "
            "calibration; zero disables capture."
        ),
    )
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument(
        "--source_jsonl",
        type=Path,
        default=None,
        help=(
            "Optional local JSONL corpus. When set, its source_field replaces "
            "the 20 Newsgroups topic stream and no dataset download is needed."
        ),
    )
    parser.add_argument("--source_field", default="context")
    parser.add_argument("--repeat_source_documents", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--omit_values",
        action="store_true",
        help="Drop captured Value tensors before saving a QK-only trace.",
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    record_steps = tuple(
        sorted(
            {
                int(item)
                for item in args.record_steps.split(",")
                if item.strip()
            }
        )
    )
    if record_steps and (
        record_steps[0] < 0 or record_steps[-1] >= args.steps
    ):
        raise ValueError("record_steps must lie in [0, steps)")

    layers = tuple(int(item) for item in args.layers.split(",") if item)
    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    required_tokens = args.history_tokens + args.steps - 1
    if args.source_jsonl is not None:
        stream = encode_jsonl_stream(
            tokenizer,
            args.source_jsonl,
            args.source_field,
            required_tokens,
            args.seed,
            repeat_documents=args.repeat_source_documents,
        )
    else:
        stream = encode_topic_stream(
            tokenizer,
            TOPICS[args.topic],
            required_tokens,
            args.dataset_cache_dir,
            args.seed,
        )
    remote_ids = stream[: args.history_tokens - 1]
    step_ids = stream[args.history_tokens - 1 : args.history_tokens - 1 + args.steps]
    bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)
    set_attention_implementation(model, "sdpa")
    if args.prefill_query_tail_tokens > 0:
        query_context = prefill_query_tail_mode(
            args.prefill_query_tail_tokens
        )
    else:
        from contextlib import nullcontext

        query_context = nullcontext({})
    with (
        head_top_fraction_mode(None),
        head_adaptive_mass_mode(None),
        head_qabs_sampled_mass_mode(None),
        query_context as prefill_queries,
    ):
        prefix_cache, prefill_seconds = lb.prefill_prefix(
            model, bundle, input_device, args.prefill_chunk_tokens
        )

    records: list[dict[str, object]] = []
    set_attention_implementation(model, "eager")
    with (
        head_top_fraction_mode(None),
        head_adaptive_mass_mode(None),
        head_qabs_sampled_mass_mode(None),
        capture_qk_trace(
            records,
            layers,
            max_records_per_layer=(
                len(record_steps) if record_steps else args.steps
            ),
            state_on_first_record_only=args.steps > 1,
            record_steps=record_steps or None,
        ),
    ):
        cache = prefix_cache
        online_seconds = 0.0
        for step, token_id in enumerate(step_ids):
            cache, _, seconds, _ = run_one_token(
                model,
                int(token_id),
                cache,
                len(remote_ids) + step,
                input_device,
            )
            online_seconds += seconds
    if args.omit_values:
        for record in records:
            record["value"] = None
    payload = {
        "config": vars(args),
        "prefill_seconds": prefill_seconds,
        "online_seconds": online_seconds,
        "record_steps": record_steps or tuple(range(args.steps)),
        "prefill_queries": {
            int(layer): tensor.detach().cpu()
            for layer, tensor in prefill_queries.items()
            if int(layer) in layers
        },
        "records": records,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    summary = {
        "layers": [int(record["layer"]) for record in records],
        "query_shapes": [list(record["query"].shape) for record in records],
        "key_shapes": [
            list(record["key"].shape) if record["key"] is not None else None
            for record in records
        ],
        "value_shapes": [
            list(record["value"].shape) if record["value"] is not None else None
            for record in records
        ],
        "prefill_query_shapes": {
            str(layer): list(tensor.shape)
            for layer, tensor in payload["prefill_queries"].items()
        },
        "output_path": str(args.output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
