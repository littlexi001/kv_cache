from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import run_controlled_public_kv_benchmark_v1 as lb
from run_critical_position_budget_probe_20260715 import load_model, run_one_token
from run_head_top2_targeted_ppl_20260714 import (
    capture_qk_trace,
    head_qabs_sampled_mass_mode,
    install_llama_head_top_fraction_patch,
    prefill_query_tail_mode,
    seed_packed_qmse_prefill_queries,
    set_attention_implementation,
)
from run_multitopic_lpcm_ppl_20260714 import (
    TOPICS,
    encode_topic_stream,
)
from run_sample_calibrated_longbench_20260717 import (
    QKSIEVE_FULLTOPK_METHOD,
    QKSIEVE_FULLTOPK_SCORE_MODE,
)


TRACE_SCHEMA = "qksieve_teacher_forced_drift_trace_v1"


def parse_nonnegative_ints(value: str, name: str) -> tuple[int, ...]:
    result = tuple(
        sorted({int(item) for item in value.split(",") if item.strip()})
    )
    if not result or result[0] < 0:
        raise ValueError(f"{name} must contain non-negative integers")
    return result


def build_teacher_forced_bundle(
    stream: list[int],
    *,
    history_tokens: int,
    recorded_query_tokens: int,
    continuation_steps: int,
) -> tuple[lb.PromptBundle, list[int]]:
    if history_tokens <= recorded_query_tokens:
        raise ValueError("history must exceed the recorded Query-tail window")
    if recorded_query_tokens <= 0 or continuation_steps <= 0:
        raise ValueError("recorded_query_tokens and continuation_steps are positive")
    required = history_tokens + continuation_steps
    if len(stream) < required:
        raise ValueError(
            f"stream has {len(stream)} tokens but {required} are required"
        )
    prompt_ids = stream[:history_tokens]
    continuation_ids = stream[history_tokens:required]
    prefix_tokens = history_tokens - recorded_query_tokens
    bundle = lb.PromptBundle(
        input_ids=torch.tensor([prompt_ids], dtype=torch.long),
        prefix_token_count=0,
        context_token_start=0,
        query_start=prefix_tokens,
        suffix_token_count=recorded_query_tokens,
        page_spans={},
    )
    return bundle, continuation_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--topic", choices=sorted(TOPICS), required=True)
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument(
        "--record_steps",
        default="0,1,3,7,15,31,63,127,255,511,1023,2047,4095",
    )
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--production_query_tokens", type=int, default=8)
    parser.add_argument("--recorded_query_tokens", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument(
        "--dataset_cache_dir",
        default="/home/fdong/ymluo/datasets/sklearn",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= args.query_shrinkage <= 1.0:
        raise ValueError("query_shrinkage must lie in [0, 1]")
    if (
        args.production_query_tokens <= 0
        or args.recorded_query_tokens < args.production_query_tokens
    ):
        raise ValueError(
            "recorded_query_tokens must cover production_query_tokens"
        )
    layers = parse_nonnegative_ints(args.layers, "layers")
    record_steps = parse_nonnegative_ints(args.record_steps, "record_steps")
    if record_steps[0] != 0 or record_steps[-1] >= args.steps:
        raise ValueError("record_steps must start at zero and lie in [0, steps)")

    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    stream = encode_topic_stream(
        tokenizer,
        TOPICS[args.topic],
        args.history_tokens + args.steps,
        args.dataset_cache_dir,
        args.seed,
    )
    bundle, continuation_ids = build_teacher_forced_bundle(
        stream,
        history_tokens=args.history_tokens,
        recorded_query_tokens=args.recorded_query_tokens,
        continuation_steps=args.steps,
    )
    suffix = bundle.input_ids[:, bundle.query_start :]
    prefill_queries: dict[int, torch.Tensor] = {}
    records: list[dict[str, Any]] = []

    with head_qabs_sampled_mass_mode(
        mass_threshold=0.75,
        budget_fractions=(0.06,),
        sample_fraction=0.0025,
        qabs_dim_count=8,
        candidate_fraction=0.06,
        use_cuda_kernels=True,
        skip_candidate_rerank=False,
        qabs_int2_onthefly=False,
        early_layer_count=0,
        early_budget_fraction=0.06,
        score_mode=QKSIEVE_FULLTOPK_SCORE_MODE,
        projection_dim=128,
        gqa_candidate_mode="independent",
        adaptive_rank_energy_threshold=0.85,
        adaptive_rank_residual_precision="int4",
        value_mass_threshold=1.0,
        partition_ucb_z=0.0,
        partition_overfetch_factor=0,
        qk_metric_query_shrinkage=args.query_shrinkage,
    ):
        set_attention_implementation(model, "sdpa")
        with head_qabs_sampled_mass_mode(None):
            cache, prefix_seconds = lb.prefill_prefix(
                model,
                bundle,
                input_device,
                args.prefill_chunk_tokens,
            )
        with (
            head_qabs_sampled_mass_mode(None),
            prefill_query_tail_mode(args.recorded_query_tokens) as captured,
        ):
            cache, _, suffix_seconds = lb.run_token_segment(
                model,
                suffix,
                cache,
                bundle.query_start,
                input_device,
            )
        prefill_queries.update(
            {
                int(layer): query.detach().cpu()
                for layer, query in captured.items()
            }
        )
        calibration_queries = {
            layer: query[..., -args.production_query_tokens :, :]
            for layer, query in captured.items()
        }
        seed_packed_qmse_prefill_queries(calibration_queries)

        set_attention_implementation(model, "eager")
        online_seconds = 0.0
        with capture_qk_trace(
            records,
            layers=layers,
            max_records_per_layer=len(record_steps),
            state_on_first_record_only=True,
            record_steps=record_steps,
        ):
            for step, token_id in enumerate(continuation_ids):
                cache, _, seconds, _ = run_one_token(
                    model,
                    int(token_id),
                    cache,
                    args.history_tokens + step,
                    input_device,
                )
                online_seconds += seconds

    for record in records:
        record["value"] = None
    payload = {
        "schema": TRACE_SCHEMA,
        "trace_kind": "teacher_forced_corpus_continuation",
        "model_name_or_path": args.model_name_or_path,
        "task": f"20newsgroups_{args.topic}",
        "sample_id": f"{args.topic}_seed{args.seed}",
        "method": QKSIEVE_FULLTOPK_METHOD,
        "score_mode": QKSIEVE_FULLTOPK_SCORE_MODE,
        "prompt_wrapper": "none",
        "prompt_truncation_mode": "teacher_forced_topic_stream",
        "prompt_tokens": args.history_tokens,
        "prefix_tokens": bundle.query_start,
        "suffix_tokens": bundle.suffix_token_count,
        "query_calibration_tokens": args.production_query_tokens,
        "recorded_prefill_query_tail_tokens": args.recorded_query_tokens,
        "qk_metric_query_shrinkage": args.query_shrinkage,
        "trace_layers": layers,
        "trace_steps": record_steps,
        "sequence_ids": continuation_ids,
        "prefill_query_tail": prefill_queries,
        "records": records,
        "timing": {
            "prefix_seconds": prefix_seconds,
            "suffix_seconds": suffix_seconds,
            "teacher_forced_online_seconds": online_seconds,
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    summary = {
        "schema": TRACE_SCHEMA,
        "topic": args.topic,
        "prompt_tokens": args.history_tokens,
        "continuation_tokens": len(continuation_ids),
        "record_steps": list(record_steps),
        "layers": list(layers),
        "records": len(records),
        "output_path": str(args.output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
