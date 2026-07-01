from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    clone_past_key_values,
    install_qwen3_attention_patch,
    pick_input_device,
    prefill_cache,
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import Config as RetrievalConfig, eval_task, write_csv  # noqa: E402
from run_qabs_downstream_task_suite import BUILDERS  # noqa: E402
from run_qabs_evidence_span_coverage import evidence_spans  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle span-retention diagnostic for QABS retrieval.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="compact_kv,json_kv,topic_table,needle_sentence")
    parser.add_argument("--tasks_per_variant", type=int, default=16)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=202606296)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.05)
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--protect_recent_tokens", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=8)
    return parser.parse_args()


def retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        tasks=args.tasks_per_variant,
        records_per_task=args.records_per_task,
        seed=args.seed,
        chunk_size=args.chunk_size,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        top_fraction=args.top_fraction,
        protect_sink_tokens=args.protect_sink_tokens,
        protect_recent_tokens=args.protect_recent_tokens,
        modes="",
        layer_budget_map_path="",
        log_every=args.log_every,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    unknown = [variant for variant in variants if variant not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(BUILDERS)}")

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, device)
    config = retrieval_config(args)

    result_rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    mode_specs = [
        ("baseline", "baseline", None, False),
        ("qabs8cand5reuse", "qabs8cand5reuse", None, False),
        ("oracle_key_label", "qabs8cand5reuse", ("key", "label"), True),
        ("oracle_record", "qabs8cand5reuse", ("record",), True),
    ]

    for variant_index, variant in enumerate(variants):
        rng = random.Random(args.seed + 1009 * variant_index)
        tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(args.tasks_per_variant)]
        for task_number, task in enumerate(tasks, start=1):
            if task_number == 1 or task_number == len(tasks) or task_number % args.log_every == 0:
                print(f"{variant} task {task_number}/{len(tasks)}", flush=True)
            spans = evidence_spans(tokenizer, task)
            span_rows.append(
                {
                    "variant": variant,
                    "task_id": task["task_id"],
                    "target_key": task["target_key"],
                    "target_label": task["target_label"],
                    "target_index": task["target_index"],
                    **{f"{name}_span": f"{span[0]}:{span[1]}" for name, span in spans.items()},
                }
            )
            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            context_cache, context_prev = prefill_cache(
                model,
                context_ids,
                context_ids.shape[-1],
                args.chunk_size,
                input_device,
            )
            for result_mode, eval_mode, span_names, force in mode_specs:
                active_spans = {name: spans[name] for name in span_names or () if name in spans}
                row = eval_task(
                    model,
                    tokenizer,
                    task,
                    config,
                    input_device,
                    eval_mode,
                    clone_past_key_values(context_cache),
                    context_prev.detach().clone(),
                    evidence_spans=active_spans,
                    force_evidence_spans=force,
                )
                row["variant"] = variant
                row["mode"] = result_mode
                row["forced_spans"] = ",".join(span_names or ())
                result_rows.append(row)

    fields = [
        "variant",
        "task_id",
        "mode",
        "forced_spans",
        "target_key",
        "target_index",
        "target_label",
        "pred_label",
        "correct",
    ] + [f"score_{label}" for label in ["A", "B", "C", "D"]]
    write_csv(output_dir / "oracle_span_results.csv", result_rows, fields)
    write_csv(
        output_dir / "task_spans.csv",
        span_rows,
        ["variant", "task_id", "target_key", "target_label", "target_index", "key_span", "label_span", "record_span"],
    )

    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        for result_mode, _, _, _ in mode_specs:
            subset = [row for row in result_rows if row["variant"] == variant and row["mode"] == result_mode]
            correct = sum(int(row["correct"]) for row in subset)
            summary_rows.append(
                {
                    "variant": variant,
                    "mode": result_mode,
                    "correct": correct,
                    "total": len(subset),
                    "accuracy": correct / max(1, len(subset)),
                }
            )
    write_csv(output_dir / "summary_by_variant_mode.csv", summary_rows, ["variant", "mode", "correct", "total", "accuracy"])
    (output_dir / "summary.json").write_text(json.dumps({"summary": summary_rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
