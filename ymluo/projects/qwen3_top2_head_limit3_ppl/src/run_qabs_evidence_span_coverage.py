from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    EvidenceSpanCoverageStats,
    QabsReuseProfileStats,
    clone_past_key_values,
    install_qwen3_attention_patch,
    pick_input_device,
    prefill_cache,
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import Config as RetrievalConfig, eval_task  # noqa: E402
from run_qabs_downstream_task_suite import BUILDERS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence span coverage diagnostic for QABS retrieval failures.")
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


def _token_count(tokenizer: Any, text: str) -> int:
    return int(tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1])


def char_span_to_token_span(tokenizer: Any, text: str, start: int, end: int) -> tuple[int, int]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        token_indices = [
            idx
            for idx, (token_start, token_end) in enumerate(offsets)
            if int(token_start) < end and int(token_end) > start
        ]
        if token_indices:
            return min(token_indices), max(token_indices) + 1
    except Exception:
        pass
    token_start = _token_count(tokenizer, text[:start])
    token_end = _token_count(tokenizer, text[:end])
    if token_end <= token_start and token_start > 0:
        token_start -= 1
    return token_start, max(token_start + 1, token_end)


def target_line_span(context: str, key: str) -> tuple[int, int]:
    key_pos = context.index(key)
    line_start = context.rfind("\n", 0, key_pos) + 1
    line_end = context.find("\n", key_pos)
    if line_end < 0:
        line_end = len(context)
    return line_start, line_end


def label_char_span(task: dict[str, Any], line: str, line_start: int) -> tuple[int, int]:
    label = re.escape(task["target_label"])
    variant = str(task.get("variant", ""))
    patterns = {
        "compact_kv": rf"=>\s*({label})\b",
        "natural_kv": rf"answer label\s+({label})\b",
        "json_kv": rf'"answer_label":"({label})"',
        "needle_sentence": rf"option\s+({label})\b",
        "topic_table": rf"class=({label})\b",
        "structured_noisy": rf"ANSWER_LABEL=({label})\b",
    }
    pattern = patterns.get(variant, rf"\b({label})\b")
    match = re.search(pattern, line)
    if match is None:
        raise ValueError(f"could not locate label span for variant={variant}: {line}")
    return line_start + match.start(1), line_start + match.end(1)


def evidence_spans(tokenizer: Any, task: dict[str, Any]) -> dict[str, tuple[int, int]]:
    context = task["context"]
    key = task["target_key"]
    key_start = context.index(key)
    key_end = key_start + len(key)
    line_start, line_end = target_line_span(context, key)
    label_start, label_end = label_char_span(task, context[line_start:line_end], line_start)
    spans = {
        "key": char_span_to_token_span(tokenizer, context, key_start, key_end),
        "label": char_span_to_token_span(tokenizer, context, label_start, label_end),
        "record": char_span_to_token_span(tokenizer, context, line_start, line_end),
    }
    return {name: span for name, span in spans.items() if span[0] < span[1]}


def retrieval_config(args: argparse.Namespace, output_dir: str) -> RetrievalConfig:
    return RetrievalConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=output_dir,
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
        modes="baseline,qabs8cand5reuse",
        layer_budget_map_path="",
        log_every=args.log_every,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    layer_count = int(getattr(model.config, "num_hidden_layers"))
    head_count = int(getattr(model.config, "num_attention_heads"))
    config = retrieval_config(args, str(output_dir))

    result_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    coverage_overall_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    for variant_index, variant in enumerate(variants):
        rng = random.Random(args.seed + 1009 * variant_index)
        tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(args.tasks_per_variant)]
        for task_number, task in enumerate(tasks, start=1):
            if task_number == 1 or task_number == len(tasks) or task_number % args.log_every == 0:
                print(f"{variant} task {task_number}/{len(tasks)}", flush=True)
            spans = evidence_spans(tokenizer, task)
            task_rows.append(
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
            baseline = eval_task(
                model,
                tokenizer,
                task,
                config,
                input_device,
                "baseline",
                clone_past_key_values(context_cache),
                context_prev.detach().clone(),
            )
            baseline["variant"] = variant
            result_rows.append(baseline)

            profile_stats = QabsReuseProfileStats(layer_count, head_count, sync_cuda=False)
            evidence_stats = EvidenceSpanCoverageStats(layer_count, head_count)
            qabs = eval_task(
                model,
                tokenizer,
                task,
                config,
                input_device,
                "qabs8cand5reuse",
                clone_past_key_values(context_cache),
                context_prev.detach().clone(),
                qabs_profile_stats=profile_stats,
                evidence_coverage_stats=evidence_stats,
                evidence_spans=spans,
            )
            qabs["variant"] = variant
            result_rows.append(qabs)
            coverage_rows.extend(evidence_stats.rows(task["task_id"], variant, "qabs8cand5reuse", int(qabs["correct"])))
            coverage_overall_rows.extend(
                evidence_stats.overall_rows(task["task_id"], variant, "qabs8cand5reuse", int(qabs["correct"]))
            )

    write_csv(
        output_dir / "task_results.csv",
        result_rows,
        ["variant", "task_id", "mode", "target_key", "target_index", "target_label", "pred_label", "correct"]
        + [f"score_{label}" for label in ["A", "B", "C", "D"]],
    )
    write_csv(
        output_dir / "task_spans.csv",
        task_rows,
        ["variant", "task_id", "target_key", "target_label", "target_index", "key_span", "label_span", "record_span"],
    )
    write_csv(
        output_dir / "coverage_by_task_layer_head.csv",
        coverage_rows,
        ["task_id", "variant", "mode", "correct", "layer", "head", "mask", "span", "metric", "hit_count", "query_count", "coverage"],
    )
    write_csv(
        output_dir / "coverage_by_task_overall.csv",
        coverage_overall_rows,
        ["task_id", "variant", "mode", "correct", "mask", "span", "metric", "hit_count", "query_count", "coverage"],
    )

    summary: list[dict[str, Any]] = []
    for variant in variants:
        for mode in ["baseline", "qabs8cand5reuse"]:
            subset = [row for row in result_rows if row["variant"] == variant and row["mode"] == mode]
            correct = sum(int(row["correct"]) for row in subset)
            summary.append({"variant": variant, "mode": mode, "correct": correct, "total": len(subset), "accuracy": correct / max(1, len(subset))})
    for variant in variants:
        for mask in ["current", "union", "final"]:
            for span in ["key", "label", "record"]:
                for metric in ["any", "all"]:
                    subset = [
                        row
                        for row in coverage_overall_rows
                        if row["variant"] == variant and row["mask"] == mask and row["span"] == span and row["metric"] == metric
                    ]
                    hits = sum(int(row["hit_count"]) for row in subset)
                    queries = sum(int(row["query_count"]) for row in subset)
                    summary.append(
                        {
                            "variant": variant,
                            "mode": "qabs8cand5reuse",
                            "coverage_mask": mask,
                            "coverage_span": span,
                            "coverage_metric": metric,
                            "hit_count": hits,
                            "query_count": queries,
                            "coverage": hits / queries if queries else 0.0,
                        }
                    )
    (output_dir / "summary.json").write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary[:16], indent=2), flush=True)


if __name__ == "__main__":
    main()
