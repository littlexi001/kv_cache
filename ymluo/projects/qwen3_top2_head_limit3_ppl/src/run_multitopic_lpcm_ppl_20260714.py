from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_20newsgroups

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    pick_input_device,
    resolve_dtype,
)


TOPICS = {
    "computer": "comp.graphics",
    "sports": "rec.sport.baseball",
    "medicine": "sci.med",
    "space": "sci.space",
    "politics": "talk.politics.mideast",
    "religion": "soc.religion.christian",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal multi-topic PPL for the current 2K KV gather + original RoPE + LPCM method."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topics", default=",".join(TOPICS))
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--eval_chunk_tokens", type=int, default=64)
    parser.add_argument("--windows_per_topic", type=int, default=1)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--budget_tokens", type=int, default=2048)
    parser.add_argument("--page_tokens", type=int, default=16)
    parser.add_argument("--sink_tokens", type=int, default=32)
    parser.add_argument("--recent_tokens", type=int, default=64)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


def selector_config(args: argparse.Namespace) -> lb.Config:
    saved_argv = sys.argv
    try:
        sys.argv = [
            "run_controlled_public_kv_benchmark_v1.py",
            "--model_name_or_path",
            args.model_name_or_path,
            "--output_dir",
            str(args.output_dir),
            "--benchmarks",
            "",
            "--methods",
            "ours_page_gather",
            "--max_context_tokens",
            str(args.history_tokens),
            "--budget_tokens",
            str(args.budget_tokens),
            "--sink_tokens",
            str(args.sink_tokens),
            "--recent_tokens",
            str(args.recent_tokens),
            "--page_tokens",
            str(args.page_tokens),
            "--ours_scorer",
            "hybrid_late_mmr_multiscale_idf_flow",
            "--ours_multiscale_group_pages",
            "4",
            "--ours_multiscale_weight",
            "0.22",
            "--ours_flow_neighbor_radius",
            "1",
            "--ours_flow_neighbor_budget_fraction",
            "0.16",
            "--ours_flow_neighbor_min_score",
            "0.12",
            "--ours_flow_score_smooth_weight",
            "0.16",
            "--ours_flow_anchor_boost",
            "0.20",
            "--ours_bridge_budget_fraction",
            "0.16",
            "--ours_bridge_min_score",
            "0.0",
            "--ours_bridge_max_terms",
            "24",
            "--dtype",
            args.dtype,
            "--device",
            args.device,
            "--device_map",
            args.device_map,
            "--attn_implementation",
            args.attn_implementation,
            "--prompt_wrapper",
            "none",
            "--sparse_query_physical_mask",
            "--sparse_position_mode",
            "original",
        ]
        return lb.parse_args()
    finally:
        sys.argv = saved_argv


def topic_names(spec: str) -> list[str]:
    names = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = sorted(set(names) - set(TOPICS))
    if unknown:
        raise ValueError(f"Unknown topics: {unknown}; available={sorted(TOPICS)}")
    return names


def encode_topic_stream(
    tokenizer: Any,
    category: str,
    required_tokens: int,
    cache_dir: str,
    seed: int,
    repeat_documents: bool = False,
) -> list[int]:
    dataset = fetch_20newsgroups(
        subset="train",
        categories=[category],
        remove=("headers", "footers", "quotes"),
        data_home=cache_dir,
        shuffle=False,
    )
    documents = [text.strip() for text in dataset.data if len(text.strip()) >= 200]
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
        if not repeat_documents:
            break
        if not cycle_documents:
            break
        cycle += 1
    if len(stream) < required_tokens:
        raise RuntimeError(f"{category} has only {len(stream)} usable tokens, need {required_tokens}")
    return stream[:required_tokens]


def make_bundle(tokenizer: Any, remote_ids: list[int], page_tokens: int) -> tuple[lb.PromptBundle, list[lb.Page]]:
    pages: list[lb.Page] = []
    for page_id, start in enumerate(range(0, len(remote_ids), page_tokens)):
        end = min(len(remote_ids), start + page_tokens)
        text = tokenizer.decode(remote_ids[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        pages.append(lb.Page(page_id, text, start, end))
    bundle = lb.PromptBundle(
        input_ids=torch.tensor([remote_ids], dtype=torch.long),
        prefix_token_count=0,
        context_token_start=0,
        query_start=len(remote_ids),
        suffix_token_count=0,
        page_spans={page.page_id: (page.token_start, page.token_end) for page in pages},
    )
    return bundle, pages


@torch.inference_mode()
def run_tokens_all_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    position_start: int,
    input_device: torch.device,
    physical_causal_mask: bool = False,
) -> tuple[Any, torch.Tensor, float]:
    """Run one causal chunk and retain every next-token logit in the chunk."""
    ids = input_ids.to(input_device)
    if ids.shape[-1] == 0:
        raise ValueError("empty token segment")
    started = time.perf_counter()
    model_inputs: dict[str, Any] = {
        "input_ids": ids,
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "cache_position": torch.arange(position_start, position_start + ids.shape[-1], device=input_device),
    }
    if physical_causal_mask:
        query_length = int(ids.shape[-1])
        past_length = lb.cache_sequence_length(past_key_values)
        mask_dtype = next(model.parameters()).dtype
        min_value = torch.finfo(mask_dtype).min
        attention_mask = torch.zeros(
            (int(ids.shape[0]), 1, query_length, past_length + query_length),
            dtype=mask_dtype,
            device=input_device,
        )
        attention_mask[..., past_length:] = torch.triu(
            torch.full((query_length, query_length), min_value, dtype=mask_dtype, device=input_device),
            diagonal=1,
        )
        model_inputs["attention_mask"] = attention_mask
    outputs = lb.model_forward(model, model_inputs)
    return outputs.past_key_values, outputs.logits.detach(), time.perf_counter() - started


@torch.inference_mode()
def evaluate_target_ppl(
    model: torch.nn.Module,
    prefix_cache: Any,
    query_ids: list[int],
    target_ids: list[int],
    logical_query_start: int,
    input_device: torch.device,
    chunk_tokens: int,
    physical_mask: bool,
) -> tuple[float, float, float, int]:
    if not query_ids or not target_ids:
        raise ValueError("query_ids and target_ids must be non-empty")
    query = torch.tensor([query_ids], dtype=torch.long, device=input_device)
    cache, previous_logits, query_seconds = lb.run_token_segment(
        model,
        query,
        prefix_cache,
        logical_query_start,
        input_device,
        chunk_tokens,
        physical_mask,
    )
    loss_sum = -float(F.log_softmax(previous_logits.float(), dim=-1)[0, int(target_ids[0])].item())
    token_count = 1
    eval_seconds = 0.0
    consumed = 0
    while consumed < len(target_ids) - 1:
        current = target_ids[consumed : min(len(target_ids) - 1, consumed + chunk_tokens)]
        labels = target_ids[consumed + 1 : consumed + 1 + len(current)]
        current_tensor = torch.tensor([current], dtype=torch.long, device=input_device)
        cache, logits, seconds = run_tokens_all_logits(
            model,
            current_tensor,
            cache,
            logical_query_start + len(query_ids) + consumed,
            input_device,
            physical_mask,
        )
        label_tensor = torch.tensor(labels, dtype=torch.long, device=logits.device)
        losses = F.cross_entropy(logits[0].float(), label_tensor, reduction="sum")
        loss_sum += float(losses.item())
        token_count += len(current)
        eval_seconds += seconds
        consumed += len(current)
    nll = loss_sum / max(1, token_count)
    return nll, math.exp(min(20.0, nll)), query_seconds + eval_seconds, token_count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = {(str(row["topic"]), int(row["window"])): row for row in rows if row["method"] == "full_kv"}
    output: list[dict[str, Any]] = []
    for method in ["full_kv", "sink_recent_2k", "ours_2k_lpcm"]:
        subset = [row for row in rows if row["method"] == method]
        tokens = sum(int(row["eval_token_count"]) for row in subset)
        weighted_nll = sum(float(row["nll"]) * int(row["eval_token_count"]) for row in subset) / max(1, tokens)
        full_nll = sum(
            float(full[(str(row["topic"]), int(row["window"]))]["nll"]) * int(row["eval_token_count"])
            for row in subset
        ) / max(1, tokens)
        output.append(
            {
                "method": method,
                "cases": len(subset),
                "tokens": tokens,
                "nll": weighted_nll,
                "ppl": math.exp(min(20.0, weighted_nll)),
                "delta_nll_vs_full": weighted_nll - full_nll,
                "ppl_ratio_vs_full": math.exp(min(20.0, weighted_nll - full_nll)),
                "mean_kv_ratio_after_query": sum(float(row["kv_ratio_after_query"]) for row in subset)
                / max(1, len(subset)),
                "mean_selector_seconds": sum(float(row["selector_seconds"]) for row in subset)
                / max(1, len(subset)),
                "mean_eval_seconds": sum(float(row["eval_seconds"]) for row in subset) / max(1, len(subset)),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.eval_chunk_tokens <= 0:
        raise ValueError("eval_chunk_tokens must be positive")
    if args.query_tokens <= 0 or args.query_tokens >= args.history_tokens:
        raise ValueError("query_tokens must be in (0, history_tokens)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    config = selector_config(args)
    lb.install_llama_layerwise_attention_mask_patch()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
    input_device = pick_input_device(model, device)

    selected_topics = topic_names(args.topics)
    required_tokens = (
        (args.windows_per_topic - 1) * args.window_stride_tokens + args.history_tokens + args.eval_tokens
    )
    rows: list[dict[str, Any]] = []
    for topic in selected_topics:
        stream = encode_topic_stream(
            tokenizer,
            TOPICS[topic],
            required_tokens,
            args.dataset_cache_dir,
            args.seed + selected_topics.index(topic),
        )
        for window in range(args.windows_per_topic):
            start = window * args.window_stride_tokens
            history = stream[start : start + args.history_tokens]
            target_ids = stream[start + args.history_tokens : start + args.history_tokens + args.eval_tokens]
            remote_ids = history[: -args.query_tokens]
            query_ids = history[-args.query_tokens :]
            query_text = tokenizer.decode(query_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            bundle, pages = make_bundle(tokenizer, remote_ids, args.page_tokens)
            example = lb.Example(
                benchmark="multitopic_ppl",
                task=topic,
                sample_id=f"{topic}_{window}",
                context="",
                query=query_text,
                answers=[],
                prefix_template="",
                suffix_template="",
                metric="qa_f1",
                max_new_tokens=0,
                length=len(history),
                all_classes=[],
            )
            print(
                f"[case] {topic}/{window} history={len(history)} remote={len(remote_ids)} "
                f"query={len(query_ids)} eval={len(target_ids)} pages={len(pages)}",
                flush=True,
            )
            full_cache, prefill_seconds = lb.prefill_prefix(
                model, bundle, input_device, config.prefill_chunk_tokens
            )

            selector_started = time.perf_counter()
            selector_extra = {"model": model, "tokenizer": tokenizer}
            ours_keep = lb.keep_ours_page(bundle, example, pages, config, selector_extra)
            ours_selector_seconds = time.perf_counter() - selector_started
            recent_config = replace(
                config,
                sink_tokens=args.sink_tokens,
                recent_tokens=max(0, args.budget_tokens - args.sink_tokens),
            )
            recent_keep = lb.keep_streaming(bundle, example, pages, recent_config, None)

            method_specs = [
                ("ours_2k_lpcm", ours_keep, ours_selector_seconds, True),
                ("sink_recent_2k", recent_keep, 0.0, True),
                ("full_kv", list(range(len(remote_ids))), 0.0, False),
            ]
            for method, keep_indices, selector_seconds, physical_mask in method_specs:
                gather_started = time.perf_counter()
                cache = full_cache if method == "full_kv" else lb.gather_past_key_values(full_cache, keep_indices)
                gather_seconds = 0.0 if method == "full_kv" else time.perf_counter() - gather_started
                nll, ppl, eval_seconds, eval_count = evaluate_target_ppl(
                    model,
                    cache,
                    query_ids,
                    target_ids,
                    len(remote_ids),
                    input_device,
                    args.eval_chunk_tokens,
                    physical_mask,
                )
                physical_tokens = len(keep_indices) + len(query_ids)
                row = {
                    "topic": topic,
                    "category": TOPICS[topic],
                    "window": window,
                    "method": method,
                    "nll": nll,
                    "ppl": ppl,
                    "raw_history_tokens": len(history),
                    "kept_remote_tokens": len(keep_indices),
                    "query_tokens": len(query_ids),
                    "physical_tokens_after_query": physical_tokens,
                    "kv_ratio_after_query": physical_tokens / max(1, len(history)),
                    "eval_token_count": eval_count,
                    "selector_seconds": selector_seconds,
                    "gather_seconds": gather_seconds,
                    "prefill_seconds_shared": prefill_seconds,
                    "eval_seconds": eval_seconds,
                    "online_seconds": selector_seconds + gather_seconds + eval_seconds,
                    "selected_pages": len(lb.selected_page_ids(bundle, keep_indices)),
                }
                rows.append(row)
                print(
                    f"  {method}: ppl={ppl:.3f} nll={nll:.4f} "
                    f"kv={100.0 * row['kv_ratio_after_query']:.2f}% online={row['online_seconds']:.3f}s",
                    flush=True,
                )
                del cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del full_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_csv(args.output_dir / "ppl_results.csv", rows)
            write_csv(args.output_dir / "ppl_summary.csv", aggregate(rows))

    summary = aggregate(rows)
    write_csv(args.output_dir / "ppl_results.csv", rows)
    write_csv(args.output_dir / "ppl_summary.csv", summary)
    (args.output_dir / "ppl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
