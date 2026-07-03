from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    clone_past_key_values,
    model_forward,
    pick_input_device,
    resolve_dtype,
)


FEATURE_NAMES = [
    "lexical",
    "entity",
    "structural",
    "coverage",
    "semantic",
    "late",
    "position",
    "length",
    "is_sink",
    "is_recent",
    "heuristic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-distilled causal page influence predictor. It labels pages by "
            "delta NLL on full-context teacher tokens, trains a lightweight ridge "
            "page scorer, then evaluates sparse KV page gather."
        )
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/qwen/LlaMa-3.1-8B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--longbench_tasks", default="qasper,hotpotqa,passage_retrieval_en")
    parser.add_argument("--max_samples_per_task", type=int, default=1)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens_override", type=int, default=32)
    parser.add_argument("--target_tokens", type=int, default=16)
    parser.add_argument("--max_label_pages", type=int, default=10)
    parser.add_argument("--budget_tokens", type=int, default=512)
    parser.add_argument("--sink_tokens", type=int, default=64)
    parser.add_argument("--recent_tokens", type=int, default=256)
    parser.add_argument("--page_tokens", type=int, default=256)
    parser.add_argument("--semantic_embed_max_tokens", type=int, default=192)
    parser.add_argument("--ridge_lambda", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2026070306)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--prompt_wrapper", choices=["none", "llama3"], default="llama3")
    parser.add_argument("--longbench_zip_path", default="")
    parser.add_argument("--hf_cache_dir", default="/home/fdong/ymluo/hf_cache")
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def make_lb_config(args: argparse.Namespace) -> lb.Config:
    return lb.Config(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        benchmarks="longbench",
        longbench_tasks=args.longbench_tasks,
        ruler_tasks="",
        max_samples_per_task=args.max_samples_per_task,
        max_context_tokens=args.max_context_tokens,
        max_new_tokens_override=args.max_new_tokens_override,
        seed=args.seed,
        methods="full_kv,ours_page_gather",
        budget_tokens=args.budget_tokens,
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
        page_tokens=args.page_tokens,
        obs_window_tokens=64,
        snap_pool_kernel=7,
        ours_scorer="hybrid_late_mmr",
        semantic_embed_max_tokens=args.semantic_embed_max_tokens,
        semantic_weight=0.55,
        lexical_weight=0.25,
        entity_weight=0.15,
        structural_weight=0.05,
        coverage_weight=0.15,
        ours_mmr_lambda=0.82,
        anchor_pages_per_key=2,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        prompt_wrapper=args.prompt_wrapper,
        longbench_zip_path=args.longbench_zip_path,
        hf_cache_dir=args.hf_cache_dir,
        lm_eval_path="/home/fdong/lm-evaluation-harness",
        ruler_lengths="",
        log_every=args.log_every,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(values: list[float]) -> list[float]:
    return lb.normalize_values(values)


def feature_rows_for_pages(
    model: torch.nn.Module,
    tokenizer: Any,
    example: lb.Example,
    bundle: lb.PromptBundle,
    pages: list[lb.Page],
    config: lb.Config,
) -> list[dict[str, Any]]:
    query = example.query or example.suffix_template
    q_words = lb.word_counter(query)
    q_entities = lb.extract_entities(query)

    lexical_raw: list[float] = []
    entity_raw: list[float] = []
    structural_raw: list[float] = []
    coverage_raw: list[float] = []
    length_raw: list[float] = []
    for page in pages:
        p_words = lb.word_counter(page.text)
        p_entities = lb.extract_entities(page.text)
        lexical_raw.append(float(sum(min(count, p_words.get(word, 0)) for word, count in q_words.items())))
        entity_raw.append(float(len(q_entities & p_entities)))
        structural_raw.append(float(page.text.count(":") + page.text.count("|") + len(page.text.splitlines())))
        if lb.task_is_global(example):
            pos = page.page_id / max(1, len(pages) - 1)
            coverage_raw.append(float(max(1.0 - abs(pos - anchor) / 0.22 for anchor in (0.08, 0.50, 0.92))))
        else:
            coverage_raw.append(0.0)
        length_raw.append(float(max(0, page.token_end - page.token_start)))

    lexical = normalize(lexical_raw)
    entity = normalize(entity_raw)
    structural = normalize(structural_raw)
    coverage = normalize(coverage_raw)
    length = [min(1.0, value / max(1, config.page_tokens)) for value in length_raw]

    with torch.inference_mode():
        query_vec = lb.static_text_embeddings(model, tokenizer, [query], config.semantic_embed_max_tokens)[0]
        page_vecs = lb.static_text_embeddings(
            model,
            tokenizer,
            [page.text for page in pages],
            config.semantic_embed_max_tokens,
        )
        semantic = normalize(torch.matmul(page_vecs, query_vec).detach().float().cpu().tolist())
        late = normalize(
            lb.late_interaction_scores(
                model,
                tokenizer,
                query,
                [page.text for page in pages],
                config.semantic_embed_max_tokens,
            )
        )

    rows: list[dict[str, Any]] = []
    for idx, page in enumerate(pages):
        position = page.page_id / max(1, len(pages) - 1)
        is_sink = float(page.token_start < bundle.context_token_start + config.sink_tokens)
        is_recent = float(page.token_end > bundle.query_start - config.recent_tokens)
        heuristic = (
            0.55 * late[idx]
            + 0.25 * lexical[idx]
            + 0.15 * entity[idx]
            + 0.05 * structural[idx]
            + 0.15 * coverage[idx]
        )
        rows.append(
            {
                "benchmark": example.benchmark,
                "task": example.task,
                "sample_id": example.sample_id,
                "page_id": page.page_id,
                "token_start": page.token_start,
                "token_end": page.token_end,
                "token_len": max(0, page.token_end - page.token_start),
                "lexical": lexical[idx],
                "entity": entity[idx],
                "structural": structural[idx],
                "coverage": coverage[idx],
                "semantic": semantic[idx],
                "late": late[idx],
                "position": position,
                "length": length[idx],
                "is_sink": is_sink,
                "is_recent": is_recent,
                "heuristic": heuristic,
                "label_delta_nll": "",
                "label_base_nll": "",
                "label_page_nll": "",
                "label_is_measured": 0,
                "predicted_delta_nll": "",
            }
        )
    return rows


def choose_label_page_ids(rows: list[dict[str, Any]], max_label_pages: int, seed: int) -> list[int]:
    if max_label_pages <= 0 or len(rows) <= max_label_pages:
        return [int(row["page_id"]) for row in rows]
    by_score = sorted(rows, key=lambda row: float(row["heuristic"]), reverse=True)
    chosen = {int(row["page_id"]) for row in by_score[: max(1, max_label_pages // 2)]}
    rng = random.Random(seed)
    remaining = [int(row["page_id"]) for row in rows if int(row["page_id"]) not in chosen]
    if remaining:
        stride = max(1, len(remaining) // max(1, max_label_pages - len(chosen)))
        for page_id in remaining[::stride]:
            chosen.add(page_id)
            if len(chosen) >= max_label_pages:
                break
    if len(chosen) < max_label_pages:
        rng.shuffle(remaining)
        for page_id in remaining:
            chosen.add(page_id)
            if len(chosen) >= max_label_pages:
                break
    return sorted(chosen)


def keep_with_page_scores(
    bundle: lb.PromptBundle,
    pages: list[lb.Page],
    config: lb.Config,
    page_scores: dict[int, float],
) -> list[int]:
    keep = lb.base_context_keep_indices(bundle, config)
    selected_context = sum(1 for idx in keep if bundle.context_token_start <= idx < bundle.query_start)
    remaining = max(0, config.budget_tokens - selected_context)
    candidates = sorted(pages, key=lambda page: (page_scores.get(page.page_id, 0.0), -page.page_id), reverse=True)
    for page in candidates:
        if remaining <= 0:
            break
        added = lb.add_page_to_keep(keep, bundle, page, remaining)
        remaining -= added
    return lb.fit_context_budget(keep, bundle, config.budget_tokens)


@torch.inference_mode()
def target_nll(
    model: torch.nn.Module,
    bundle: lb.PromptBundle,
    prefix_cache: Any,
    target_ids: list[int],
    input_device: torch.device,
) -> float:
    if not target_ids:
        return float("nan")
    suffix_ids = bundle.input_ids[:, bundle.query_start :].to(input_device)
    query_cache, prev_logits, _ = lb.run_tokens(model, suffix_ids, prefix_cache, bundle.query_start, input_device)
    total = 0.0
    for offset, token_id in enumerate(target_ids):
        log_probs = F.log_softmax(prev_logits.float(), dim=-1)
        total -= float(log_probs[0, int(token_id)].item())
        if offset == len(target_ids) - 1:
            break
        token = torch.tensor([[int(token_id)]], dtype=torch.long, device=input_device)
        outputs = model_forward(
            model,
            {
                "input_ids": token,
                "past_key_values": query_cache,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "cache_position": torch.tensor(
                    [bundle.query_start + bundle.suffix_token_count + offset],
                    device=input_device,
                ),
            },
        )
        query_cache = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
    return total / max(1, len(target_ids))


def label_pages_for_example(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    example: lb.Example,
    bundle: lb.PromptBundle,
    pages: list[lb.Page],
    full_prefix_cache: Any,
    config: lb.Config,
    args: argparse.Namespace,
    feature_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction, generated_ids, query_seconds, decode_seconds = lb.generate_with_cache(
        model,
        tokenizer,
        bundle,
        clone_past_key_values(full_prefix_cache),
        example.max_new_tokens,
        input_device,
    )
    target_ids = generated_ids[: args.target_tokens]
    if not target_ids and example.answers:
        target_ids = tokenizer(example.answers[0], add_special_tokens=False)["input_ids"][: args.target_tokens]
    teacher_score = lb.score_prediction(example.metric, prediction, example.answers)

    base_keep = lb.keep_streaming(bundle, example, pages, config, None)
    base_cache = lb.gather_past_key_values(clone_past_key_values(full_prefix_cache), base_keep)
    base_nll = target_nll(model, bundle, base_cache, target_ids, input_device)

    rows_by_page = {int(row["page_id"]): row for row in feature_rows}
    sample_seed = args.seed + int(hashlib.sha1(str(example.sample_id).encode("utf-8")).hexdigest()[:8], 16)
    label_page_ids = choose_label_page_ids(feature_rows, args.max_label_pages, sample_seed)
    for page_id in label_page_ids:
        page = pages[page_id]
        keep = set(base_keep)
        lb.add_page_to_keep(keep, bundle, page, config.budget_tokens)
        keep_indices = lb.fit_context_budget(keep, bundle, config.budget_tokens)
        page_cache = lb.gather_past_key_values(clone_past_key_values(full_prefix_cache), keep_indices)
        page_nll = target_nll(model, bundle, page_cache, target_ids, input_device)
        row = rows_by_page[page_id]
        row["label_base_nll"] = base_nll
        row["label_page_nll"] = page_nll
        row["label_delta_nll"] = base_nll - page_nll
        row["label_is_measured"] = 1

    teacher_row = {
        "benchmark": example.benchmark,
        "task": example.task,
        "sample_id": example.sample_id,
        "teacher_prediction": prediction.replace("\n", "\\n")[:500],
        "teacher_score": teacher_score,
        "teacher_generated_tokens": len(generated_ids),
        "teacher_query_seconds": query_seconds,
        "teacher_decode_seconds": decode_seconds,
        "target_tokens": len(target_ids),
        "base_target_nll": base_nll,
        "measured_pages": len(label_page_ids),
        "page_count": len(pages),
    }
    return feature_rows, teacher_row


def train_ridge(label_rows: list[dict[str, Any]], ridge_lambda: float) -> dict[str, Any]:
    measured = [row for row in label_rows if int(row.get("label_is_measured", 0)) == 1]
    if not measured:
        raise ValueError("No measured page labels; increase --max_label_pages")
    x = torch.tensor([[float(row[name]) for name in FEATURE_NAMES] for row in measured], dtype=torch.float64)
    y = torch.tensor([float(row["label_delta_nll"]) for row in measured], dtype=torch.float64).view(-1, 1)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    xz = (x - mean) / std
    xb = torch.cat([torch.ones((xz.shape[0], 1), dtype=torch.float64), xz], dim=1)
    reg = torch.eye(xb.shape[1], dtype=torch.float64) * ridge_lambda
    reg[0, 0] = 0.0
    weights = torch.linalg.solve(xb.T @ xb + reg, xb.T @ y).view(-1)
    pred = xb @ weights.view(-1, 1)
    ss_res = float(((pred - y) ** 2).sum().item())
    ss_tot = float(((y - y.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {
        "feature_names": FEATURE_NAMES,
        "mean": mean.view(-1).tolist(),
        "std": std.view(-1).tolist(),
        "weights": weights.tolist(),
        "ridge_lambda": ridge_lambda,
        "train_pages": len(measured),
        "train_r2_in_sample": r2,
    }


def predict_rows(rows: list[dict[str, Any]], model_info: dict[str, Any]) -> None:
    mean = torch.tensor(model_info["mean"], dtype=torch.float64)
    std = torch.tensor(model_info["std"], dtype=torch.float64)
    weights = torch.tensor(model_info["weights"], dtype=torch.float64)
    for row in rows:
        x = torch.tensor([float(row[name]) for name in FEATURE_NAMES], dtype=torch.float64)
        xb = torch.cat([torch.ones(1, dtype=torch.float64), (x - mean) / std])
        row["predicted_delta_nll"] = float(torch.dot(xb, weights).item())


def evaluate_sparse_method(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    example: lb.Example,
    bundle: lb.PromptBundle,
    pages: list[lb.Page],
    full_prefix_cache: Any,
    prefill_seconds: float,
    config: lb.Config,
    method: str,
    page_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if method == "full_kv":
        keep_indices = lb.keep_full(bundle, example, pages, config, None)
    elif method == "heuristic_page_gather":
        keep_indices = lb.keep_ours_page(bundle, example, pages, config, {"model": model, "tokenizer": tokenizer})
    elif method == "causal_ridge_page_gather":
        scores = {int(row["page_id"]): float(row["predicted_delta_nll"] or 0.0) for row in page_rows}
        keep_indices = keep_with_page_scores(bundle, pages, config, scores)
    elif method == "causal_label_oracle":
        scores = {
            int(row["page_id"]): float(row["label_delta_nll"] or 0.0)
            for row in page_rows
            if int(row.get("label_is_measured", 0)) == 1
        }
        keep_indices = keep_with_page_scores(bundle, pages, config, scores)
    else:
        raise ValueError(method)

    gather_started = time.perf_counter()
    sparse_cache = full_prefix_cache if method == "full_kv" else lb.gather_past_key_values(full_prefix_cache, keep_indices)
    gather_seconds = 0.0 if method == "full_kv" else time.perf_counter() - gather_started
    prediction, generated_ids, query_seconds, decode_seconds = lb.generate_with_cache(
        model,
        tokenizer,
        bundle,
        sparse_cache,
        example.max_new_tokens,
        input_device,
    )
    score = lb.score_prediction(example.metric, prediction, example.answers)
    context_kept = sum(1 for idx in keep_indices if bundle.context_token_start <= idx < bundle.query_start)
    return {
        "benchmark": example.benchmark,
        "task": example.task,
        "sample_id": example.sample_id,
        "method": method,
        "metric": example.metric,
        "score": score,
        "prediction": prediction.replace("\n", "\\n")[:500],
        "answers": json.dumps(example.answers, ensure_ascii=False),
        "generated_tokens": len(generated_ids),
        "prefill_seconds": prefill_seconds,
        "kv_gather_seconds": gather_seconds,
        "query_seconds": query_seconds,
        "decode_seconds": decode_seconds,
        "online_seconds": gather_seconds + query_seconds + decode_seconds,
        "total_seconds": prefill_seconds + gather_seconds + query_seconds + decode_seconds,
        "raw_prefix_tokens": bundle.query_start,
        "kept_prefix_tokens": len(keep_indices),
        "kept_context_tokens": context_kept,
        "keep_fraction": len(keep_indices) / max(1, bundle.query_start),
        "selected_pages": ",".join(str(page_id) for page_id in lb.selected_page_ids(bundle, keep_indices)),
        "page_count": len(pages),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        keys = [
            (str(row["benchmark"]), str(row["task"]), str(row["method"])),
            (str(row["benchmark"]), "ALL", str(row["method"])),
            ("ALL", "ALL", str(row["method"])),
        ]
        for key in keys:
            grouped.setdefault(key, []).append(row)
    summary = []
    for (benchmark, task, method), subset in sorted(grouped.items()):
        n = max(1, len(subset))
        summary.append(
            {
                "benchmark": benchmark,
                "task": task,
                "method": method,
                "samples": len(subset),
                "score": sum(float(row["score"]) for row in subset) / n,
                "mean_total_seconds": sum(float(row["total_seconds"]) for row in subset) / n,
                "mean_online_seconds": sum(float(row["online_seconds"]) for row in subset) / n,
                "mean_prefill_seconds": sum(float(row["prefill_seconds"]) for row in subset) / n,
                "mean_kv_gather_seconds": sum(float(row["kv_gather_seconds"]) for row in subset) / n,
                "mean_query_seconds": sum(float(row["query_seconds"]) for row in subset) / n,
                "mean_decode_seconds": sum(float(row["decode_seconds"]) for row in subset) / n,
                "mean_kept_prefix_tokens": sum(int(row["kept_prefix_tokens"]) for row in subset) / n,
                "mean_keep_fraction": sum(float(row["keep_fraction"]) for row in subset) / n,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    config = make_lb_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

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

    examples = lb.load_longbench_examples(config)
    sampled_ids = [
        {
            "benchmark": example.benchmark,
            "task": example.task,
            "sample_id": example.sample_id,
            "length": example.length,
        }
        for example in examples
    ]
    write_csv(output_dir / "sampled_ids.csv", sampled_ids)

    all_page_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx, example in enumerate(examples):
        bundle, pages, _, _, _ = lb.build_bundle(tokenizer, example, config)
        print(
            f"[label {idx + 1}/{len(examples)}] {example.task}/{example.sample_id} "
            f"prefix_tokens={bundle.query_start} pages={len(pages)}",
            flush=True,
        )
        feature_rows = feature_rows_for_pages(model, tokenizer, example, bundle, pages, config)
        full_prefix_cache, _ = lb.prefill_prefix(model, bundle, input_device)
        labeled_rows, teacher_row = label_pages_for_example(
            model,
            tokenizer,
            input_device,
            example,
            bundle,
            pages,
            full_prefix_cache,
            config,
            args,
            feature_rows,
        )
        all_page_rows.extend(labeled_rows)
        teacher_rows.append(teacher_row)
        del full_prefix_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictor = train_ridge(all_page_rows, args.ridge_lambda)
    predict_rows(all_page_rows, predictor)
    (output_dir / "causal_page_predictor.json").write_text(
        json.dumps(predictor, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "page_labels.csv", all_page_rows)
    write_csv(output_dir / "teacher_rows.csv", teacher_rows)

    page_rows_by_sample: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in all_page_rows:
        key = (str(row["benchmark"]), str(row["task"]), str(row["sample_id"]))
        page_rows_by_sample.setdefault(key, []).append(row)

    result_rows: list[dict[str, Any]] = []
    methods = ["full_kv", "heuristic_page_gather", "causal_ridge_page_gather", "causal_label_oracle"]
    for idx, example in enumerate(examples):
        bundle, pages, _, _, _ = lb.build_bundle(tokenizer, example, config)
        print(
            f"[eval {idx + 1}/{len(examples)}] {example.task}/{example.sample_id} "
            f"prefix_tokens={bundle.query_start} pages={len(pages)}",
            flush=True,
        )
        full_prefix_cache, prefill_seconds = lb.prefill_prefix(model, bundle, input_device)
        page_rows = page_rows_by_sample[(example.benchmark, example.task, str(example.sample_id))]
        for method in methods:
            row = evaluate_sparse_method(
                model,
                tokenizer,
                input_device,
                example,
                bundle,
                pages,
                clone_past_key_values(full_prefix_cache),
                prefill_seconds,
                config,
                method,
                page_rows,
            )
            result_rows.append(row)
            print(
                f"  {method}: score={row['score']:.3f} kept={row['kept_prefix_tokens']}/{row['raw_prefix_tokens']} "
                f"online={row['online_seconds']:.3f}s pred={row['prediction'][:80]}",
                flush=True,
            )
        del full_prefix_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = summarize(result_rows)
    write_csv(output_dir / "task_results.csv", result_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "elapsed_seconds": time.perf_counter() - started,
        "examples": len(examples),
        "methods": methods,
        "feature_names": FEATURE_NAMES,
        "labeling": "delta_nll_of_full_context_teacher_tokens_after_adding_one_page_to_sink_recent",
        "train_eval_note": "v6 smoke trains the ridge scorer on measured pages from the same sampled set; use held-out IDs for final claims.",
        "predictor": predictor,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
