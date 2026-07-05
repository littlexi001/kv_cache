from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_causal_page_influence_predictor_v6 as causal_v6  # noqa: E402
import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    model_forward,
    pick_input_device,
    resolve_dtype,
)


@dataclass(frozen=True)
class PplCase:
    name: str
    path: str


def parse_cases(spec: str) -> list[PplCase]:
    cases: list[PplCase] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, path = item.split(":", 1)
        else:
            path = item
            name = Path(path).stem
        cases.append(PplCase(name.strip(), path.strip()))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPL test for causal-ridge page routing with prefill-time sparse text path."
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/qwen/LlaMa-3.1-8B")
    parser.add_argument("--predictor_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--text_cases",
        default=(
            "war:data/war_and_peace_pg2600.txt,"
            "monte:data/count_monte_cristo_pg1184.txt"
        ),
    )
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--eval_chunk_size", type=int, default=64)
    parser.add_argument("--budget_tokens", type=int, default=512)
    parser.add_argument("--sink_tokens", type=int, default=64)
    parser.add_argument("--recent_tokens", type=int, default=256)
    parser.add_argument("--page_tokens", type=int, default=256)
    parser.add_argument("--query_window_tokens", type=int, default=256)
    parser.add_argument("--semantic_embed_max_tokens", type=int, default=192)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


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


def make_config(args: argparse.Namespace, output_dir: str) -> lb.Config:
    return lb.Config(
        model_name_or_path=args.model_name_or_path,
        output_dir=output_dir,
        benchmarks="ppl",
        longbench_tasks="",
        ruler_tasks="",
        max_samples_per_task=1,
        max_context_tokens=args.prefill_tokens,
        max_new_tokens_override=0,
        seed=2026070307,
        methods="",
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
        prompt_wrapper="none",
        longbench_zip_path="",
        hf_cache_dir="/home/fdong/ymluo/hf_cache",
        lm_eval_path="/home/fdong/lm-evaluation-harness",
        ruler_lengths="",
        log_every=1,
    )


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]


def decode_ids(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def make_ppl_bundle(tokenizer: Any, prefix_ids: list[int], page_tokens: int) -> tuple[lb.PromptBundle, list[lb.Page]]:
    pages: list[lb.Page] = []
    for page_id, start in enumerate(range(0, len(prefix_ids), page_tokens)):
        end = min(len(prefix_ids), start + page_tokens)
        pages.append(lb.Page(page_id, decode_ids(tokenizer, prefix_ids[start:end]), start, end))
    page_spans = {page.page_id: (page.token_start, page.token_end) for page in pages}
    bundle = lb.PromptBundle(
        input_ids=torch.tensor([prefix_ids], dtype=torch.long),
        prefix_token_count=0,
        context_token_start=0,
        query_start=len(prefix_ids),
        suffix_token_count=0,
        page_spans=page_spans,
    )
    return bundle, pages


def selected_prefix_ids(bundle: lb.PromptBundle, keep_indices: list[int]) -> list[int]:
    original = bundle.input_ids[0].detach().cpu().tolist()
    return [
        original[idx]
        for idx in sorted(keep_indices)
        if bundle.context_token_start <= idx < bundle.query_start
    ]


def selected_prefix_ids_and_positions(
    bundle: lb.PromptBundle,
    keep_indices: list[int],
) -> tuple[list[int], list[int]]:
    original = bundle.input_ids[0].detach().cpu().tolist()
    positions = [
        idx
        for idx in sorted(keep_indices)
        if bundle.context_token_start <= idx < bundle.query_start
    ]
    return [original[idx] for idx in positions], positions


@torch.inference_mode()
def eval_ppl_with_prefix(
    model: torch.nn.Module,
    prefix_ids: list[int],
    target_ids: list[int],
    input_device: torch.device,
    eval_chunk_size: int,
    prefix_position_ids: list[int] | None = None,
    target_position_start: int | None = None,
) -> tuple[float, float, float, float]:
    if not prefix_ids:
        raise ValueError("prefix_ids must not be empty")
    if not target_ids:
        raise ValueError("target_ids must not be empty")
    prefix = torch.tensor([prefix_ids], dtype=torch.long, device=input_device)
    if prefix_position_ids is not None and len(prefix_position_ids) != len(prefix_ids):
        raise ValueError(
            f"prefix_position_ids length mismatch: positions={len(prefix_position_ids)} ids={len(prefix_ids)}"
        )
    if prefix_position_ids is None:
        prefill_position_ids = None
    else:
        prefill_position_ids = torch.tensor([prefix_position_ids], dtype=torch.long, device=input_device)
    started = time.perf_counter()
    kwargs = {
        "input_ids": prefix,
        "use_cache": True,
        "return_dict": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "cache_position": torch.arange(len(prefix_ids), device=input_device),
    }
    if prefill_position_ids is not None:
        kwargs["position_ids"] = prefill_position_ids
    outputs = model_forward(model, kwargs)
    prefill_seconds = time.perf_counter() - started
    cache = outputs.past_key_values
    prev_logits = outputs.logits[:, -1, :].detach()

    loss_sum = 0.0
    token_count = 0
    eval_started = time.perf_counter()
    log_probs = F.log_softmax(prev_logits.float(), dim=-1)
    loss_sum -= float(log_probs[0, int(target_ids[0])].item())
    token_count += 1

    consumed = 0
    while consumed < len(target_ids) - 1:
        chunk_ids = target_ids[consumed : min(len(target_ids) - 1, consumed + eval_chunk_size)]
        labels = target_ids[consumed + 1 : consumed + 1 + len(chunk_ids)]
        chunk = torch.tensor([chunk_ids], dtype=torch.long, device=input_device)
        compact_position_start = len(prefix_ids) + consumed
        kwargs = {
            "input_ids": chunk,
            "past_key_values": cache,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(
                compact_position_start,
                compact_position_start + len(chunk_ids),
                device=input_device,
            ),
        }
        if target_position_start is not None:
            kwargs["position_ids"] = torch.arange(
                target_position_start + consumed,
                target_position_start + consumed + len(chunk_ids),
                device=input_device,
            ).view(1, -1)
        outputs = model_forward(model, kwargs)
        cache = outputs.past_key_values
        logits = outputs.logits.float()
        label_tensor = torch.tensor(labels, dtype=torch.long, device=logits.device)
        losses = F.cross_entropy(logits[0, :, :], label_tensor, reduction="sum")
        loss_sum += float(losses.item())
        token_count += len(labels)
        consumed += len(chunk_ids)
    eval_seconds = time.perf_counter() - eval_started
    nll = loss_sum / max(1, token_count)
    ppl = math.exp(min(20.0, nll))
    return nll, ppl, prefill_seconds, eval_seconds


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    summary = []
    for method, subset in sorted(by_method.items()):
        n = max(1, len(subset))
        summary.append(
            {
                "method": method,
                "cases": len(subset),
                "mean_nll": sum(float(row["nll"]) for row in subset) / n,
                "mean_ppl": sum(float(row["ppl"]) for row in subset) / n,
                "mean_total_seconds": sum(float(row["total_seconds"]) for row in subset) / n,
                "mean_prefill_seconds": sum(float(row["prefill_seconds"]) for row in subset) / n,
                "mean_eval_seconds": sum(float(row["eval_seconds"]) for row in subset) / n,
                "mean_selector_seconds": sum(float(row["selector_seconds"]) for row in subset) / n,
                "mean_prefix_tokens": sum(int(row["prefix_tokens"]) for row in subset) / n,
                "mean_keep_fraction": sum(float(row["keep_fraction"]) for row in subset) / n,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    predictor = json.loads(Path(args.predictor_path).read_text(encoding="utf-8"))
    config = make_config(args, str(output_dir))

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

    rows: list[dict[str, Any]] = []
    for case in parse_cases(args.text_cases):
        text = Path(case.path).read_text(encoding="utf-8", errors="replace")[: args.max_chars]
        ids = encode_text(tokenizer, text)
        needed = args.prefill_tokens + args.eval_tokens
        if len(ids) < needed:
            print(f"[skip] {case.name}: only {len(ids)} tokens, need {needed}", flush=True)
            continue
        prefix_ids = ids[: args.prefill_tokens]
        target_ids = ids[args.prefill_tokens : args.prefill_tokens + args.eval_tokens]
        bundle, pages = make_ppl_bundle(tokenizer, prefix_ids, args.page_tokens)
        query_ids = prefix_ids[-min(args.query_window_tokens, len(prefix_ids)) :]
        example = lb.Example(
            benchmark="ppl",
            task=case.name,
            sample_id=case.name,
            context="",
            query=decode_ids(tokenizer, query_ids),
            answers=[],
            prefix_template="",
            suffix_template="",
            metric="qa_f1",
            max_new_tokens=0,
            length=len(ids),
        )
        print(f"[case] {case.name} prefix={len(prefix_ids)} eval={len(target_ids)} pages={len(pages)}", flush=True)

        feature_started = time.perf_counter()
        page_rows = causal_v6.feature_rows_for_pages(model, tokenizer, example, bundle, pages, config)
        causal_v6.predict_rows(page_rows, predictor)
        feature_seconds = time.perf_counter() - feature_started

        mode_to_keep: dict[str, list[int]] = {
            "full_prefill": list(range(bundle.query_start)),
            "recent_sparse_prefill": lb.keep_streaming(bundle, example, pages, config, None),
        }
        heuristic_scores = {int(row["page_id"]): float(row["heuristic"]) for row in page_rows}
        causal_scores = {int(row["page_id"]): float(row["predicted_delta_nll"] or 0.0) for row in page_rows}
        mode_to_keep["heuristic_sparse_prefill"] = causal_v6.keep_with_page_scores(
            bundle, pages, config, heuristic_scores
        )
        mode_to_keep["causal_ridge_sparse_prefill"] = causal_v6.keep_with_page_scores(
            bundle, pages, config, causal_scores
        )
        mode_to_keep["heuristic_rangepos_sparse_prefill"] = mode_to_keep["heuristic_sparse_prefill"]
        mode_to_keep["causal_ridge_rangepos_sparse_prefill"] = mode_to_keep["causal_ridge_sparse_prefill"]

        for method, keep_indices in mode_to_keep.items():
            prefix_position_ids = None
            target_position_start = None
            if method == "full_prefill":
                selected_ids = prefix_ids
            elif method.endswith("_rangepos_sparse_prefill"):
                selected_ids, prefix_position_ids = selected_prefix_ids_and_positions(bundle, keep_indices)
                target_position_start = args.prefill_tokens
            else:
                selected_ids = selected_prefix_ids(bundle, keep_indices)
            selector_seconds = feature_seconds if method in {"heuristic_sparse_prefill", "causal_ridge_sparse_prefill"} else 0.0
            if method in {"heuristic_rangepos_sparse_prefill", "causal_ridge_rangepos_sparse_prefill"}:
                selector_seconds = feature_seconds
            nll, ppl, prefill_seconds, eval_seconds = eval_ppl_with_prefix(
                model,
                selected_ids,
                target_ids,
                input_device,
                args.eval_chunk_size,
                prefix_position_ids=prefix_position_ids,
                target_position_start=target_position_start,
            )
            row = {
                "case": case.name,
                "method": method,
                "nll": nll,
                "ppl": ppl,
                "prefill_seconds": prefill_seconds,
                "eval_seconds": eval_seconds,
                "selector_seconds": selector_seconds,
                "total_seconds": selector_seconds + prefill_seconds + eval_seconds,
                "raw_prefix_tokens": len(prefix_ids),
                "prefix_tokens": len(selected_ids),
                "eval_tokens": len(target_ids),
                "keep_fraction": len(selected_ids) / max(1, len(prefix_ids)),
                "page_count": len(pages),
            }
            rows.append(row)
            print(
                f"  {method}: ppl={ppl:.3f} nll={nll:.4f} "
                f"prefix={len(selected_ids)}/{len(prefix_ids)} total={row['total_seconds']:.3f}s",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize(rows)
    write_csv(output_dir / "ppl_results.csv", rows)
    write_csv(output_dir / "ppl_summary.csv", summary)
    (output_dir / "ppl_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cases": len(parse_cases(args.text_cases)), "rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
