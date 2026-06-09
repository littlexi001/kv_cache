from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-0.6B"
DEFAULT_DATA_PATH = (
    "ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl"
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Top-1% attention retrieval, token text dump, and category ablation."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", default="outputs/experiment2_top1_token_retrieval_ablation")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top_ratio", type=float, default=0.01)
    parser.add_argument("--query_last_tokens", type=int, default=16)
    parser.add_argument("--ablation_layer", default="last", help="last, all_union, or an integer layer id.")
    parser.add_argument("--max_context_chars", type=int, default=24000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--include_special_tokens", type=str2bool, default=False)
    parser.add_argument("--random_keep_trials", type=int, default=3)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def pick_input_device(model: torch.nn.Module, fallback_device: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def load_samples(path: Path, max_samples: int, max_context_chars: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            context = str(item.get("context", ""))
            if max_context_chars > 0:
                context = context[:max_context_chars]
            samples.append(
                {
                    "sample_id": item.get("sample_id", f"sample_{len(samples)}"),
                    "context": context,
                    "question": item.get("question", ""),
                    "answer": item.get("expected_answer", item.get("answer", "")),
                    "needle_text": item.get("needle_text", ""),
                }
            )
            if len(samples) >= max_samples:
                break
    return samples


def build_prompt(sample: dict[str, Any]) -> str:
    return (
        "Use the context to answer the question. Answer with the exact phrase when possible.\n\n"
        f"Context:\n{sample['context']}\n\n"
        f"Question: {sample['question']}\n"
        "Answer:"
    )


def token_offsets(tokenizer: Any, prompt: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return encoded.input_ids[0].tolist(), [(int(a), int(b)) for a, b in encoded.offset_mapping[0].tolist()]


def compact_piece(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t")


def classify_token(
    token_index: int,
    offsets: list[tuple[int, int]],
    prompt: str,
    sample: dict[str, Any],
    total_tokens: int,
) -> str:
    start, end = offsets[token_index]
    if token_index < 4:
        return "sink_prefix"
    if token_index >= max(0, total_tokens - 24):
        return "query_or_recent"
    piece = prompt[start:end].lower()
    answer = str(sample["answer"]).lower()
    needle = str(sample["needle_text"]).lower()
    question = str(sample["question"]).lower()
    if piece.strip() and piece in answer:
        return "answer_evidence"
    if piece.strip() and piece in needle:
        return "needle_context"
    if piece.strip() and piece in question:
        return "query_bridge"
    if any(ch.isdigit() for ch in piece):
        return "rare_or_number"
    if any(ch in piece for ch in ["#", "{", "}", "_", "@", "/", "\\", "="]):
        return "rare_or_symbol"
    if not piece.strip():
        return "whitespace"
    return "other_context"


@torch.inference_mode()
def answer_nll(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    answer: str,
    input_device: torch.device,
) -> dict[str, float]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(input_device)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    answer_start = prompt_ids.shape[1] - 1
    answer_logits = logits[:, answer_start : answer_start + answer_ids.shape[1], :]
    answer_labels = labels[:, answer_start : answer_start + answer_ids.shape[1]]
    losses = F.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        answer_labels.reshape(-1),
        reduction="none",
    )
    loss = float(losses.mean())
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 80.0))),
        "prompt_token_count": int(prompt_ids.shape[1]),
        "answer_token_count": int(answer_ids.shape[1]),
    }


@torch.inference_mode()
def collect_attention_top_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    top_ratio: float,
    query_last_tokens: int,
    input_device: torch.device,
) -> tuple[dict[int, list[tuple[int, float]]], int]:
    input_ids = input_ids.to(input_device)
    outputs = model(
        input_ids=input_ids,
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return attentions. Use --attn_implementation eager for Qwen models.")
    total_tokens = int(input_ids.shape[1])
    query_start = max(0, total_tokens - query_last_tokens)
    keep_count = max(1, math.ceil(top_ratio * total_tokens))
    layer_to_tokens: dict[int, list[tuple[int, float]]] = {}
    for layer_idx, attn in enumerate(attentions):
        # [batch, heads, query, key]
        score = attn[0, :, query_start:, :].float().mean(dim=(0, 1))
        score = score[:total_tokens]
        values, indices = torch.topk(score, k=min(keep_count, score.numel()))
        layer_to_tokens[layer_idx] = [
            (int(index), float(value)) for index, value in zip(indices.cpu(), values.cpu())
        ]
    return layer_to_tokens, keep_count


def dump_top_token_rows(
    path: Path,
    tokenizer: Any,
    prompt: str,
    sample: dict[str, Any],
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    layer_to_tokens: dict[int, list[tuple[int, float]]],
) -> None:
    fields = [
        "sample_id",
        "layer",
        "rank",
        "token_index",
        "attention_score",
        "token_id",
        "token_text",
        "context_piece",
        "left_context",
        "right_context",
        "category",
    ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        total_tokens = len(token_ids)
        for layer, pairs in layer_to_tokens.items():
            for rank, (token_index, score) in enumerate(pairs, start=1):
                start, end = offsets[token_index]
                left = prompt[max(0, start - 40) : start]
                right = prompt[end : min(len(prompt), end + 40)]
                row = {
                    "sample_id": sample["sample_id"],
                    "layer": layer,
                    "rank": rank,
                    "token_index": token_index,
                    "attention_score": score,
                    "token_id": token_ids[token_index],
                    "token_text": compact_piece(tokenizer.decode([token_ids[token_index]], skip_special_tokens=False)),
                    "context_piece": compact_piece(prompt[start:end]),
                    "left_context": compact_piece(left),
                    "right_context": compact_piece(right),
                    "category": classify_token(token_index, offsets, prompt, sample, total_tokens),
                }
                writer.writerow(row)


def choose_ablation_tokens(
    layer_to_tokens: dict[int, list[tuple[int, float]]],
    layer_spec: str,
) -> set[int]:
    if layer_spec == "last":
        layer = max(layer_to_tokens)
        return {idx for idx, _ in layer_to_tokens[layer]}
    if layer_spec == "all_union":
        return {idx for pairs in layer_to_tokens.values() for idx, _ in pairs}
    layer = int(layer_spec)
    return {idx for idx, _ in layer_to_tokens[layer]}


def compress_prompt_by_token_indices(
    tokenizer: Any,
    token_ids: list[int],
    keep_indices: set[int],
    include_special_tokens: bool,
) -> str:
    kept = [token_id for idx, token_id in enumerate(token_ids) if idx in keep_indices]
    return tokenizer.decode(kept, skip_special_tokens=not include_special_tokens)


def ablation_keep_sets(
    selected: set[int],
    offsets: list[tuple[int, int]],
    prompt: str,
    sample: dict[str, Any],
    total_tokens: int,
    rng: random.Random,
    random_trials: int,
) -> dict[str, set[int]]:
    categories: dict[str, set[int]] = {}
    for idx in selected:
        category = classify_token(idx, offsets, prompt, sample, total_tokens)
        categories.setdefault(category, set()).add(idx)
    answer_like = categories.get("answer_evidence", set()) | categories.get("needle_context", set())
    query_like = categories.get("query_bridge", set()) | categories.get("query_or_recent", set())
    sink_like = categories.get("sink_prefix", set())
    rare_like = categories.get("rare_or_number", set()) | categories.get("rare_or_symbol", set())

    keep_sets = {
        "attention_top1_all": set(selected),
        "answer_or_needle_only": set(answer_like),
        "answer_plus_query": set(answer_like | query_like),
        "answer_plus_sink": set(answer_like | sink_like),
        "answer_plus_rare": set(answer_like | rare_like),
        "drop_answer_or_needle": set(selected - answer_like),
        "drop_query_recent": set(selected - query_like),
        "drop_sink_prefix": set(selected - sink_like),
        "drop_rare": set(selected - rare_like),
    }
    population = list(range(total_tokens))
    for trial in range(random_trials):
        keep_sets[f"random_same_budget_{trial + 1}"] = set(rng.sample(population, k=min(len(selected), total_tokens)))
    return keep_sets


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "sample_count",
        "mean_loss",
        "mean_ppl",
        "mean_delta_loss_vs_full",
        "mean_prompt_token_count",
    ]
    modes = sorted({row["mode"] for row in rows})
    summary: list[dict[str, Any]] = []
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        mean_loss = sum(float(row["loss"]) for row in mode_rows) / len(mode_rows)
        summary.append(
            {
                "mode": mode,
                "sample_count": len(mode_rows),
                "mean_loss": mean_loss,
                "mean_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_delta_loss_vs_full": sum(float(row["delta_loss_vs_full"]) for row in mode_rows)
                / len(mode_rows),
                "mean_prompt_token_count": sum(int(row["prompt_token_count"]) for row in mode_rows)
                / len(mode_rows),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    args = parse_args()
    if args.top_ratio <= 0.0 or args.top_ratio > 1.0:
        raise ValueError("--top_ratio must be in (0, 1].")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    requested_device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map if requested_device.type != "cpu" else None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    if requested_device.type == "cpu":
        model.to(requested_device)
    model.eval()
    input_device = pick_input_device(model, requested_device)

    samples = load_samples(Path(args.data_path), args.max_samples, args.max_context_chars)
    result_rows: list[dict[str, Any]] = []
    per_sample_path = output_dir / "ablation_results.jsonl"
    top_token_path = output_dir / "layer_top1_tokens.csv"
    if top_token_path.exists():
        top_token_path.unlink()

    with per_sample_path.open("w", encoding="utf-8") as result_handle:
        for sample in samples:
            prompt = build_prompt(sample)
            token_ids, offsets = token_offsets(tokenizer, prompt)
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            layer_to_tokens, keep_count = collect_attention_top_tokens(
                model, input_ids, args.top_ratio, args.query_last_tokens, input_device
            )
            dump_top_token_rows(top_token_path, tokenizer, prompt, sample, token_ids, offsets, layer_to_tokens)

            full_metrics = answer_nll(model, tokenizer, prompt, sample["answer"], input_device)
            selected = choose_ablation_tokens(layer_to_tokens, args.ablation_layer)
            keep_sets = ablation_keep_sets(
                selected, offsets, prompt, sample, len(token_ids), rng, args.random_keep_trials
            )
            modes = {"full_context": prompt}
            for mode, keep_indices in keep_sets.items():
                if not keep_indices:
                    continue
                compressed = compress_prompt_by_token_indices(
                    tokenizer, token_ids, keep_indices, args.include_special_tokens
                )
                modes[mode] = (
                    "Use the compressed context tokens to answer the question.\n\n"
                    f"Compressed context:\n{compressed}\n\n"
                    f"Question: {sample['question']}\n"
                    "Answer:"
                )

            for mode, mode_prompt in modes.items():
                metrics = full_metrics if mode == "full_context" else answer_nll(
                    model, tokenizer, mode_prompt, sample["answer"], input_device
                )
                row = {
                    "sample_id": sample["sample_id"],
                    "mode": mode,
                    "answer": sample["answer"],
                    "loss": metrics["loss"],
                    "ppl": metrics["ppl"],
                    "delta_loss_vs_full": metrics["loss"] - full_metrics["loss"],
                    "prompt_token_count": metrics["prompt_token_count"],
                    "answer_token_count": metrics["answer_token_count"],
                    "top_ratio": args.top_ratio,
                    "keep_count": keep_count,
                    "ablation_layer": args.ablation_layer,
                }
                result_rows.append(row)
                result_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                result_handle.flush()
            print(f"finished {sample['sample_id']} with {keep_count} selected tokens per layer", flush=True)

    write_summary(output_dir / "ablation_summary.csv", result_rows)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    print(f"wrote {top_token_path}, {per_sample_path}, and {output_dir / 'ablation_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
