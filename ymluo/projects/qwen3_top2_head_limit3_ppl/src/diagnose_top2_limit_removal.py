from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="ymluo/models/Qwen3-0.6B")
    parser.add_argument("--text_path", default="external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt")
    parser.add_argument("--output_dir", default="outputs/removal_diagnostics")
    parser.add_argument("--prefill_tokens", type=int, default=1024)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--caps", default="3,8,12,15")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(max_chars) if max_chars > 0 else handle.read()


def model_forward(model: torch.nn.Module, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" in kwargs and "cache_position" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("cache_position")
            return model(**kwargs)
        raise


def pick_input_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def prefill_cache(model: torch.nn.Module, input_ids: torch.Tensor, prefill_tokens: int, chunk_size: int, device: torch.device) -> Any:
    past = None
    for idx, start in enumerate(range(0, prefill_tokens, chunk_size), start=1):
        end = min(start + chunk_size, prefill_tokens)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "cache_position": torch.arange(start, end, device=device),
        }
        if past is not None:
            kwargs["past_key_values"] = past
        print(f"prefill {idx}: {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past = outputs.past_key_values
    return past


def cap_keep_by_weight(top_indices: torch.Tensor, top_values: torch.Tensor, cap: int) -> torch.Tensor:
    # top_indices/top_values: [heads, top_k]
    heads, _ = top_indices.shape
    keep = torch.ones_like(top_indices, dtype=torch.bool)
    by_token: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for head in range(heads):
        for pos, key in enumerate(top_indices[head].tolist()):
            by_token[int(key)].append((head, pos, float(top_values[head, pos])))
    for entries in by_token.values():
        if len(entries) <= cap:
            continue
        entries.sort(key=lambda item: item[2], reverse=True)
        for head, pos, _ in entries[cap:]:
            keep[head, pos] = False
    return keep


def main() -> None:
    args = parse_args()
    caps = [int(part) for part in args.caps.split(",") if part.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    needed = args.prefill_tokens + args.eval_tokens
    if args.require_total_tokens and len(token_ids) < needed:
        raise ValueError(f"Need {needed} tokens, got {len(token_ids)}.")
    input_ids = torch.tensor(token_ids[:needed], dtype=torch.long).view(1, -1)
    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    device = pick_input_device(model, requested_device)
    layer_count = int(model.config.num_hidden_layers)
    head_count = int(model.config.num_attention_heads)
    past = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, device)

    stats: dict[tuple[int, int], dict[str, float]] = {}
    for layer in range(layer_count):
        for cap in caps:
            stats[(layer, cap)] = {
                "query_count": 0.0,
                "original_links": 0.0,
                "removed_links": 0.0,
                "original_weight": 0.0,
                "removed_weight": 0.0,
            }
    eval_end = args.prefill_tokens + args.eval_tokens
    for chunk_idx, start in enumerate(range(args.prefill_tokens, eval_end, args.chunk_size), start=1):
        end = min(start + args.chunk_size, eval_end)
        kwargs = {
            "input_ids": input_ids[:, start:end].to(device),
            "past_key_values": past,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": True,
            "cache_position": torch.arange(start, end, device=device),
        }
        print(f"diagnostic attention chunk {chunk_idx}: {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past = outputs.past_key_values
        for layer in range(layer_count):
            attention = outputs.attentions[layer][0].detach().float().cpu()
            for local_query in range(end - start):
                query_token = start + local_query
                history_count = query_token
                top_k = min(history_count, max(1, math.ceil(args.top_fraction * history_count)))
                if top_k <= 0:
                    continue
                values, indices = torch.topk(attention[:, local_query, :query_token], k=top_k, dim=-1)
                original_weight = float(values.sum())
                original_links = float(values.numel())
                for cap in caps:
                    keep = cap_keep_by_weight(indices, values, cap)
                    removed_weight = float(values.masked_fill(keep, 0.0).sum())
                    removed_links = float((~keep).sum())
                    s = stats[(layer, cap)]
                    s["query_count"] += 1
                    s["original_links"] += original_links
                    s["removed_links"] += removed_links
                    s["original_weight"] += original_weight
                    s["removed_weight"] += removed_weight
    rows: list[dict[str, Any]] = []
    for (layer, cap), s in sorted(stats.items()):
        rows.append(
            {
                "layer": layer,
                "cap": cap,
                "query_count": int(s["query_count"]),
                "removed_link_fraction": s["removed_links"] / s["original_links"] if s["original_links"] else 0.0,
                "removed_weight_fraction": s["removed_weight"] / s["original_weight"] if s["original_weight"] else 0.0,
                "mean_original_links_per_query": s["original_links"] / s["query_count"] if s["query_count"] else 0.0,
                "mean_removed_links_per_query": s["removed_links"] / s["query_count"] if s["query_count"] else 0.0,
                "mean_original_top2_weight_per_query": s["original_weight"] / s["query_count"] if s["query_count"] else 0.0,
                "mean_removed_top2_weight_per_query": s["removed_weight"] / s["query_count"] if s["query_count"] else 0.0,
            }
        )
    write_csv(
        output_dir / "removed_weight_by_layer_cap.csv",
        rows,
        [
            "layer",
            "cap",
            "query_count",
            "removed_link_fraction",
            "removed_weight_fraction",
            "mean_original_links_per_query",
            "mean_removed_links_per_query",
            "mean_original_top2_weight_per_query",
            "mean_removed_top2_weight_per_query",
        ],
    )

    summary: list[dict[str, Any]] = []
    for cap in caps:
        cap_rows = [row for row in rows if int(row["cap"]) == cap]
        summary.append(
            {
                "cap": cap,
                "mean_removed_link_fraction": sum(float(row["removed_link_fraction"]) for row in cap_rows) / len(cap_rows),
                "mean_removed_weight_fraction": sum(float(row["removed_weight_fraction"]) for row in cap_rows) / len(cap_rows),
                "mean_removed_links_per_query": sum(float(row["mean_removed_links_per_query"]) for row in cap_rows) / len(cap_rows),
                "mean_removed_top2_weight_per_query": sum(float(row["mean_removed_top2_weight_per_query"]) for row in cap_rows) / len(cap_rows),
            }
        )
    write_csv(
        output_dir / "removed_weight_summary_by_cap.csv",
        summary,
        ["cap", "mean_removed_link_fraction", "mean_removed_weight_fraction", "mean_removed_links_per_query", "mean_removed_top2_weight_per_query"],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=args.plot_dpi)
    ax.plot([row["cap"] for row in summary], [row["mean_removed_link_fraction"] for row in summary], marker="o", label="removed link fraction")
    ax.plot([row["cap"] for row in summary], [row["mean_removed_weight_fraction"] for row in summary], marker="o", label="removed attention-weight fraction")
    ax.set_title("What the score-based cap removes from original top2 selections")
    ax.set_xlabel("Maximum selecting heads allowed per historical token")
    ax.set_ylabel("Fraction of original top2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "removed_link_vs_weight_fraction_by_cap.png")
    plt.close(fig)
    (output_dir / "summary.json").write_text(json.dumps({"args": vars(args), "caps": caps}, indent=2), encoding="utf-8")
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
