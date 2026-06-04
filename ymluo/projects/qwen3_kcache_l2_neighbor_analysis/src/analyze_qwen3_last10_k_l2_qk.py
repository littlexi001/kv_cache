from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoConfig, AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer

from analyze_qwen3_kcache_l2_neighbors import (
    build_k_cache,
    extract_key_tensors,
    key_tensor_to_head_token_dim,
    load_config_with_rope_limit,
    parse_index_spec,
    pick_input_device,
    read_text_prefix,
    resolve_dtype,
    str2bool,
)


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-0.6B"
DEFAULT_TEXT_PATH = (
    "ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/"
    "prompts/niah_len8000_depth50.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot, for each selected layer/KV head, the final N tokens against all "
            "previous tokens: K-K L2 and scaled QK attention scores."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/last10_k_l2_qk")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_max_tokens", type=str2bool, default=False)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="auto")
    parser.add_argument("--rope_max_position_embeddings", type=int, default=8192)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--last_token_count", type=int, default=10)
    parser.add_argument(
        "--qk_reduce",
        choices=["mean", "max", "first"],
        default="mean",
        help="How to reduce the query heads that share one KV head in GQA.",
    )
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--line_alpha", type=float, default=0.85)
    parser.add_argument("--line_width", type=float, default=0.9)
    parser.add_argument(
        "--save_csv",
        type=str2bool,
        default=False,
        help="Save dense per-index values. This can be large for long contexts.",
    )
    return parser.parse_args()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_one(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (states * cos) + (rotate_half(states) * sin)


def attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules = [module for module in model.modules() if module.__class__.__name__ == "Qwen3Attention"]
    if not modules:
        raise RuntimeError("No Qwen3Attention modules found.")
    return modules


def install_query_capture_hooks(model: torch.nn.Module) -> tuple[dict[int, torch.Tensor], list[Any]]:
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(layer_idx: int):
        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            del output
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None:
                if len(args) >= 2:
                    position_embeddings = args[1]
                elif len(args) >= 1 and isinstance(args[-1], tuple):
                    position_embeddings = args[-1]
            if hidden_states is None or position_embeddings is None:
                return
            cos, sin = position_embeddings
            with torch.no_grad():
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)
                query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                query_states = apply_rope_one(query_states, cos, sin)
                captured[layer_idx] = query_states.detach().float().cpu()

        return hook

    for layer_idx, module in enumerate(attention_modules(model)):
        handles.append(module.register_forward_hook(make_hook(layer_idx), with_kwargs=True))
    return captured, handles


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reduce_qk_scores(scores: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        return scores.mean(dim=0)
    if mode == "max":
        return scores.max(dim=0).values
    if mode == "first":
        return scores[0]
    raise ValueError(f"Unsupported qk_reduce: {mode}")


def plot_lines(
    values_by_target: list[tuple[int, torch.Tensor]],
    output_path: Path,
    title: str,
    ylabel: str,
    dpi: int,
    alpha: float,
    line_width: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=dpi)
    for target_index, values in values_by_target:
        if values.numel() == 0:
            continue
        x = torch.arange(values.numel())
        ax.plot(
            x.tolist(),
            values.float().tolist(),
            linewidth=line_width,
            alpha=alpha,
            label=f"token {target_index}",
        )
    ax.set_title(title)
    ax.set_xlabel("Previous token index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, ncol=5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def load_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, list[int]]:
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    if args.require_max_tokens and len(token_ids) < args.max_tokens:
        raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than {args.max_tokens}.")
    token_ids = token_ids[: args.max_tokens]
    if len(token_ids) <= 1:
        raise ValueError("Need at least two tokens.")
    return torch.tensor(token_ids, dtype=torch.long).view(1, -1), token_ids


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    config = load_config_with_rope_limit(args.model_name_or_path, args.rope_max_position_embeddings)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype, "config": config}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True
    return model


def main() -> None:
    args = parse_args()
    if args.last_token_count <= 0:
        raise ValueError("--last_token_count must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_ids, token_ids = load_inputs(args)
    total_tokens = int(input_ids.shape[1])
    last_count = min(args.last_token_count, total_tokens - 1)
    target_indices = list(range(total_tokens - last_count, total_tokens))

    model = load_model(args)
    captured_queries, handles = install_query_capture_hooks(model)
    try:
        input_device = pick_input_device(model, torch.device(args.device if torch.cuda.is_available() else "cpu"))
        past_key_values, timing_rows = build_k_cache(model, input_ids, args.chunk_size, input_device)
    finally:
        for handle in handles:
            handle.remove()

    write_csv(
        output_dir / "profile_timings.csv",
        timing_rows,
        ["chunk", "start_token", "end_token_exclusive", "token_count", "seconds"],
    )

    key_tensors = extract_key_tensors(past_key_values)
    expected_kv_heads = getattr(model.config, "num_key_value_heads", None)
    num_attention_heads = int(getattr(model.config, "num_attention_heads"))
    num_kv_heads = int(expected_kv_heads or getattr(model.config, "num_key_value_heads", num_attention_heads))
    query_heads_per_kv = max(1, num_attention_heads // num_kv_heads)
    layer_indices = parse_index_spec(args.layers, len(key_tensors), "layers")

    csv_rows: list[dict[str, Any]] = []
    plot_paths: list[str] = []
    for layer_idx in layer_indices:
        if layer_idx not in captured_queries:
            raise RuntimeError(f"No captured query states for layer {layer_idx}.")
        key_by_head = key_tensor_to_head_token_dim(key_tensors[layer_idx], expected_kv_heads)
        query_by_head_chunk = captured_queries[layer_idx].squeeze(0)
        kv_heads, tokens, head_dim = key_by_head.shape
        if tokens != total_tokens:
            raise RuntimeError(f"Layer {layer_idx}: expected {total_tokens} K tokens, got {tokens}.")
        head_indices = parse_index_spec(args.heads, int(kv_heads), "heads")
        query_last = query_by_head_chunk[:, -last_count:, :]
        scaling = float(getattr(attention_modules(model)[layer_idx], "scaling", 1.0 / math.sqrt(head_dim)))

        for kv_head in head_indices:
            k_vectors = key_by_head[kv_head].float()
            q_start = kv_head * query_heads_per_kv
            q_end = min(q_start + query_heads_per_kv, int(query_last.shape[0]))
            if q_start >= q_end:
                raise RuntimeError(
                    f"KV head {kv_head} maps to empty query-head range {q_start}:{q_end}."
                )
            q_vectors = query_last[q_start:q_end].float()

            kk_values: list[tuple[int, torch.Tensor]] = []
            qk_values: list[tuple[int, torch.Tensor]] = []
            for local_idx, target_index in enumerate(target_indices):
                previous_k = k_vectors[:target_index]
                target_k = k_vectors[target_index].view(1, -1)
                kk_l2 = torch.linalg.vector_norm(previous_k - target_k, dim=-1)
                q_scores_by_head = torch.matmul(q_vectors[:, local_idx, :], previous_k.T) * scaling
                qk_scores = reduce_qk_scores(q_scores_by_head, args.qk_reduce)
                kk_values.append((target_index, kk_l2.cpu()))
                qk_values.append((target_index, qk_scores.cpu()))
                if args.save_csv:
                    for previous_index in range(target_index):
                        csv_rows.append(
                            {
                                "layer": layer_idx,
                                "kv_head": kv_head,
                                "target_token_index": target_index,
                                "previous_token_index": previous_index,
                                "k_l2": float(kk_l2[previous_index]),
                                "qk_score": float(qk_scores[previous_index]),
                            }
                        )

            kk_path = output_dir / "plots" / "k_l2" / f"layer_{layer_idx:02d}" / f"head_{kv_head:02d}.png"
            qk_path = output_dir / "plots" / "qk_score" / f"layer_{layer_idx:02d}" / f"head_{kv_head:02d}.png"
            plot_lines(
                kk_values,
                kk_path,
                f"K-K L2, layer {layer_idx}, KV head {kv_head}",
                "L2 distance",
                args.plot_dpi,
                args.line_alpha,
                args.line_width,
            )
            plot_lines(
                qk_values,
                qk_path,
                f"Scaled QK score, layer {layer_idx}, KV head {kv_head} ({args.qk_reduce} over shared Q heads)",
                "q · k / sqrt(d)",
                args.plot_dpi,
                args.line_alpha,
                args.line_width,
            )
            plot_paths.extend([str(kk_path), str(qk_path)])
        del key_by_head

    if args.save_csv:
        write_csv(
            output_dir / "last_tokens_k_l2_qk_by_index.csv",
            csv_rows,
            ["layer", "kv_head", "target_token_index", "previous_token_index", "k_l2", "qk_score"],
        )

    summary = {
        "args": vars(args),
        "resolved": {
            "tokens": total_tokens,
            "last_token_count": last_count,
            "target_token_indices": target_indices,
            "layers": layer_indices,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_kv_heads,
            "query_heads_per_kv": query_heads_per_kv,
            "qk_reduce": args.qk_reduce,
            "plot_count": len(plot_paths),
            "csv_saved": bool(args.save_csv),
        },
        "paths": {
            "plots_dir": str(output_dir / "plots"),
            "k_l2_plots_dir": str(output_dir / "plots" / "k_l2"),
            "qk_score_plots_dir": str(output_dir / "plots" / "qk_score"),
            "profile_timings": str(output_dir / "profile_timings.csv"),
            "dense_csv": str(output_dir / "last_tokens_k_l2_qk_by_index.csv") if args.save_csv else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(plot_paths)} plots to: {output_dir / 'plots'}", flush=True)


if __name__ == "__main__":
    main()
