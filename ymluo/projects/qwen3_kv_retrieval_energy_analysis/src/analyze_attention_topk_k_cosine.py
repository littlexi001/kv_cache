from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-0.6B"
DEFAULT_TEXT_PATH = (
    "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pairwise K cosine for attention top-2% tokens versus ordinary K tokens."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/attention_topk_k_cosine")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_max_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--kv_heads", default="all")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--query_stride", type=int, default=128)
    parser.add_argument("--min_visible_tokens", type=int, default=128)
    parser.add_argument("--random_samples_per_query", type=int, default=1)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--percentiles", default="0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99")
    parser.add_argument("--make_heatmaps", type=str2bool, default=True)
    parser.add_argument("--heatmap_query_positions", default="0.25,0.5,0.75,last")
    parser.add_argument("--heatmap_layers", default="0,13,27")
    parser.add_argument("--heatmap_kv_heads", default="0")
    parser.add_argument("--heatmap_attention_heads", default="auto")
    parser.add_argument("--heatmap_max_vectors", type=int, default=192)
    parser.add_argument("--plot_dpi", type=int, default=180)
    return parser.parse_args()


def parse_index_spec(spec: str, max_count: int, name: str) -> list[int]:
    normalized = spec.strip().lower()
    if normalized == "all":
        return list(range(max_count))
    selected: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid {name} range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = sorted(index for index in selected if index < 0 or index >= max_count)
    if invalid:
        raise ValueError(f"{name} out of range 0..{max_count - 1}: {invalid}")
    return sorted(selected)


def parse_percentiles(spec: str) -> list[float]:
    values = [float(part.strip()) for part in spec.split(",") if part.strip()]
    invalid = [value for value in values if value < 0.0 or value > 1.0]
    if invalid:
        raise ValueError(f"Percentiles must be in [0, 1], got {invalid}")
    return values


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


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(max_chars) if max_chars > 0 else handle.read()


def pick_input_device(model: torch.nn.Module, fallback_device: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def model_forward(model: torch.nn.Module, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" in kwargs and "cache_position" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("cache_position")
            return model(**kwargs)
        raise


def extract_key_tensors(past_key_values: Any) -> list[torch.Tensor]:
    if hasattr(past_key_values, "key_cache"):
        return list(past_key_values.key_cache)
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        return [layer_cache[0] for layer_cache in legacy_cache]
    if isinstance(past_key_values, (list, tuple)):
        if past_key_values and isinstance(past_key_values[0], (list, tuple)):
            return [layer_cache[0] for layer_cache in past_key_values]
    if hasattr(past_key_values, "layers"):
        key_tensors: list[torch.Tensor] = []
        for layer_cache in past_key_values.layers:
            for attr_name in ("keys", "key_cache", "key_states"):
                if hasattr(layer_cache, attr_name):
                    key_tensors.append(getattr(layer_cache, attr_name))
                    break
        if key_tensors:
            return key_tensors
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def key_tensor_to_head_token_dim(key_tensor: torch.Tensor, expected_heads: int | None) -> torch.Tensor:
    key = key_tensor.detach()
    if key.ndim == 4:
        batch, dim1, dim2, head_dim = key.shape
        if expected_heads is not None and dim1 == expected_heads:
            return key.permute(1, 0, 2, 3).reshape(dim1, batch * dim2, head_dim).float().cpu()
        if expected_heads is not None and dim2 == expected_heads:
            return key.permute(2, 0, 1, 3).reshape(dim2, batch * dim1, head_dim).float().cpu()
        if dim1 <= dim2:
            return key.permute(1, 0, 2, 3).reshape(dim1, batch * dim2, head_dim).float().cpu()
        return key.permute(2, 0, 1, 3).reshape(dim2, batch * dim1, head_dim).float().cpu()
    if key.ndim == 3:
        dim1, dim2, _ = key.shape
        if expected_heads is not None and dim1 == expected_heads:
            return key.float().cpu()
        if expected_heads is not None and dim2 == expected_heads:
            return key.permute(1, 0, 2).float().cpu()
        if dim1 <= dim2:
            return key.float().cpu()
        return key.permute(1, 0, 2).float().cpu()
    raise ValueError(f"Expected 3D or 4D key tensor, got shape {tuple(key.shape)}")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def build_k_cache(model: torch.nn.Module, input_ids: torch.Tensor, chunk_size: int, input_device: torch.device) -> Any:
    total_tokens = int(input_ids.shape[1])
    total_chunks = math.ceil(total_tokens / chunk_size)
    past_key_values = None
    for chunk_idx, start in enumerate(range(0, total_tokens, chunk_size), start=1):
        end = min(start + chunk_size, total_tokens)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        print(f"K-cache chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return past_key_values


def cosine_matrix(vectors: torch.Tensor, centered: bool) -> torch.Tensor:
    source = vectors.float()
    if centered:
        source = source - source.mean(dim=0, keepdim=True)
    source = torch.nn.functional.normalize(source, p=2, dim=-1, eps=1e-8)
    return source @ source.transpose(0, 1)


def offdiag_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[0] <= 1:
        return torch.empty(0, dtype=matrix.dtype)
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def summarize_cosines(values: list[float], percentiles: list[float]) -> dict[str, Any]:
    tensor = torch.tensor(values, dtype=torch.float32)
    row: dict[str, Any] = {
        "pair_count": int(tensor.numel()),
        "mean": float(tensor.mean()) if tensor.numel() else "",
        "std": float(tensor.std(unbiased=False)) if tensor.numel() else "",
        "min": float(tensor.min()) if tensor.numel() else "",
        "max": float(tensor.max()) if tensor.numel() else "",
    }
    for percentile in percentiles:
        key = f"p{int(round(percentile * 100)):02d}"
        row[key] = float(torch.quantile(tensor, percentile)) if tensor.numel() else ""
    return row


def summary_fields(percentiles: list[float]) -> list[str]:
    fields = [
        "layer",
        "kv_head",
        "attention_head",
        "source",
        "variant",
        "query_count",
        "vector_count_mean",
        "pair_count",
        "mean",
        "std",
        "min",
    ]
    fields.extend(f"p{int(round(percentile * 100)):02d}" for percentile in percentiles)
    fields.append("max")
    return fields


def token_fields(percentiles: list[float]) -> list[str]:
    fields = [
        "layer",
        "kv_head",
        "attention_head",
        "query_token",
        "visible_tokens",
        "source",
        "variant",
        "vector_count",
        "pair_count",
        "mean",
        "std",
        "min",
    ]
    fields.extend(f"p{int(round(percentile * 100)):02d}" for percentile in percentiles)
    fields.append("max")
    return fields


def sample_ordinary_indices(visible: int, count: int, rng: random.Random) -> torch.Tensor:
    if count >= visible:
        return torch.arange(visible, dtype=torch.long)
    return torch.tensor(sorted(rng.sample(range(visible), count)), dtype=torch.long)


def save_centered_heatmap(matrix: torch.Tensor, path: Path, title: str, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=dpi)
    image = ax.imshow(matrix.cpu().numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("selected token rank")
    ax.set_ylabel("selected token rank")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="centered cosine")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_heatmap_query_positions(spec: str, max_tokens: int) -> list[int]:
    result: set[int] = set()
    for part in spec.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item == "last":
            result.add(max_tokens - 1)
        elif "." in item:
            value = float(item)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Heatmap fractional position must be in [0, 1], got {item}")
            result.add(max(0, min(max_tokens - 1, int(round(value * (max_tokens - 1))))))
        else:
            result.add(max(0, min(max_tokens - 1, int(item))))
    return sorted(result)


def pick_heatmap_attention_heads(spec: str, kv_heads: list[int], group_size: int, attention_heads: int) -> list[int]:
    if spec.strip().lower() != "auto":
        return parse_index_spec(spec, attention_heads, "heatmap_attention_heads")
    heads: list[int] = []
    for kv_head in kv_heads:
        head = kv_head * group_size
        if head < attention_heads:
            heads.append(head)
    return sorted(set(heads))


def main() -> None:
    args = parse_args()
    if args.top_fraction <= 0.0:
        raise ValueError("--top_fraction must be positive.")
    if args.query_stride <= 0:
        raise ValueError("--query_stride must be positive.")
    if args.random_samples_per_query <= 0:
        raise ValueError("--random_samples_per_query must be positive.")
    percentiles = parse_percentiles(args.percentiles)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    if args.require_max_tokens and len(token_ids) < args.max_tokens:
        raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than --max_tokens {args.max_tokens}.")
    token_ids = token_ids[: args.max_tokens]
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True

    layer_count = int(getattr(model.config, "num_hidden_layers"))
    attention_head_count = int(getattr(model.config, "num_attention_heads"))
    kv_head_count = int(getattr(model.config, "num_key_value_heads"))
    group_size = attention_head_count // kv_head_count
    layers = parse_index_spec(args.layers, layer_count, "layers")
    kv_heads = parse_index_spec(args.kv_heads, kv_head_count, "kv_heads")
    selected_attention_heads = [
        head
        for kv_head in kv_heads
        for head in range(kv_head * group_size, min((kv_head + 1) * group_size, attention_head_count))
    ]
    input_device = pick_input_device(model, requested_device)

    print("building K cache", flush=True)
    past_key_values = build_k_cache(model, input_ids, args.chunk_size, input_device)
    key_tensors = extract_key_tensors(past_key_values)
    keys_by_layer = {
        layer: key_tensor_to_head_token_dim(key_tensors[layer], kv_head_count)
        for layer in layers
    }
    del past_key_values, key_tensors
    if input_device.type == "cuda":
        torch.cuda.empty_cache()

    heatmap_queries = set(parse_heatmap_query_positions(args.heatmap_query_positions, int(input_ids.shape[1])))
    heatmap_layers = set(parse_index_spec(args.heatmap_layers, layer_count, "heatmap_layers")) & set(layers)
    heatmap_kv_heads = set(parse_index_spec(args.heatmap_kv_heads, kv_head_count, "heatmap_kv_heads")) & set(kv_heads)
    heatmap_attention_heads = set(
        pick_heatmap_attention_heads(args.heatmap_attention_heads, sorted(heatmap_kv_heads), group_size, attention_head_count)
    )
    rng = random.Random(args.sample_seed)

    token_rows: list[dict[str, Any]] = []
    grouped_values: dict[tuple[int, int, int, str, str], list[float]] = defaultdict(list)
    grouped_counts: dict[tuple[int, int, int, str, str], list[int]] = defaultdict(list)

    total_tokens = int(input_ids.shape[1])
    total_chunks = math.ceil(total_tokens / args.chunk_size)
    past_key_values = None
    for chunk_idx, start in enumerate(range(0, total_tokens, args.chunk_size), start=1):
        end = min(start + args.chunk_size, total_tokens)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": True,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        print(f"attention chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        with torch.inference_mode():
            outputs = model_forward(model, kwargs)
        if outputs.attentions is None:
            raise RuntimeError("Model did not return attentions. Use ATTN_IMPLEMENTATION=eager.")
        past_key_values = outputs.past_key_values

        for local_query in range(end - start):
            query_token = start + local_query
            visible = query_token + 1
            if visible < args.min_visible_tokens or query_token % args.query_stride != 0:
                if query_token not in heatmap_queries:
                    continue
            count = max(1, math.ceil(args.top_fraction * visible))
            for layer in layers:
                attention_layer = outputs.attentions[layer][0].detach().float().cpu()
                key_by_kv_head = keys_by_layer[layer]
                for attention_head in selected_attention_heads:
                    kv_head = attention_head // group_size
                    attention = attention_layer[attention_head, local_query, :visible]
                    top_indices = torch.topk(attention, k=min(count, visible), largest=True).indices.long()
                    ordinary_indices_list = [
                        sample_ordinary_indices(visible, int(top_indices.numel()), rng)
                        for _ in range(args.random_samples_per_query)
                    ]
                    sources = [("attention_top", top_indices)]
                    sources.extend((f"ordinary_random_{idx}", ordinary_indices) for idx, ordinary_indices in enumerate(ordinary_indices_list))
                    for source_name, indices in sources:
                        vectors = key_by_kv_head[kv_head, indices]
                        for variant, centered in (("raw", False), ("centered", True)):
                            matrix = cosine_matrix(vectors, centered=centered)
                            values = offdiag_values(matrix)
                            row = {
                                "layer": layer,
                                "kv_head": kv_head,
                                "attention_head": attention_head,
                                "query_token": query_token,
                                "visible_tokens": visible,
                                "source": "ordinary_random" if source_name.startswith("ordinary_random") else source_name,
                                "variant": variant,
                                "vector_count": int(indices.numel()),
                            }
                            row.update(summarize_cosines(values.tolist(), percentiles))
                            token_rows.append(row)
                            group_key = (layer, kv_head, attention_head, row["source"], variant)
                            grouped_values[group_key].extend(float(value) for value in values.tolist())
                            grouped_counts[group_key].append(int(indices.numel()))
                        if (
                            args.make_heatmaps
                            and source_name in {"attention_top", "ordinary_random_0"}
                            and layer in heatmap_layers
                            and kv_head in heatmap_kv_heads
                            and attention_head in heatmap_attention_heads
                            and query_token in heatmap_queries
                        ):
                            heatmap_indices = indices[: args.heatmap_max_vectors]
                            heatmap_vectors = key_by_kv_head[kv_head, heatmap_indices]
                            heatmap = cosine_matrix(heatmap_vectors, centered=True)
                            source_label = "ordinary_random" if source_name.startswith("ordinary_random") else source_name
                            plot_path = (
                                output_dir
                                / "plots"
                                / f"layer_{layer:02d}"
                                / f"kv_head_{kv_head:02d}"
                                / f"attention_head_{attention_head:02d}"
                                / f"query_{query_token:06d}_{source_label}_centered_cosine.png"
                            )
                            save_centered_heatmap(
                                heatmap,
                                plot_path,
                                f"L{layer} KV{kv_head} H{attention_head} q{query_token} {source_label}",
                                args.plot_dpi,
                            )
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows: list[dict[str, Any]] = []
    for (layer, kv_head, attention_head, source, variant), values in sorted(grouped_values.items()):
        row = {
            "layer": layer,
            "kv_head": kv_head,
            "attention_head": attention_head,
            "source": source,
            "variant": variant,
            "query_count": len(grouped_counts[(layer, kv_head, attention_head, source, variant)]),
            "vector_count_mean": sum(grouped_counts[(layer, kv_head, attention_head, source, variant)])
            / max(1, len(grouped_counts[(layer, kv_head, attention_head, source, variant)])),
        }
        row.update(summarize_cosines(values, percentiles))
        summary_rows.append(row)

    summary_path = output_dir / "cosine_summary_by_layer_head.csv"
    token_path = output_dir / "cosine_by_query.csv"
    write_csv(summary_path, summary_rows, summary_fields(percentiles))
    write_csv(token_path, token_rows, token_fields(percentiles))
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "tokens": total_tokens,
                    "layers": layers,
                    "kv_heads": kv_heads,
                    "attention_heads": selected_attention_heads,
                    "heatmap_queries": sorted(heatmap_queries),
                    "heatmap_layers": sorted(heatmap_layers),
                    "heatmap_kv_heads": sorted(heatmap_kv_heads),
                    "heatmap_attention_heads": sorted(heatmap_attention_heads),
                },
                "paths": {
                    "cosine_summary_by_layer_head": str(summary_path),
                    "cosine_by_query": str(token_path),
                    "plots_dir": str(output_dir / "plots") if args.make_heatmaps else None,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote attention top-K cosine outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
