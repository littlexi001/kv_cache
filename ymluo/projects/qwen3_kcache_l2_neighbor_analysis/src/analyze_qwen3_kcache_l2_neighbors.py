from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoConfig, AutoModelWithLMHead as AutoModelForCausalLM
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
        description="Find per-token top-k nearest K-cache vectors by pairwise L2 distance."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/k_l2_neighbors")
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_max_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="auto")
    parser.add_argument(
        "--rope_max_position_embeddings",
        type=int,
        default=8192,
        help="Ensure config.max_position_embeddings is at least this value before loading the model.",
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--neighbor_count", type=int, default=5)
    parser.add_argument(
        "--neighbor_scope",
        choices=["all", "previous"],
        default="all",
        help='Use "all" for nearest neighbors among all other tokens, or "previous" for j < i only.',
    )
    parser.add_argument(
        "--variants",
        default="raw",
        help='Vector variants, e.g. "raw", "centered", or "raw,centered".',
    )
    parser.add_argument("--distance_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save_neighbor_csv", type=str2bool, default=True)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--point_alpha", type=float, default=0.35)
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
    invalid = sorted(item for item in selected if item < 0 or item >= max_count)
    if invalid:
        raise ValueError(f"{name} out of range 0..{max_count - 1}: {invalid}")
    return sorted(selected)


def parse_variants(value: str) -> list[str]:
    variants = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not variants:
        raise ValueError("At least one variant is required.")
    invalid = [item for item in variants if item not in {"raw", "centered"}]
    if invalid:
        raise ValueError(f'Unsupported variants: {invalid}. Expected "raw" and/or "centered".')
    return list(dict.fromkeys(variants))


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


def resolve_distance_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--distance_device cuda requested, but CUDA is not available.")
    return torch.device(name)


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        if max_chars > 0:
            return handle.read(max_chars)
        return handle.read()


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
            key_by_head = key.permute(1, 0, 2, 3).reshape(dim1, batch * dim2, head_dim)
        elif expected_heads is not None and dim2 == expected_heads:
            key_by_head = key.permute(2, 0, 1, 3).reshape(dim2, batch * dim1, head_dim)
        elif dim1 <= dim2:
            key_by_head = key.permute(1, 0, 2, 3).reshape(dim1, batch * dim2, head_dim)
        else:
            key_by_head = key.permute(2, 0, 1, 3).reshape(dim2, batch * dim1, head_dim)
    elif key.ndim == 3:
        dim1, dim2, _ = key.shape
        if expected_heads is not None and dim1 == expected_heads:
            key_by_head = key
        elif expected_heads is not None and dim2 == expected_heads:
            key_by_head = key.permute(1, 0, 2)
        elif dim1 <= dim2:
            key_by_head = key
        else:
            key_by_head = key.permute(1, 0, 2)
    else:
        raise ValueError(f"Expected 3D or 4D key tensor, got shape {tuple(key.shape)}")
    return key_by_head.float().cpu()


@torch.inference_mode()
def build_k_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, list[dict[str, Any]]]:
    total_tokens = int(input_ids.shape[1])
    past_key_values = None
    timing_rows: list[dict[str, Any]] = []
    total_chunks = math.ceil(total_tokens / chunk_size)
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
        print(f"profile chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        started = time.perf_counter()
        outputs = model_forward(model, kwargs)
        seconds = time.perf_counter() - started
        past_key_values = outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError("Model did not return past_key_values.")
        timing_rows.append(
            {
                "chunk": chunk_idx,
                "start_token": start,
                "end_token_exclusive": end,
                "token_count": end - start,
                "seconds": seconds,
            }
        )
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return past_key_values, timing_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], append: bool) -> None:
    with path.open("a" if append else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def vector_variant(vectors: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "raw":
        return vectors
    if variant == "centered":
        return vectors - vectors.mean(dim=0, keepdim=True)
    raise ValueError(f"Unsupported variant: {variant}")


@torch.inference_mode()
def top_l2_neighbors(
    vectors: torch.Tensor,
    neighbor_count: int,
    scope: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    working = vectors.to(device=device, dtype=torch.float32)
    distances = torch.cdist(working, working, p=2)
    tokens = int(distances.shape[0])
    if scope == "all":
        distances.fill_diagonal_(float("inf"))
    elif scope == "previous":
        mask = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool, device=device), diagonal=0)
        distances.masked_fill_(mask, float("inf"))
    else:
        raise ValueError(f"Unsupported neighbor scope: {scope}")
    k = min(neighbor_count, max(1, tokens - 1))
    values, indices = torch.topk(distances, k=k, dim=1, largest=False)
    del distances, working
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values.cpu(), indices.cpu()


def neighbor_fieldnames() -> list[str]:
    return [
        "cache_type",
        "variant",
        "layer",
        "head",
        "token_index",
        "neighbor_rank",
        "neighbor_index",
        "index_distance",
        "l2_distance",
    ]


def summary_fieldnames() -> list[str]:
    return [
        "cache_type",
        "variant",
        "layer",
        "head",
        "tokens",
        "head_dim",
        "neighbor_count",
        "neighbor_scope",
        "mean_l2_distance",
        "mean_index_distance",
        "median_index_distance",
        "p95_index_distance",
        "max_index_distance",
    ]


def rows_from_neighbors(
    cache_type: str,
    variant: str,
    layer: int,
    head: int,
    distances: torch.Tensor,
    indices: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token_index in range(int(indices.shape[0])):
        for rank in range(int(indices.shape[1])):
            neighbor_index = int(indices[token_index, rank])
            l2_distance = float(distances[token_index, rank])
            if not math.isfinite(l2_distance):
                continue
            rows.append(
                {
                    "cache_type": cache_type,
                    "variant": variant,
                    "layer": layer,
                    "head": head,
                    "token_index": token_index,
                    "neighbor_rank": rank + 1,
                    "neighbor_index": neighbor_index,
                    "index_distance": abs(token_index - neighbor_index),
                    "l2_distance": l2_distance,
                }
            )
    return rows


def summary_from_neighbors(
    cache_type: str,
    variant: str,
    layer: int,
    head: int,
    tokens: int,
    head_dim: int,
    neighbor_count: int,
    neighbor_scope: str,
    distances: torch.Tensor,
    indices: torch.Tensor,
) -> dict[str, Any]:
    token_indices = torch.arange(tokens, dtype=torch.long).view(-1, 1)
    finite_mask = torch.isfinite(distances)
    index_distances = (token_indices - indices.long()).abs().float()[finite_mask]
    finite_distances = distances.float()[finite_mask]
    return {
        "cache_type": cache_type,
        "variant": variant,
        "layer": layer,
        "head": head,
        "tokens": tokens,
        "head_dim": head_dim,
        "neighbor_count": neighbor_count,
        "neighbor_scope": neighbor_scope,
        "mean_l2_distance": float(finite_distances.mean()) if finite_distances.numel() else 0.0,
        "mean_index_distance": float(index_distances.mean()) if index_distances.numel() else 0.0,
        "median_index_distance": float(index_distances.median()) if index_distances.numel() else 0.0,
        "p95_index_distance": float(torch.quantile(index_distances, 0.95)) if index_distances.numel() else 0.0,
        "max_index_distance": float(index_distances.max()) if index_distances.numel() else 0.0,
    }


def plot_neighbor_scatter(
    distances: torch.Tensor,
    indices: torch.Tensor,
    output_dir: Path,
    cache_type: str,
    variant: str,
    layer: int,
    head: int,
    dpi: int,
    alpha: float,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / cache_type / variant / f"layer_{layer:02d}" / f"head_{head:02d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    token_indices = torch.arange(int(indices.shape[0]))
    finite_mask = torch.isfinite(distances)
    index_distances = (token_indices.view(-1, 1) - indices.long()).abs()
    paths: list[str] = []
    specs = [
        ("index_distance_by_rank_tokens.png", index_distances.float(), "Index distance"),
        ("l2_distance_by_rank_tokens.png", distances.float(), "L2 distance"),
    ]
    for filename, values, ylabel in specs:
        fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
        for rank in range(int(indices.shape[1])):
            rank_mask = finite_mask[:, rank]
            if not bool(rank_mask.any()):
                continue
            ax.scatter(
                token_indices[rank_mask].tolist(),
                values[:, rank][rank_mask].tolist(),
                s=2,
                alpha=alpha,
                linewidths=0,
                label=f"top{rank + 1}",
            )
        ax.set_title(f"{cache_type.upper()} {variant} L{layer} H{head}: nearest-L2 {ylabel}")
        ax.set_xlabel("Token index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
        ax.legend(markerscale=4, fontsize=8, ncol=min(int(indices.shape[1]), 5))
        fig.tight_layout()
        path = plot_dir / filename
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def load_config_with_rope_limit(model_path: str, rope_max_position_embeddings: int) -> Any:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    original = getattr(config, "max_position_embeddings", None)
    if rope_max_position_embeddings > 0 and (
        original is None or int(original) < rope_max_position_embeddings
    ):
        print(
            f"setting config.max_position_embeddings from {original} to {rope_max_position_embeddings}",
            flush=True,
        )
        config.max_position_embeddings = rope_max_position_embeddings
    return config


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 1:
        raise ValueError("--max_tokens must be greater than 1.")
    if args.neighbor_count <= 0:
        raise ValueError("--neighbor_count must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = Path(args.text_path)
    variants = parse_variants(args.variants)

    text = read_text_prefix(text_path, args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    if args.require_max_tokens and len(token_ids) < args.max_tokens:
        raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than {args.max_tokens}.")
    token_ids = token_ids[: args.max_tokens]
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

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

    input_device = pick_input_device(model, requested_device)
    past_key_values, timing_rows = build_k_cache(model, input_ids, args.chunk_size, input_device)
    write_csv(output_dir / "profile_timings.csv", timing_rows, ["chunk", "start_token", "end_token_exclusive", "token_count", "seconds"])

    key_tensors = extract_key_tensors(past_key_values)
    expected_heads = getattr(model.config, "num_key_value_heads", None)
    layer_indices = parse_index_spec(args.layers, len(key_tensors), "layers")
    distance_device = resolve_distance_device(args.distance_device)

    neighbor_path = output_dir / "nearest_neighbors_by_token.csv"
    wrote_neighbors = False
    summary_rows: list[dict[str, Any]] = []
    plot_paths: list[str] = []

    for layer_idx in layer_indices:
        key_by_head = key_tensor_to_head_token_dim(key_tensors[layer_idx], expected_heads)
        kv_heads, tokens, head_dim = key_by_head.shape
        head_indices = parse_index_spec(args.heads, int(kv_heads), "heads")
        for head_idx in head_indices:
            for variant in variants:
                print(
                    f"computing top-{args.neighbor_count} L2 neighbors for K {variant} "
                    f"layer {layer_idx}, head {head_idx}",
                    flush=True,
                )
                vectors = vector_variant(key_by_head[head_idx], variant)
                distances, indices = top_l2_neighbors(vectors, args.neighbor_count, args.neighbor_scope, distance_device)
                summary_rows.append(
                    summary_from_neighbors(
                        "k",
                        variant,
                        layer_idx,
                        head_idx,
                        int(tokens),
                        int(head_dim),
                        args.neighbor_count,
                        args.neighbor_scope,
                        distances,
                        indices,
                    )
                )
                if args.save_neighbor_csv:
                    rows = rows_from_neighbors("k", variant, layer_idx, head_idx, distances, indices)
                    append_rows(neighbor_path, rows, neighbor_fieldnames(), append=wrote_neighbors)
                    wrote_neighbors = True
                if args.make_plots:
                    plot_paths.extend(
                        plot_neighbor_scatter(
                            distances,
                            indices,
                            output_dir,
                            "k",
                            variant,
                            layer_idx,
                            head_idx,
                            args.plot_dpi,
                            args.point_alpha,
                        )
                    )
                del distances, indices
        del key_by_head

    write_csv(output_dir / "summary_by_head.csv", summary_rows, summary_fieldnames())
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "tokens": int(input_ids.shape[1]),
                    "config_max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
                    "layers": layer_indices,
                    "variants": variants,
                },
                "paths": {
                    "nearest_neighbors_by_token": str(neighbor_path) if args.save_neighbor_csv else None,
                    "summary_by_head": str(output_dir / "summary_by_head.csv"),
                    "plots_dir": str(output_dir / "plots") if args.make_plots else None,
                    "plot_count": len(plot_paths),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
