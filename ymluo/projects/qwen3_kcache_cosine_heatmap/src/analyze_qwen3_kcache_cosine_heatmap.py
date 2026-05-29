from __future__ import annotations

import argparse
import csv
import json
import math
import time
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
DEFAULT_TEXT_PATH = (
    "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"
)
DEFAULT_PERCENTILES = "1,5,25,50,75,95,99"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Qwen3 K cache from a DCLM prefix and plot per-layer/head "
            "token-token cosine similarity heatmaps."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/kcache_cosine_heatmap")
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument(
        "--max_chars",
        type=int,
        default=8_000_000,
        help="Read at most this many characters from text_path. Use 0 to read the full file.",
    )
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument(
        "--require_max_tokens",
        type=str2bool,
        default=True,
        help="Fail if tokenization produces fewer than --max_tokens tokens.",
    )
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Use "auto" for accelerate placement, or "none" to move the full model to --device.',
    )
    parser.add_argument(
        "--attn_implementation",
        default="auto",
        help='Attention backend passed to from_pretrained. Use "auto" to leave it unset.',
    )
    parser.add_argument(
        "--layers",
        default="all",
        help='Layer selection, e.g. "all", "0", "0,7,15", or "0-3,10".',
    )
    parser.add_argument(
        "--heads",
        default="all",
        help='KV-head selection, e.g. "all", "0", "0,3", or "0-3".',
    )
    parser.add_argument("--similarity_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--similarity_dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--summary_percentiles", default=DEFAULT_PERCENTILES)
    parser.add_argument(
        "--summary_sample_size",
        type=int,
        default=1_000_000,
        help="Use at most this many values for percentile columns. Use 0 for exact percentiles.",
    )
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument(
        "--plot_max_tokens",
        type=int,
        default=5000,
        help="Downsample plotted matrices above this token count. Use 0 to plot every token.",
    )
    parser.add_argument("--figure_size", type=float, default=7.5)
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--save_similarity_tensors", type=str2bool, default=False)
    parser.add_argument("--saved_matrix_dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--write_token_csv", type=str2bool, default=True)
    parser.add_argument(
        "--histogram_bins",
        type=int,
        default=200,
        help="Number of bins for cosine value distribution CSVs. Use 0 to skip histograms.",
    )
    parser.add_argument("--histogram_min", type=float, default=-1.0)
    parser.add_argument("--histogram_max", type=float, default=1.0)
    parser.add_argument("--compute_pairwise_distances", type=str2bool, default=True)
    parser.add_argument(
        "--distance_cache_types",
        default="k,v",
        help='Cache types for pairwise L2 distance analysis, e.g. "k", "v", or "k,v".',
    )
    parser.add_argument(
        "--distance_bins",
        type=int,
        default=200,
        help="Number of bins for pairwise L2 distance distribution CSVs. Use 0 to skip distance histograms.",
    )
    parser.add_argument("--distance_min", type=float, default=0.0)
    parser.add_argument(
        "--distance_max",
        type=float,
        default=0.0,
        help="Right edge for distance histograms. Use 0 to infer the max from selected K/V heads.",
    )
    parser.add_argument("--compute_top_p_previous_distances", type=str2bool, default=True)
    parser.add_argument(
        "--top_p_previous_cache_types",
        default="k",
        help='Cache types for previous-token top-p sequence distance analysis, e.g. "k", "v", or "k,v".',
    )
    parser.add_argument(
        "--top_p_previous_count",
        type=int,
        default=5,
        help="For each token i, select this many most-similar previous vectors. If fewer exist, select all.",
    )
    parser.add_argument("--save_top_p_previous_token_rows", type=str2bool, default=True)
    return parser.parse_args()


def parse_percentiles(value: str) -> list[float]:
    percentiles = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not percentiles:
        raise ValueError("At least one percentile is required.")
    for percentile in percentiles:
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError(f"Percentile must be in [0, 100], got {percentile}.")
    return sorted(percentiles)


def percentile_field(percentile: float) -> str:
    if float(percentile).is_integer():
        return f"p{int(percentile)}"
    return "p" + str(percentile).replace(".", "_")


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

    if not selected:
        raise ValueError(f"No {name} selected from spec: {spec!r}")
    invalid = sorted(index for index in selected if index < 0 or index >= max_count)
    if invalid:
        raise ValueError(f"{name} out of range 0..{max_count - 1}: {invalid}")
    return sorted(selected)


def parse_cache_types(value: str) -> list[str]:
    cache_types = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not cache_types:
        raise ValueError("At least one cache type is required.")
    invalid = [item for item in cache_types if item not in {"k", "v"}]
    if invalid:
        raise ValueError(f'Unsupported cache types: {invalid}. Expected "k" and/or "v".')
    return list(dict.fromkeys(cache_types))


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


def named_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_similarity_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--similarity_device cuda requested, but CUDA is not available.")
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


def extract_value_tensors(past_key_values: Any) -> list[torch.Tensor]:
    if hasattr(past_key_values, "value_cache"):
        return list(past_key_values.value_cache)

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        return [layer_cache[1] for layer_cache in legacy_cache]

    if isinstance(past_key_values, (list, tuple)):
        if past_key_values and isinstance(past_key_values[0], (list, tuple)):
            return [layer_cache[1] for layer_cache in past_key_values]

    if hasattr(past_key_values, "layers"):
        value_tensors: list[torch.Tensor] = []
        for layer_cache in past_key_values.layers:
            for attr_name in ("values", "value_cache", "value_states"):
                if hasattr(layer_cache, attr_name):
                    value_tensors.append(getattr(layer_cache, attr_name))
                    break
        if value_tensors:
            return value_tensors

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def key_tensor_to_head_token_dim(
    key_tensor: torch.Tensor,
    expected_heads: int | None,
) -> torch.Tensor:
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
        dim1, dim2, head_dim = key.shape
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


def token_piece(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.convert_ids_to_tokens([token_id])[0]
    except Exception:
        return ""


def token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return ""


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_tokens_csv(path: Path, tokenizer: Any, token_ids: list[int]) -> None:
    fields = ["token_index", "token_id", "token_piece", "token_text"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, token_id in enumerate(token_ids):
            writer.writerow(
                {
                    "token_index": idx,
                    "token_id": int(token_id),
                    "token_piece": token_piece(tokenizer, int(token_id)),
                    "token_text": token_text(tokenizer, int(token_id)),
                }
            )


def write_profile_timings(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["chunk", "start_token", "end_token_exclusive", "token_count", "seconds"]
    write_csv(path, rows, fields)


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
            raise RuntimeError("Model did not return past_key_values. Check model/config use_cache support.")

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


def stats_for_values(
    prefix: str,
    values: torch.Tensor,
    percentiles: list[float],
    sample_size: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        row: dict[str, Any] = {
            f"{prefix}_count": 0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_rms": 0.0,
            f"{prefix}_percentile_source": "none",
        }
        for percentile in percentiles:
            row[f"{prefix}_{percentile_field(percentile)}"] = 0.0
        return row

    mean = flat.mean()
    std = flat.std(unbiased=False)
    row = {
        f"{prefix}_count": int(flat.numel()),
        f"{prefix}_mean": float(mean),
        f"{prefix}_std": float(std),
        f"{prefix}_min": float(flat.min()),
        f"{prefix}_max": float(flat.max()),
        f"{prefix}_rms": float(flat.square().mean().sqrt()),
        f"{prefix}_percentile_source": "exact",
    }

    if sample_size > 0 and flat.numel() > sample_size:
        indices = torch.randint(int(flat.numel()), (sample_size,), generator=generator, dtype=torch.long)
        quantile_values = flat[indices]
        row[f"{prefix}_percentile_source"] = f"sampled_{sample_size}"
    else:
        quantile_values = flat

    quantiles = torch.quantile(
        quantile_values,
        torch.tensor([p / 100.0 for p in percentiles], dtype=torch.float32),
    )
    for percentile, quantile in zip(percentiles, quantiles):
        row[f"{prefix}_{percentile_field(percentile)}"] = float(quantile)
    return row


def summarize_similarity_matrix(
    matrix: torch.Tensor,
    percentiles: list[float],
    sample_size: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    diagonal = matrix.diagonal().float()
    row: dict[str, Any] = {
        "diag_mean": float(diagonal.mean()),
        "diag_min": float(diagonal.min()),
        "diag_max": float(diagonal.max()),
    }
    row.update(stats_for_values("all", matrix, percentiles, sample_size, generator))

    offdiag = matrix.clone()
    offdiag.fill_diagonal_(float("nan"))
    row.update(stats_for_values("offdiag", offdiag, percentiles, sample_size, generator))
    return row


def stat_fieldnames(prefix: str, percentiles: list[float]) -> list[str]:
    return [
        f"{prefix}_count",
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_rms",
        f"{prefix}_percentile_source",
    ] + [f"{prefix}_{percentile_field(percentile)}" for percentile in percentiles]


def build_summary_fieldnames(percentiles: list[float]) -> list[str]:
    return [
        "layer",
        "head",
        "tokens",
        "head_dim",
        "similarity_device",
        "similarity_dtype",
        "seconds",
        "plot_path",
        "plot_stride",
        "plotted_tokens",
        "tensor_path",
        "diag_mean",
        "diag_min",
        "diag_max",
    ] + stat_fieldnames("all", percentiles) + stat_fieldnames("offdiag", percentiles)


def histogram_fieldnames() -> list[str]:
    return [
        "layer",
        "head",
        "scope",
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_center",
        "count",
        "total",
        "probability",
    ]


def finite_values(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().float().reshape(-1)
    return flat[torch.isfinite(flat)]


def matrix_values_for_scope(matrix: torch.Tensor, scope: str) -> torch.Tensor:
    if scope == "all":
        return finite_values(matrix)
    if scope == "offdiag":
        values = matrix.clone()
        values.fill_diagonal_(float("nan"))
        return finite_values(values)
    raise ValueError(f"Unsupported histogram scope: {scope}")


def histogram_counts(values: torch.Tensor, bins: int, min_value: float, max_value: float) -> tuple[torch.Tensor, int]:
    finite = finite_values(values)
    total = int(finite.numel())
    if total == 0:
        return torch.zeros(bins, dtype=torch.long), 0
    counts = torch.histc(finite, bins=bins, min=min_value, max=max_value).to(torch.long)
    return counts.cpu(), total


def histogram_rows(
    counts: torch.Tensor,
    total: int,
    bins: int,
    min_value: float,
    max_value: float,
    layer: int | str,
    head: int | str,
    scope: str,
) -> list[dict[str, Any]]:
    width = (max_value - min_value) / bins
    rows: list[dict[str, Any]] = []
    for bin_index, count_tensor in enumerate(counts.tolist()):
        left = min_value + width * bin_index
        right = min_value + width * (bin_index + 1)
        count = int(count_tensor)
        rows.append(
            {
                "layer": layer,
                "head": head,
                "scope": scope,
                "bin_index": bin_index,
                "bin_left": left,
                "bin_right": right,
                "bin_center": (left + right) / 2.0,
                "count": count,
                "total": total,
                "probability": (count / total) if total else 0.0,
            }
        )
    return rows


def distance_summary_fieldnames(percentiles: list[float]) -> list[str]:
    return [
        "cache_type",
        "layer",
        "head",
        "tokens",
        "head_dim",
        "distance_device",
        "distance_dtype",
        "seconds",
    ] + stat_fieldnames("all", percentiles) + stat_fieldnames("offdiag", percentiles)


def distance_histogram_fieldnames() -> list[str]:
    return [
        "cache_type",
        "layer",
        "head",
        "scope",
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_center",
        "count",
        "total",
        "probability",
    ]


def distance_histogram_rows(
    counts: torch.Tensor,
    total: int,
    bins: int,
    min_value: float,
    max_value: float,
    cache_type: str,
    layer: int | str,
    head: int | str,
    scope: str,
) -> list[dict[str, Any]]:
    rows = histogram_rows(counts, total, bins, min_value, max_value, layer, head, scope)
    for row in rows:
        row["cache_type"] = cache_type
    return rows


def summarize_pairwise_distance_matrix(
    matrix: torch.Tensor,
    percentiles: list[float],
    sample_size: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    row = stats_for_values("all", matrix, percentiles, sample_size, generator)
    offdiag = matrix.clone()
    offdiag.fill_diagonal_(float("nan"))
    row.update(stats_for_values("offdiag", offdiag, percentiles, sample_size, generator))
    return row


def top_p_previous_summary_fieldnames(percentiles: list[float]) -> list[str]:
    return [
        "cache_type",
        "layer",
        "head",
        "tokens",
        "head_dim",
        "top_p",
        "token_count_with_previous",
    ] + stat_fieldnames("mean_index_distance", percentiles)


def top_p_previous_token_fieldnames() -> list[str]:
    return [
        "cache_type",
        "layer",
        "head",
        "token_index",
        "available_previous",
        "selected_count",
        "mean_index_distance",
        "min_index_distance",
        "max_index_distance",
        "selected_indices",
        "selected_similarities",
    ]


def top_p_previous_distance_rows(
    matrix: torch.Tensor,
    cache_type: str,
    layer: int,
    head: int,
    tokens: int,
    head_dim: int,
    top_p: int,
    percentiles: list[float],
    sample_size: int,
    generator: torch.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    token_rows: list[dict[str, Any]] = []
    mean_distances: list[float] = []

    for token_index in range(tokens):
        available_previous = token_index
        if available_previous == 0:
            token_rows.append(
                {
                    "cache_type": cache_type,
                    "layer": layer,
                    "head": head,
                    "token_index": token_index,
                    "available_previous": 0,
                    "selected_count": 0,
                    "mean_index_distance": "",
                    "min_index_distance": "",
                    "max_index_distance": "",
                    "selected_indices": "",
                    "selected_similarities": "",
                }
            )
            continue

        selected_count = min(top_p, available_previous)
        similarities, indices = torch.topk(matrix[token_index, :token_index], k=selected_count)
        index_distances = token_index - indices
        mean_distance = float(index_distances.float().mean())
        mean_distances.append(mean_distance)
        token_rows.append(
            {
                "cache_type": cache_type,
                "layer": layer,
                "head": head,
                "token_index": token_index,
                "available_previous": available_previous,
                "selected_count": selected_count,
                "mean_index_distance": mean_distance,
                "min_index_distance": int(index_distances.min()),
                "max_index_distance": int(index_distances.max()),
                "selected_indices": ";".join(str(int(item)) for item in indices.tolist()),
                "selected_similarities": ";".join(f"{float(item):.8g}" for item in similarities.tolist()),
            }
        )

    values = torch.tensor(mean_distances, dtype=torch.float32)
    summary: dict[str, Any] = {
        "cache_type": cache_type,
        "layer": layer,
        "head": head,
        "tokens": tokens,
        "head_dim": head_dim,
        "top_p": top_p,
        "token_count_with_previous": int(values.numel()),
    }
    summary.update(
        stats_for_values(
            "mean_index_distance",
            values,
            percentiles,
            sample_size,
            generator,
        )
    )
    return summary, token_rows


@torch.inference_mode()
def compute_cosine_matrix(
    vectors: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-12,
) -> torch.Tensor:
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    working = vectors.to(device=device, dtype=dtype)
    normalized = F.normalize(working, p=2, dim=-1, eps=eps)
    matrix = normalized @ normalized.transpose(0, 1)
    matrix = matrix.clamp(min=-1.0, max=1.0).float().cpu()
    del working, normalized
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return matrix


@torch.inference_mode()
def compute_pairwise_l2_distance_matrix(
    vectors: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    working = vectors.to(device=device, dtype=dtype)
    matrix = torch.cdist(working, working, p=2).clamp_min(0.0).float().cpu()
    del working
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return matrix


def save_similarity_heatmap(
    matrix: torch.Tensor,
    path: Path,
    layer_idx: int,
    head_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = int(matrix.shape[0])
    stride = 1
    if args.plot_max_tokens > 0 and tokens > args.plot_max_tokens:
        stride = math.ceil(tokens / args.plot_max_tokens)
    plot_matrix = matrix[::stride, ::stride].float()
    plotted_tokens = int(plot_matrix.shape[0])

    fig, ax = plt.subplots(figsize=(args.figure_size, args.figure_size), dpi=args.plot_dpi)
    image = ax.imshow(
        plot_matrix.numpy(),
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(f"Layer {layer_idx} KV head {head_idx} K-cache cosine")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Token index")
    if plotted_tokens > 1:
        tick_count = min(6, plotted_tokens)
        tick_positions = torch.linspace(0, plotted_tokens - 1, tick_count).round().long().tolist()
        tick_labels = [str(min(tokens - 1, int(pos) * stride)) for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cosine")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "plot_path": str(path),
        "plot_stride": stride,
        "plotted_tokens": plotted_tokens,
    }


def save_layer_head_metric_heatmap(
    rows: list[dict[str, Any]],
    output_dir: Path,
    metric: str,
    title: str,
    cmap: str,
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        return None
    max_layer = max(int(row["layer"]) for row in rows)
    max_head = max(int(row["head"]) for row in rows)
    matrix = np.full((max_layer + 1, max_head + 1), np.nan, dtype=np.float32)
    for row in rows:
        matrix[int(row["layer"]), int(row["head"])] = float(row[metric])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("KV head")
    ax.set_ylabel("Layer")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = output_dir / f"layer_head_{metric}_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def save_similarity_tensor(
    matrix: torch.Tensor,
    path: Path,
    dtype_name: str,
    metadata: dict[str, Any],
) -> None:
    dtype = named_torch_dtype(dtype_name)
    torch.save(
        {
            "metadata": metadata,
            "similarity": matrix.to(dtype),
        },
        path,
    )


def analyze_pairwise_distances(
    cache_tensors_by_type: dict[str, list[torch.Tensor]],
    layer_indices: list[int],
    expected_heads: int | None,
    distance_device: torch.device,
    distance_dtype: torch.dtype,
    percentiles: list[float],
    args: argparse.Namespace,
    output_dir: Path,
    generator: torch.Generator,
) -> dict[str, str | None]:
    summary_rows: list[dict[str, Any]] = []
    max_distance = float(args.distance_max)

    for cache_type, cache_tensors in cache_tensors_by_type.items():
        for layer_idx in layer_indices:
            cache_by_head = key_tensor_to_head_token_dim(cache_tensors[layer_idx], expected_heads)
            kv_heads, tokens, head_dim = cache_by_head.shape
            head_indices = parse_index_spec(args.heads, int(kv_heads), "heads")
            for head_idx in head_indices:
                print(f"computing {cache_type.upper()} pairwise L2 distances for layer {layer_idx}, head {head_idx}", flush=True)
                started = time.perf_counter()
                matrix = compute_pairwise_l2_distance_matrix(
                    cache_by_head[head_idx],
                    distance_device,
                    distance_dtype,
                )
                seconds = time.perf_counter() - started
                stats = summarize_pairwise_distance_matrix(
                    matrix,
                    percentiles,
                    args.summary_sample_size,
                    generator,
                )
                if args.distance_max <= 0.0:
                    max_distance = max(max_distance, float(stats["all_max"]))
                row: dict[str, Any] = {
                    "cache_type": cache_type,
                    "layer": layer_idx,
                    "head": head_idx,
                    "tokens": int(tokens),
                    "head_dim": int(head_dim),
                    "distance_device": str(distance_device),
                    "distance_dtype": str(distance_dtype),
                    "seconds": seconds,
                }
                row.update(stats)
                summary_rows.append(row)
                del matrix
                if distance_device.type == "cuda":
                    torch.cuda.empty_cache()
            del cache_by_head

    summary_path = output_dir / "distance_summary_by_head.csv"
    write_csv(summary_path, summary_rows, distance_summary_fieldnames(percentiles))

    by_head_path: Path | None = None
    global_path: Path | None = None
    if args.distance_bins > 0 and summary_rows:
        if max_distance <= args.distance_min:
            max_distance = args.distance_min + 1.0
        by_head_rows: list[dict[str, Any]] = []
        global_histograms: dict[str, dict[str, Any]] = {}
        for cache_type in cache_tensors_by_type:
            for scope in ("all", "offdiag"):
                global_histograms[f"{cache_type}:{scope}"] = {
                    "cache_type": cache_type,
                    "scope": scope,
                    "counts": torch.zeros(args.distance_bins, dtype=torch.long),
                    "total": 0,
                }

        for cache_type, cache_tensors in cache_tensors_by_type.items():
            for layer_idx in layer_indices:
                cache_by_head = key_tensor_to_head_token_dim(cache_tensors[layer_idx], expected_heads)
                kv_heads = int(cache_by_head.shape[0])
                head_indices = parse_index_spec(args.heads, kv_heads, "heads")
                for head_idx in head_indices:
                    matrix = compute_pairwise_l2_distance_matrix(
                        cache_by_head[head_idx],
                        distance_device,
                        distance_dtype,
                    )
                    for scope in ("all", "offdiag"):
                        scope_values = matrix_values_for_scope(matrix, scope)
                        counts, total = histogram_counts(
                            scope_values,
                            args.distance_bins,
                            args.distance_min,
                            max_distance,
                        )
                        by_head_rows.extend(
                            distance_histogram_rows(
                                counts,
                                total,
                                args.distance_bins,
                                args.distance_min,
                                max_distance,
                                cache_type,
                                layer_idx,
                                head_idx,
                                scope,
                            )
                        )
                        global_key = f"{cache_type}:{scope}"
                        global_histograms[global_key]["counts"] += counts
                        global_histograms[global_key]["total"] += total
                    del matrix
                    if distance_device.type == "cuda":
                        torch.cuda.empty_cache()
                del cache_by_head

        by_head_path = output_dir / "distance_histogram_by_head.csv"
        global_path = output_dir / "distance_histogram_global.csv"
        write_csv(by_head_path, by_head_rows, distance_histogram_fieldnames())

        global_rows: list[dict[str, Any]] = []
        for payload in global_histograms.values():
            global_rows.extend(
                distance_histogram_rows(
                    payload["counts"],
                    int(payload["total"]),
                    args.distance_bins,
                    args.distance_min,
                    max_distance,
                    str(payload["cache_type"]),
                    "all",
                    "all",
                    str(payload["scope"]),
                )
            )
        write_csv(global_path, global_rows, distance_histogram_fieldnames())

    return {
        "distance_summary_by_head": str(summary_path),
        "distance_histogram_by_head": str(by_head_path) if by_head_path is not None else None,
        "distance_histogram_global": str(global_path) if global_path is not None else None,
    }


def analyze_top_p_previous_for_cache_type(
    cache_type: str,
    cache_tensors: list[torch.Tensor],
    layer_indices: list[int],
    expected_heads: int | None,
    similarity_device: torch.device,
    similarity_dtype: torch.dtype,
    percentiles: list[float],
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for layer_idx in layer_indices:
        cache_by_head = key_tensor_to_head_token_dim(cache_tensors[layer_idx], expected_heads)
        kv_heads, tokens, head_dim = cache_by_head.shape
        head_indices = parse_index_spec(args.heads, int(kv_heads), "heads")
        for head_idx in head_indices:
            print(
                f"computing {cache_type.upper()} top-{args.top_p_previous_count} previous-neighbor distances "
                f"for layer {layer_idx}, head {head_idx}",
                flush=True,
            )
            matrix = compute_cosine_matrix(cache_by_head[head_idx], similarity_device, similarity_dtype)
            summary, rows = top_p_previous_distance_rows(
                matrix,
                cache_type,
                layer_idx,
                head_idx,
                int(tokens),
                int(head_dim),
                args.top_p_previous_count,
                percentiles,
                args.summary_sample_size,
                generator,
            )
            summary_rows.append(summary)
            token_rows.extend(rows)
            del matrix
            if similarity_device.type == "cuda":
                torch.cuda.empty_cache()
        del cache_by_head
    return summary_rows, token_rows


def main() -> None:
    args = parse_args()
    percentiles = parse_percentiles(args.summary_percentiles)
    text_path = Path(args.text_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not text_path.exists():
        raise FileNotFoundError(f"text_path does not exist: {text_path}")
    if args.max_tokens <= 0:
        raise ValueError("--max_tokens must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    if args.plot_dpi <= 0:
        raise ValueError("--plot_dpi must be positive.")
    if args.histogram_bins < 0:
        raise ValueError("--histogram_bins must be non-negative.")
    if args.histogram_bins > 0 and args.histogram_max <= args.histogram_min:
        raise ValueError("--histogram_max must be greater than --histogram_min.")
    if args.distance_bins < 0:
        raise ValueError("--distance_bins must be non-negative.")
    if args.distance_min < 0.0:
        raise ValueError("--distance_min must be non-negative for L2 distances.")
    if args.distance_max > 0.0 and args.distance_max <= args.distance_min:
        raise ValueError("--distance_max must be greater than --distance_min when set.")
    if args.top_p_previous_count <= 0:
        raise ValueError("--top_p_previous_count must be positive.")

    print(f"reading text: {text_path}", flush=True)
    text = read_text_prefix(text_path, args.max_chars)
    if not text.strip():
        raise ValueError(f"No usable text read from {text_path}")

    print(f"loading tokenizer: {args.model_name_or_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    if args.require_max_tokens and len(token_ids) < args.max_tokens:
        raise ValueError(
            f"Tokenization produced {len(token_ids)} tokens, fewer than --max_tokens {args.max_tokens}."
        )
    token_ids = token_ids[: args.max_tokens]
    if len(token_ids) < 2:
        raise ValueError("Tokenization produced fewer than two tokens.")
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)
    print(f"using tokens: {input_ids.shape[1]} (max_tokens={args.max_tokens})", flush=True)

    if args.write_token_csv:
        write_tokens_csv(output_dir / "tokens.csv", tokenizer, token_ids)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": model_dtype,
    }
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation

    print(f"loading causal LM: {args.model_name_or_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True

    input_device = pick_input_device(model, requested_device)
    past_key_values, timing_rows = build_k_cache(model, input_ids, args.chunk_size, input_device)
    write_profile_timings(output_dir / "profile_timings.csv", timing_rows)

    key_tensors = extract_key_tensors(past_key_values)
    if not key_tensors:
        raise RuntimeError("No key tensors were extracted from past_key_values.")
    distance_cache_types = parse_cache_types(args.distance_cache_types)
    top_p_previous_cache_types = parse_cache_types(args.top_p_previous_cache_types)
    value_tensors: list[torch.Tensor] | None = None
    needs_value_tensors = (
        (args.compute_pairwise_distances and "v" in distance_cache_types)
        or (args.compute_top_p_previous_distances and "v" in top_p_previous_cache_types)
    )
    if needs_value_tensors:
        value_tensors = extract_value_tensors(past_key_values)
        if len(key_tensors) != len(value_tensors):
            raise RuntimeError(f"K/V layer count mismatch: {len(key_tensors)} keys vs {len(value_tensors)} values")

    expected_heads = getattr(model.config, "num_key_value_heads", None)
    layer_indices = parse_index_spec(args.layers, len(key_tensors), "layers")
    similarity_device = resolve_similarity_device(args.similarity_device)
    similarity_dtype = named_torch_dtype(args.similarity_dtype)
    if similarity_device.type == "cpu" and similarity_dtype != torch.float32:
        print("CPU similarity matmul uses float32 regardless of --similarity_dtype.", flush=True)
        similarity_dtype = torch.float32

    plots_dir = output_dir / "plots"
    tensors_dir = output_dir / "similarity_tensors"
    if args.make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    if args.save_similarity_tensors:
        tensors_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    histogram_by_head_rows: list[dict[str, Any]] = []
    top_p_previous_summary_rows: list[dict[str, Any]] = []
    top_p_previous_token_rows: list[dict[str, Any]] = []
    cache_shapes: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.sample_seed)
    global_histograms: dict[str, dict[str, Any]] = {}
    if args.histogram_bins > 0:
        for scope in ("all", "offdiag"):
            global_histograms[scope] = {
                "counts": torch.zeros(args.histogram_bins, dtype=torch.long),
                "total": 0,
            }

    for layer_idx in layer_indices:
        key_by_head = key_tensor_to_head_token_dim(key_tensors[layer_idx], expected_heads)
        kv_heads, tokens, head_dim = key_by_head.shape
        cache_shapes.append(
            {
                "layer": layer_idx,
                "kv_heads": int(kv_heads),
                "tokens": int(tokens),
                "head_dim": int(head_dim),
                "raw_shape": list(key_tensors[layer_idx].shape),
                "raw_dtype": str(key_tensors[layer_idx].dtype),
                "raw_device": str(key_tensors[layer_idx].device),
            }
        )
        head_indices = parse_index_spec(args.heads, int(kv_heads), "heads")
        print(
            f"layer {layer_idx}: kv_heads={kv_heads} tokens={tokens} head_dim={head_dim}",
            flush=True,
        )

        for head_idx in head_indices:
            print(f"computing cosine matrix for layer {layer_idx}, head {head_idx}", flush=True)
            started = time.perf_counter()
            matrix = compute_cosine_matrix(
                key_by_head[head_idx],
                similarity_device,
                similarity_dtype,
            )
            seconds = time.perf_counter() - started

            row: dict[str, Any] = {
                "layer": layer_idx,
                "head": head_idx,
                "tokens": int(tokens),
                "head_dim": int(head_dim),
                "similarity_device": str(similarity_device),
                "similarity_dtype": str(similarity_dtype),
                "seconds": seconds,
                "plot_path": None,
                "plot_stride": None,
                "plotted_tokens": None,
                "tensor_path": None,
            }
            row.update(
                summarize_similarity_matrix(
                    matrix,
                    percentiles,
                    args.summary_sample_size,
                    generator,
                )
            )

            if args.histogram_bins > 0:
                for scope in ("all", "offdiag"):
                    scope_values = matrix_values_for_scope(matrix, scope)
                    counts, total = histogram_counts(
                        scope_values,
                        args.histogram_bins,
                        args.histogram_min,
                        args.histogram_max,
                    )
                    histogram_by_head_rows.extend(
                        histogram_rows(
                            counts,
                            total,
                            args.histogram_bins,
                            args.histogram_min,
                            args.histogram_max,
                            layer_idx,
                            head_idx,
                            scope,
                        )
                    )
                    global_histograms[scope]["counts"] += counts
                    global_histograms[scope]["total"] += total

            if args.compute_top_p_previous_distances and "k" in top_p_previous_cache_types:
                top_p_summary, top_p_rows = top_p_previous_distance_rows(
                    matrix,
                    "k",
                    layer_idx,
                    head_idx,
                    int(tokens),
                    int(head_dim),
                    args.top_p_previous_count,
                    percentiles,
                    args.summary_sample_size,
                    generator,
                )
                top_p_previous_summary_rows.append(top_p_summary)
                if args.save_top_p_previous_token_rows:
                    top_p_previous_token_rows.extend(top_p_rows)

            if args.make_plots:
                plot_path = plots_dir / f"layer_{layer_idx:02d}_head_{head_idx:02d}_cosine.png"
                row.update(save_similarity_heatmap(matrix, plot_path, layer_idx, head_idx, args))

            if args.save_similarity_tensors:
                tensor_path = tensors_dir / f"layer_{layer_idx:02d}_head_{head_idx:02d}_cosine.pt"
                save_similarity_tensor(matrix, tensor_path, args.saved_matrix_dtype, row)
                row["tensor_path"] = str(tensor_path)

            summary_rows.append(row)
            del matrix
            if similarity_device.type == "cuda":
                torch.cuda.empty_cache()

        del key_by_head

    write_csv(output_dir / "summary_by_head.csv", summary_rows, build_summary_fieldnames(percentiles))
    if args.histogram_bins > 0:
        write_csv(output_dir / "histogram_by_head.csv", histogram_by_head_rows, histogram_fieldnames())
        histogram_global_rows: list[dict[str, Any]] = []
        for scope, payload in global_histograms.items():
            histogram_global_rows.extend(
                histogram_rows(
                    payload["counts"],
                    int(payload["total"]),
                    args.histogram_bins,
                    args.histogram_min,
                    args.histogram_max,
                    "all",
                    "all",
                    scope,
                )
            )
        write_csv(output_dir / "histogram_global.csv", histogram_global_rows, histogram_fieldnames())

    top_p_previous_summary_path: Path | None = None
    top_p_previous_token_path: Path | None = None
    if args.compute_top_p_previous_distances:
        if "v" in top_p_previous_cache_types:
            if value_tensors is None:
                value_tensors = extract_value_tensors(past_key_values)
            v_summary_rows, v_token_rows = analyze_top_p_previous_for_cache_type(
                "v",
                value_tensors,
                layer_indices,
                expected_heads,
                similarity_device,
                similarity_dtype,
                percentiles,
                args,
                generator,
            )
            top_p_previous_summary_rows.extend(v_summary_rows)
            if args.save_top_p_previous_token_rows:
                top_p_previous_token_rows.extend(v_token_rows)

        top_p_previous_summary_path = output_dir / "top_p_previous_distance_summary_by_head.csv"
        write_csv(
            top_p_previous_summary_path,
            top_p_previous_summary_rows,
            top_p_previous_summary_fieldnames(percentiles),
        )
        if args.save_top_p_previous_token_rows:
            top_p_previous_token_path = output_dir / "top_p_previous_distance_by_token.csv"
            write_csv(
                top_p_previous_token_path,
                top_p_previous_token_rows,
                top_p_previous_token_fieldnames(),
            )

    aggregate_plot_paths: dict[str, str | None] = {}
    if args.make_plots:
        aggregate_plot_paths["layer_head_offdiag_mean_heatmap"] = save_layer_head_metric_heatmap(
            summary_rows,
            plots_dir,
            "offdiag_mean",
            "Mean off-diagonal K-cache cosine by layer/head",
            args.cmap,
        )
        aggregate_plot_paths["layer_head_offdiag_std_heatmap"] = save_layer_head_metric_heatmap(
            summary_rows,
            plots_dir,
            "offdiag_std",
            "Off-diagonal K-cache cosine std by layer/head",
            args.cmap,
        )

    distance_paths: dict[str, str | None] = {}
    if args.compute_pairwise_distances:
        cache_tensors_by_type: dict[str, list[torch.Tensor]] = {}
        if "k" in distance_cache_types:
            cache_tensors_by_type["k"] = key_tensors
        if "v" in distance_cache_types:
            if value_tensors is None:
                value_tensors = extract_value_tensors(past_key_values)
            cache_tensors_by_type["v"] = value_tensors
        distance_paths = analyze_pairwise_distances(
            cache_tensors_by_type,
            layer_indices,
            expected_heads,
            similarity_device,
            similarity_dtype,
            percentiles,
            args,
            output_dir,
            generator,
        )

    payload = {
        "args": vars(args),
        "resolved": {
            "tokens": int(input_ids.shape[1]),
            "text_path": str(text_path),
            "model_name_or_path": args.model_name_or_path,
            "layers": layer_indices,
            "summary_percentiles": percentiles,
            "similarity_device": str(similarity_device),
            "similarity_dtype": str(similarity_dtype),
            "cache_shapes": cache_shapes,
        },
        "paths": {
            "tokens": str(output_dir / "tokens.csv") if args.write_token_csv else None,
            "profile_timings": str(output_dir / "profile_timings.csv"),
            "summary_by_head": str(output_dir / "summary_by_head.csv"),
            "histogram_by_head": str(output_dir / "histogram_by_head.csv") if args.histogram_bins > 0 else None,
            "histogram_global": str(output_dir / "histogram_global.csv") if args.histogram_bins > 0 else None,
            "top_p_previous_distance_summary_by_head": (
                str(top_p_previous_summary_path) if top_p_previous_summary_path is not None else None
            ),
            "top_p_previous_distance_by_token": (
                str(top_p_previous_token_path) if top_p_previous_token_path is not None else None
            ),
            **distance_paths,
            "plots_dir": str(plots_dir) if args.make_plots else None,
            "similarity_tensors_dir": str(tensors_dir) if args.save_similarity_tensors else None,
            "aggregate_plots": aggregate_plot_paths,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
