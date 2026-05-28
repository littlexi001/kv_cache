from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-8B"
DEFAULT_TEXT_PATH = (
    "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"
)
DEFAULT_LENGTHS = "1k,10k,100k"
DEFAULT_PERCENTILES = "1,5,25,50,75,95,99"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile Qwen KV caches at multiple prefix lengths, compute per-dim "
            "statistics, run SVD, and compare corresponding singular vectors."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/kvcache_svd_profile")
    parser.add_argument("--cache_lengths", default=DEFAULT_LENGTHS)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument(
        "--max_chars",
        type=int,
        default=0,
        help="Read at most this many characters from text_path. Use 0 to read the full file.",
    )
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument(
        "--require_max_length",
        type=str2bool,
        default=True,
        help="Fail if tokenization produces fewer tokens than the largest requested cache length.",
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
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument(
        "--cache_kinds",
        default="key,value",
        help='Comma-separated cache kinds to analyze: "key", "value", or "key,value".',
    )
    parser.add_argument("--percentiles", default=DEFAULT_PERCENTILES)
    parser.add_argument("--svd_device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument(
        "--svd_devices",
        default="",
        help=(
            "Comma-separated devices for parallel SVD, e.g. cuda:0,cuda:1,...,cuda:7. "
            "When set, this overrides --svd_device for SVD work."
        ),
    )
    parser.add_argument("--svd_dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--max_svd_rank",
        type=int,
        default=128,
        help="Keep at most this many singular vectors/values for comparison and plots.",
    )
    parser.add_argument("--svd_full_matrices", type=str2bool, default=False)
    parser.add_argument("--save_svd_tensors", type=str2bool, default=False)
    parser.add_argument(
        "--offload_cache_to_cpu",
        type=str2bool,
        default=True,
        help="Move KV cache tensors to CPU and release the model before SVD.",
    )
    parser.add_argument("--write_dimension_stats", type=str2bool, default=True)
    parser.add_argument("--write_token_norm_stats", type=str2bool, default=True)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--plot_dpi", type=int, default=160)
    parser.add_argument("--sample_seed", type=int, default=1234)
    return parser.parse_args()


def parse_count_token(value: str) -> int:
    token = value.strip().lower().replace("_", "")
    multiplier = 1
    if token.endswith("k"):
        multiplier = 1_000
        token = token[:-1]
    elif token.endswith("m"):
        multiplier = 1_000_000
        token = token[:-1]
    if not token:
        raise ValueError(f"Invalid count value: {value!r}")
    return int(float(token) * multiplier)


def parse_positive_ints(value: str, name: str) -> list[int]:
    parsed = [parse_count_token(item) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError(f"At least one {name} value is required.")
    invalid = [item for item in parsed if item <= 0]
    if invalid:
        raise ValueError(f"{name} values must be positive, got {invalid}.")
    return sorted(set(parsed))


def parse_percentiles(value: str) -> list[float]:
    percentiles = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not percentiles:
        raise ValueError("At least one percentile is required.")
    for percentile in percentiles:
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError(f"Percentile must be in [0, 100], got {percentile}.")
    return sorted(percentiles)


def parse_cache_kinds(value: str) -> list[str]:
    kinds = [item.strip().lower() for item in value.split(",") if item.strip()]
    valid = {"key", "value"}
    unknown = sorted(set(kinds) - valid)
    if unknown:
        raise ValueError(f"Unknown cache kinds: {unknown}. Valid kinds: {sorted(valid)}")
    if not kinds:
        raise ValueError("At least one cache kind is required.")
    return list(dict.fromkeys(kinds))


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
    if not selected:
        raise ValueError(f"No {name} selected from spec: {spec!r}")
    return sorted(selected)


def percentile_field(percentile: float) -> str:
    if float(percentile).is_integer():
        return f"p{int(percentile)}"
    return "p" + str(percentile).replace(".", "_")


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


def resolve_svd_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--svd_device cuda requested, but CUDA is not available.")
    return torch.device(name)


def resolve_svd_devices(args: argparse.Namespace) -> list[torch.device]:
    if args.svd_devices.strip():
        devices = [
            torch.device(item.strip())
            for item in args.svd_devices.split(",")
            if item.strip()
        ]
        if not devices:
            raise ValueError("--svd_devices was set but no devices were parsed.")
        if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
            raise RuntimeError("--svd_devices includes CUDA devices, but CUDA is not available.")
        return devices
    if args.svd_device == "auto" and torch.cuda.is_available():
        return [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]
    return [resolve_svd_device(args.svd_device)]


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


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


@torch.inference_mode()
def build_kv_cache(
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


def extract_layer_cache_tensors(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        return [(layer_cache[0], layer_cache[1]) for layer_cache in legacy_cache]

    if isinstance(past_key_values, (list, tuple)):
        if past_key_values and isinstance(past_key_values[0], (list, tuple)):
            return [(layer_cache[0], layer_cache[1]) for layer_cache in past_key_values]

    if hasattr(past_key_values, "layers"):
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_cache in past_key_values.layers:
            key_tensor = None
            value_tensor = None
            for attr_name in ("keys", "key_cache", "key_states"):
                if hasattr(layer_cache, attr_name):
                    key_tensor = getattr(layer_cache, attr_name)
                    break
            for attr_name in ("values", "value_cache", "value_states"):
                if hasattr(layer_cache, attr_name):
                    value_tensor = getattr(layer_cache, attr_name)
                    break
            if key_tensor is None or value_tensor is None:
                raise TypeError(f"Unsupported cache layer type: {type(layer_cache)!r}")
            pairs.append((key_tensor, value_tensor))
        if pairs:
            return pairs

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def offload_layer_cache_pairs_to_cpu(
    layer_cache_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    offloaded: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx, (key_tensor, value_tensor) in enumerate(layer_cache_pairs):
        print(f"offloading KV cache layer {layer_idx} to CPU", flush=True)
        offloaded.append((key_tensor.detach().cpu(), value_tensor.detach().cpu()))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return offloaded


def cache_tensor_to_head_token_dim(
    tensor: torch.Tensor,
    expected_heads: int | None,
) -> torch.Tensor:
    cache = tensor.detach()
    if cache.ndim == 4:
        batch, dim1, dim2, head_dim = cache.shape
        if expected_heads is not None and dim1 == expected_heads:
            by_head = cache.permute(1, 0, 2, 3).reshape(dim1, batch * dim2, head_dim)
        elif expected_heads is not None and dim2 == expected_heads:
            by_head = cache.permute(2, 0, 1, 3).reshape(dim2, batch * dim1, head_dim)
        elif dim1 <= dim2:
            by_head = cache.permute(1, 0, 2, 3).reshape(dim1, batch * dim2, head_dim)
        else:
            by_head = cache.permute(2, 0, 1, 3).reshape(dim2, batch * dim1, head_dim)
    elif cache.ndim == 3:
        dim1, dim2, _ = cache.shape
        if expected_heads is not None and dim1 == expected_heads:
            by_head = cache
        elif expected_heads is not None and dim2 == expected_heads:
            by_head = cache.permute(1, 0, 2)
        elif dim1 <= dim2:
            by_head = cache
        else:
            by_head = cache.permute(1, 0, 2)
    else:
        raise ValueError(f"Expected 3D or 4D cache tensor, got shape {tuple(cache.shape)}")
    return by_head.float().cpu()


def summarize_tensor(values: torch.Tensor, percentiles: list[float]) -> dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        row: dict[str, Any] = {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "rms": 0.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
        }
        for percentile in percentiles:
            row[percentile_field(percentile)] = 0.0
        return row

    quantiles = torch.quantile(
        flat,
        torch.tensor([p / 100.0 for p in percentiles], dtype=torch.float32),
    )
    row = {
        "count": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "rms": float(flat.square().mean().sqrt()),
        "mean_abs": float(flat.abs().mean()),
        "max_abs": float(flat.abs().max()),
    }
    for percentile, quantile in zip(percentiles, quantiles):
        row[percentile_field(percentile)] = float(quantile)
    return row


def dim_stat_rows(
    matrix: torch.Tensor,
    layer: int,
    head: int,
    cache_kind: str,
    cache_length: int,
    percentiles: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dim_l2 = torch.linalg.vector_norm(matrix.float(), ord=2, dim=0)
    for dim_idx in range(matrix.shape[1]):
        values = matrix[:, dim_idx]
        row: dict[str, Any] = {
            "cache_kind": cache_kind,
            "cache_length": cache_length,
            "layer": layer,
            "head": head,
            "dim": dim_idx,
            "dim_l2_norm": float(dim_l2[dim_idx]),
        }
        row.update(summarize_tensor(values, percentiles))
        rows.append(row)
    return rows


def head_norm_row(
    matrix: torch.Tensor,
    layer: int,
    head: int,
    cache_kind: str,
    cache_length: int,
    percentiles: list[float],
) -> dict[str, Any]:
    token_norms = torch.linalg.vector_norm(matrix.float(), ord=2, dim=1)
    row: dict[str, Any] = {
        "cache_kind": cache_kind,
        "cache_length": cache_length,
        "layer": layer,
        "head": head,
        "tokens": int(matrix.shape[0]),
        "head_dim": int(matrix.shape[1]),
    }
    row.update(summarize_tensor(token_norms, percentiles))
    return row


def run_svd(
    matrix: torch.Tensor,
    svd_device: torch.device,
    dtype: torch.dtype,
    full_matrices: bool,
    max_rank: int,
) -> dict[str, torch.Tensor | float]:
    working = matrix.to(device=svd_device, dtype=dtype)
    started = time.perf_counter()
    u, s, vh = torch.linalg.svd(working, full_matrices=full_matrices)
    seconds = time.perf_counter() - started
    keep = min(max_rank, int(s.numel()))
    result = {
        "u": u[:, :keep].float().cpu(),
        "s": s[:keep].float().cpu(),
        "vh": vh[:keep, :].float().cpu(),
        "seconds": seconds,
    }
    del working, u, s, vh
    if svd_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def svd_value_rows(
    singular_values: torch.Tensor,
    layer: int,
    head: int,
    cache_kind: str,
    cache_length: int,
    seconds: float,
) -> list[dict[str, Any]]:
    total_energy = float(singular_values.square().sum().item())
    cumulative = singular_values.square().cumsum(dim=0)
    rows: list[dict[str, Any]] = []
    for rank_idx, value in enumerate(singular_values.tolist()):
        energy = float(value * value)
        rows.append(
            {
                "cache_kind": cache_kind,
                "cache_length": cache_length,
                "layer": layer,
                "head": head,
                "rank": rank_idx,
                "singular_value": value,
                "energy": energy,
                "energy_fraction": energy / total_energy if total_energy > 0.0 else 0.0,
                "cumulative_energy_fraction": float(cumulative[rank_idx] / total_energy)
                if total_energy > 0.0
                else 0.0,
                "svd_seconds": seconds,
            }
        )
    return rows


def aligned_column_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    rank: int,
    prefix: int | None = None,
) -> float:
    left_vec = left[:, rank]
    right_vec = right[:, rank]
    if prefix is not None:
        left_vec = left_vec[:prefix]
        right_vec = right_vec[:prefix]
    return float(F.cosine_similarity(left_vec, right_vec, dim=0, eps=1e-12))


def aligned_row_cosine(left: torch.Tensor, right: torch.Tensor, rank: int) -> float:
    return float(F.cosine_similarity(left[rank], right[rank], dim=0, eps=1e-12))


def comparison_rows(
    svd_by_length: dict[int, dict[str, torch.Tensor | float]],
    layer: int,
    head: int,
    cache_kind: str,
    sign_invariant: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lengths = sorted(svd_by_length)
    for left_idx, left_length in enumerate(lengths):
        for right_length in lengths[left_idx + 1 :]:
            left = svd_by_length[left_length]
            right = svd_by_length[right_length]
            left_u = left["u"]
            right_u = right["u"]
            left_vh = left["vh"]
            right_vh = right["vh"]
            if not isinstance(left_u, torch.Tensor) or not isinstance(right_u, torch.Tensor):
                raise TypeError("SVD payload is missing U tensors.")
            if not isinstance(left_vh, torch.Tensor) or not isinstance(right_vh, torch.Tensor):
                raise TypeError("SVD payload is missing right singular vectors.")
            rank_count = min(left_u.shape[1], right_u.shape[1], left_vh.shape[0], right_vh.shape[0])
            prefix = min(left_length, right_length)
            for rank_idx in range(rank_count):
                u_cos = aligned_column_cosine(left_u, right_u, rank_idx, prefix)
                right_cos = aligned_row_cosine(left_vh, right_vh, rank_idx)
                rows.append(
                    {
                        "cache_kind": cache_kind,
                        "layer": layer,
                        "head": head,
                        "left_length": left_length,
                        "right_length": right_length,
                        "rank": rank_idx,
                        "u_prefix_tokens": prefix,
                        "u_cosine": abs(u_cos) if sign_invariant else u_cos,
                        "right_singular_vector_cosine": abs(right_cos) if sign_invariant else right_cos,
                    }
                )
    return rows


def format_float_list(values: torch.Tensor, limit: int = 5) -> str:
    shown = values[:limit].detach().float().tolist()
    return "[" + ", ".join(f"{value:.6g}" for value in shown) + "]"


def save_singular_value_plot(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> str | None:
    if not rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipping singular value plot: {exc}", flush=True)
        return None

    by_series: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["cache_kind"], int(row["layer"]), int(row["head"]), int(row["cache_length"]))
        by_series.setdefault(key, []).append(row)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (cache_kind, layer, head, cache_length), series in by_series.items():
        series = sorted(series, key=lambda item: int(item["rank"]))
        ranks = [int(item["rank"]) for item in series]
        values = [float(item["singular_value"]) for item in series]
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=args.plot_dpi)
        ax.plot(ranks, values, marker="o", markersize=2.5, linewidth=1.2)
        ax.set_title(f"{cache_kind} SVD L{layer} H{head} len={cache_length}")
        ax.set_xlabel("rank")
        ax.set_ylabel("singular value")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = plot_dir / f"{cache_kind}_layer_{layer:02d}_head_{head:02d}_len_{cache_length}_singular_values.png"
        fig.savefig(path)
        plt.close(fig)
    return str(plot_dir)


def save_cosine_plot(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> str | None:
    if not rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipping cosine plots: {exc}", flush=True)
        return None

    by_series: dict[tuple[str, int, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["cache_kind"],
            int(row["layer"]),
            int(row["head"]),
            int(row["left_length"]),
            int(row["right_length"]),
        )
        by_series.setdefault(key, []).append(row)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (cache_kind, layer, head, left_length, right_length), series in by_series.items():
        series = sorted(series, key=lambda item: int(item["rank"]))
        ranks = [int(item["rank"]) for item in series]
        u_values = [float(item["u_cosine"]) for item in series]
        right_values = [float(item["right_singular_vector_cosine"]) for item in series]
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=args.plot_dpi)
        ax.plot(ranks, u_values, label="U prefix cosine", marker="o", markersize=2.5, linewidth=1.2)
        ax.plot(ranks, right_values, label="right singular vector cosine", marker="x", markersize=3, linewidth=1.2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{cache_kind} SVD cosine L{layer} H{head}: {left_length} vs {right_length}")
        ax.set_xlabel("rank")
        ax.set_ylabel("cosine")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = plot_dir / (
            f"{cache_kind}_layer_{layer:02d}_head_{head:02d}_"
            f"{left_length}_vs_{right_length}_svd_cosine.png"
        )
        fig.savefig(path)
        plt.close(fig)
    return str(plot_dir)


def fieldnames_for_stats(percentiles: list[float]) -> list[str]:
    return [
        "cache_kind",
        "cache_length",
        "layer",
        "head",
        "dim",
        "dim_l2_norm",
        "count",
        "mean",
        "std",
        "min",
        "max",
        "rms",
        "mean_abs",
        "max_abs",
    ] + [percentile_field(p) for p in percentiles]


def fieldnames_for_norms(percentiles: list[float]) -> list[str]:
    return [
        "cache_kind",
        "cache_length",
        "layer",
        "head",
        "tokens",
        "head_dim",
        "count",
        "mean",
        "std",
        "min",
        "max",
        "rms",
        "mean_abs",
        "max_abs",
    ] + [percentile_field(p) for p in percentiles]


def singular_value_fieldnames() -> list[str]:
    return [
        "cache_kind",
        "cache_length",
        "layer",
        "head",
        "rank",
        "singular_value",
        "energy",
        "energy_fraction",
        "cumulative_energy_fraction",
        "svd_seconds",
    ]


def cosine_fieldnames() -> list[str]:
    return [
        "cache_kind",
        "layer",
        "head",
        "left_length",
        "right_length",
        "rank",
        "u_prefix_tokens",
        "u_cosine",
        "right_singular_vector_cosine",
    ]


def analyze_caches(
    layer_cache_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    expected_heads: int | None,
    output_dir: Path,
    cache_lengths: list[int],
    cache_kinds: list[str],
    percentiles: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    layer_indices = parse_index_spec(args.layers, len(layer_cache_pairs), "layers")
    svd_devices = resolve_svd_devices(args)
    svd_dtype = torch.float64 if args.svd_dtype == "float64" else torch.float32
    if any(device.type == "cuda" for device in svd_devices) and svd_dtype == torch.float64:
        print("warning: float64 SVD on CUDA can be slow; continuing as requested.", flush=True)
    print(
        "SVD devices: " + ", ".join(str(device) for device in svd_devices),
        flush=True,
    )

    singular_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    cache_shapes: list[dict[str, Any]] = []
    svd_tensors_dir = output_dir / "svd_tensors"
    if args.save_svd_tensors:
        svd_tensors_dir.mkdir(parents=True, exist_ok=True)

    dimension_stats_path = output_dir / "dimension_stats.csv"
    token_norm_stats_path = output_dir / "token_norm_stats.csv"
    singular_values_path = output_dir / "singular_values.csv"
    cosine_path = output_dir / "svd_vector_cosines.csv"
    dim_fields = fieldnames_for_stats(percentiles)
    norm_fields = fieldnames_for_norms(percentiles)
    singular_fields = singular_value_fieldnames()
    cosine_fields = cosine_fieldnames()
    write_csv(dimension_stats_path, [], dim_fields)
    write_csv(token_norm_stats_path, [], norm_fields)
    write_csv(singular_values_path, [], singular_fields)
    write_csv(cosine_path, [], cosine_fields)

    for layer_idx in layer_indices:
        key_tensor, value_tensor = layer_cache_pairs[layer_idx]
        tensors_by_kind = {
            "key": key_tensor,
            "value": value_tensor,
        }
        for cache_kind in cache_kinds:
            by_head = cache_tensor_to_head_token_dim(tensors_by_kind[cache_kind], expected_heads)
            kv_heads, total_tokens, head_dim = by_head.shape
            head_indices = parse_index_spec(args.heads, int(kv_heads), "heads")
            cache_shapes.append(
                {
                    "cache_kind": cache_kind,
                    "layer": layer_idx,
                    "kv_heads": int(kv_heads),
                    "tokens": int(total_tokens),
                    "head_dim": int(head_dim),
                    "raw_shape": list(tensors_by_kind[cache_kind].shape),
                    "raw_dtype": str(tensors_by_kind[cache_kind].dtype),
                    "raw_device": str(tensors_by_kind[cache_kind].device),
                }
            )
            print(
                f"analyzing {cache_kind} layer {layer_idx}: "
                f"heads={kv_heads} tokens={total_tokens} head_dim={head_dim}",
                flush=True,
            )

            svd_by_head: dict[int, dict[int, dict[str, torch.Tensor | float]]] = {
                head_idx: {} for head_idx in head_indices
            }
            futures: dict[Any, tuple[int, int, torch.device]] = {}
            stat_tasks: list[tuple[int, int, torch.Tensor]] = []
            task_idx = 0
            max_workers = max(1, len(svd_devices))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for head_idx in head_indices:
                    for cache_length in cache_lengths:
                        if cache_length > total_tokens:
                            continue
                        matrix = by_head[head_idx, :cache_length, :]
                        stat_tasks.append((head_idx, cache_length, matrix))
                        svd_device = svd_devices[task_idx % len(svd_devices)]
                        task_idx += 1
                        print(
                            f"SVD {cache_kind} layer {layer_idx} head {head_idx} "
                            f"length {cache_length} on {svd_device}",
                            flush=True,
                        )
                        future = executor.submit(
                            run_svd,
                            matrix,
                            svd_device,
                            svd_dtype,
                            args.svd_full_matrices,
                            args.max_svd_rank,
                        )
                        futures[future] = (head_idx, cache_length, svd_device)

                for future in as_completed(futures):
                    head_idx, cache_length, svd_device = futures[future]
                    svd_payload = future.result()
                    svd_by_head[head_idx][cache_length] = svd_payload
                    singular_values = svd_payload["s"]
                    if not isinstance(singular_values, torch.Tensor):
                        raise TypeError("SVD payload is missing singular values.")
                    print(
                        f"completed SVD {cache_kind} layer {layer_idx} head {head_idx} "
                        f"length {cache_length} on {svd_device}: "
                        f"seconds={float(svd_payload['seconds']):.3f} "
                        f"top_singular_values={format_float_list(singular_values)}",
                        flush=True,
                    )
                    svd_rows = svd_value_rows(
                        singular_values,
                        layer_idx,
                        head_idx,
                        cache_kind,
                        cache_length,
                        float(svd_payload["seconds"]),
                    )
                    singular_rows.extend(svd_rows)
                    append_csv(singular_values_path, svd_rows, singular_fields)
                    if args.make_plots:
                        save_singular_value_plot(svd_rows, output_dir, args)

                    if args.save_svd_tensors:
                        tensor_path = svd_tensors_dir / (
                            f"{cache_kind}_layer_{layer_idx:02d}_head_{head_idx:02d}_"
                            f"len_{cache_length}_svd.pt"
                        )
                        torch.save(
                            {
                                "metadata": {
                                    "cache_kind": cache_kind,
                                    "layer": layer_idx,
                                    "head": head_idx,
                                    "cache_length": cache_length,
                                    "head_dim": int(head_dim),
                                    "svd_device": str(svd_device),
                                },
                                "u": svd_payload["u"].to(torch.float16),
                                "singular_values": svd_payload["s"],
                                "vh": svd_payload["vh"].to(torch.float16),
                            },
                            tensor_path,
                        )

            for head_idx, svd_by_length in svd_by_head.items():
                head_cosine_rows = comparison_rows(svd_by_length, layer_idx, head_idx, cache_kind)
                cosine_rows.extend(head_cosine_rows)
                append_csv(cosine_path, head_cosine_rows, cosine_fields)
                if args.make_plots:
                    save_cosine_plot(head_cosine_rows, output_dir, args)
                completed_lengths = sorted(svd_by_length)
                if completed_lengths:
                    print(
                        f"completed {cache_kind} layer {layer_idx} head {head_idx}: "
                        f"lengths={completed_lengths} cosine_rows={len(head_cosine_rows)}",
                        flush=True,
                    )
            del svd_by_head

            for head_idx, cache_length, matrix in stat_tasks:
                if args.write_token_norm_stats:
                    norm_row = head_norm_row(
                        matrix,
                        layer_idx,
                        head_idx,
                        cache_kind,
                        cache_length,
                        percentiles,
                    )
                    append_csv(token_norm_stats_path, [norm_row], norm_fields)
                if args.write_dimension_stats:
                    dim_batch = dim_stat_rows(
                        matrix,
                        layer_idx,
                        head_idx,
                        cache_kind,
                        cache_length,
                        percentiles,
                    )
                    append_csv(dimension_stats_path, dim_batch, dim_fields)
            del stat_tasks

            del by_head

    plot_paths: dict[str, str | None] = {}
    if args.make_plots:
        plot_paths["plots_dir"] = str(output_dir / "plots")
        plot_paths["cosine_plots_dir"] = str(output_dir / "plots")

    return {
        "cache_shapes": cache_shapes,
        "paths": {
            "dimension_stats": str(output_dir / "dimension_stats.csv"),
            "token_norm_stats": str(output_dir / "token_norm_stats.csv"),
            "singular_values": str(output_dir / "singular_values.csv"),
            "svd_vector_cosines": str(output_dir / "svd_vector_cosines.csv"),
            "svd_tensors_dir": str(svd_tensors_dir) if args.save_svd_tensors else None,
            "plots": plot_paths,
        },
    }


def main() -> None:
    args = parse_args()
    cache_lengths = parse_positive_ints(args.cache_lengths, "cache_lengths")
    percentiles = parse_percentiles(args.percentiles)
    cache_kinds = parse_cache_kinds(args.cache_kinds)
    text_path = Path(args.text_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not text_path.exists():
        raise FileNotFoundError(f"text_path does not exist: {text_path}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    if args.max_svd_rank <= 0:
        raise ValueError("--max_svd_rank must be positive.")
    if args.plot_dpi <= 0:
        raise ValueError("--plot_dpi must be positive.")

    max_tokens = max(cache_lengths)
    print(f"reading text: {text_path}", flush=True)
    text = read_text_prefix(text_path, args.max_chars)
    if not text.strip():
        raise ValueError(f"No usable text read from {text_path}")

    print(f"loading tokenizer: {args.model_name_or_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    if args.require_max_length and len(token_ids) < max_tokens:
        raise ValueError(
            f"Tokenization produced {len(token_ids)} tokens, fewer than largest requested "
            f"cache length {max_tokens}."
        )
    token_ids = token_ids[:max_tokens]
    if len(token_ids) < 2:
        raise ValueError("Tokenization produced fewer than two tokens.")
    usable_lengths = [length for length in cache_lengths if length <= len(token_ids)]
    if not usable_lengths:
        raise ValueError("No requested cache length is <= the tokenized input length.")
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)
    write_tokens_csv(output_dir / "tokens.csv", tokenizer, token_ids)
    print(f"using tokens: {input_ids.shape[1]} requested_lengths={usable_lengths}", flush=True)

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
    past_key_values, timing_rows = build_kv_cache(model, input_ids, args.chunk_size, input_device)
    write_csv(
        output_dir / "profile_timings.csv",
        timing_rows,
        ["chunk", "start_token", "end_token_exclusive", "token_count", "seconds"],
    )

    expected_heads = getattr(model.config, "num_key_value_heads", None)
    layer_cache_pairs = extract_layer_cache_tensors(past_key_values)
    if args.offload_cache_to_cpu:
        layer_cache_pairs = offload_layer_cache_pairs_to_cpu(layer_cache_pairs)
        del past_key_values
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    analysis = analyze_caches(
        layer_cache_pairs,
        expected_heads,
        output_dir,
        usable_lengths,
        cache_kinds,
        percentiles,
        args,
    )

    payload = {
        "args": vars(args),
        "resolved": {
            "tokens": int(input_ids.shape[1]),
            "text_path": str(text_path),
            "model_name_or_path": args.model_name_or_path,
            "cache_lengths": usable_lengths,
            "cache_kinds": cache_kinds,
            "percentiles": percentiles,
        },
        "paths": {
            "tokens": str(output_dir / "tokens.csv"),
            "profile_timings": str(output_dir / "profile_timings.csv"),
        },
        "analysis": analysis,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
