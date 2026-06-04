from __future__ import annotations

import argparse
import csv
import json
import math
import time
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
        description="Evaluate KV-head retrieval candidates by true attention-energy coverage."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/retrieval_energy")
    parser.add_argument("--max_tokens", type=int, default=5000)
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
    parser.add_argument("--boundary_fraction", type=float, default=0.01)
    parser.add_argument("--seed_fraction", type=float, default=0.01)
    parser.add_argument("--neighbor_count", type=int, default=20)
    parser.add_argument("--knn_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--compute_oracle_baseline", type=str2bool, default=True)
    parser.add_argument("--save_token_rows", type=str2bool, default=True)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--token_bins", type=int, default=100)
    parser.add_argument("--plot_smoothing_window", type=int, default=0)
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


def resolve_knn_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--knn_device cuda requested, but CUDA is not available.")
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
        for row in rows:
            writer.writerow(row)


def append_rows(path: Path, rows: list[dict[str, Any]], fields: list[str], append: bool) -> None:
    with path.open("a" if append else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


@torch.inference_mode()
def previous_l2_neighbors(vectors: torch.Tensor, neighbor_count: int, device: torch.device) -> list[list[int]]:
    working = vectors.to(device=device, dtype=torch.float32)
    distances = torch.cdist(working, working, p=2)
    tokens = int(distances.shape[0])
    mask = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool, device=device), diagonal=0)
    distances.masked_fill_(mask, float("inf"))
    k = min(neighbor_count, max(1, tokens - 1))
    values, indices = torch.topk(distances, k=k, dim=1, largest=False)
    result: list[list[int]] = []
    for token_idx in range(tokens):
        valid = [int(idx) for idx, value in zip(indices[token_idx].tolist(), values[token_idx].tolist()) if math.isfinite(float(value))]
        result.append(valid)
    del working, distances, values, indices
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_neighbor_graphs(
    key_tensors: list[torch.Tensor],
    layer_indices: list[int],
    kv_head_indices: list[int],
    expected_kv_heads: int,
    neighbor_count: int,
    device: torch.device,
) -> dict[tuple[int, int], list[list[int]]]:
    graphs: dict[tuple[int, int], list[list[int]]] = {}
    for layer in layer_indices:
        key_by_head = key_tensor_to_head_token_dim(key_tensors[layer], expected_kv_heads)
        for kv_head in kv_head_indices:
            print(f"building previous-only L2 graph: layer {layer}, kv_head {kv_head}", flush=True)
            graphs[(layer, kv_head)] = previous_l2_neighbors(key_by_head[kv_head], neighbor_count, device)
        del key_by_head
    return graphs


def visible_regions(token: int, boundary_fraction: float) -> tuple[set[int], set[int], tuple[int, int]]:
    visible = token + 1
    boundary_count = max(1, math.ceil(boundary_fraction * visible))
    prefix = set(range(0, min(boundary_count, visible)))
    recent_start = max(0, visible - boundary_count)
    recent = set(range(recent_start, visible))
    middle_start = min(boundary_count, visible)
    middle_end = max(middle_start, recent_start)
    return prefix, recent, (middle_start, middle_end)


def top_candidate_seeds(
    prev_attention: torch.Tensor | None,
    prev_candidates: set[int] | None,
    middle_range: tuple[int, int],
    visible: int,
    seed_fraction: float,
) -> list[int]:
    if prev_attention is None or not prev_candidates or middle_range[1] <= middle_range[0]:
        return []
    pool = sorted(
        idx
        for idx in prev_candidates
        if middle_range[0] <= idx < middle_range[1] and idx < prev_attention.numel()
    )
    if not pool:
        return []
    seed_count = min(len(pool), max(1, math.ceil(seed_fraction * visible)))
    pool_tensor = torch.tensor(pool, dtype=torch.long)
    weights = prev_attention[pool_tensor]
    _, top_positions = torch.topk(weights, k=seed_count, largest=True)
    return [pool[int(pos)] for pos in top_positions.tolist()]


def candidate_set_for_kv_head(
    token: int,
    layer: int,
    kv_head: int,
    attention_heads: list[int],
    prev_attention_by_head: dict[tuple[int, int], torch.Tensor],
    prev_candidates_by_head: dict[tuple[int, int], set[int]],
    neighbor_graphs: dict[tuple[int, int], list[list[int]]],
    boundary_fraction: float,
    seed_fraction: float,
) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
    visible = token + 1
    prefix, recent, middle_range = visible_regions(token, boundary_fraction)
    seeds: set[int] = set()
    for attention_head in attention_heads:
        seeds.update(
            top_candidate_seeds(
                prev_attention_by_head.get((layer, attention_head)),
                prev_candidates_by_head.get((layer, attention_head)),
                middle_range,
                visible,
                seed_fraction,
            )
        )
    expanded: set[int] = set(seeds)
    graph = neighbor_graphs[(layer, kv_head)]
    for seed in seeds:
        expanded.update(neighbor for neighbor in graph[seed] if neighbor <= token)
    candidates = set(prefix) | set(recent) | expanded
    candidates = {idx for idx in candidates if 0 <= idx <= token}
    return candidates, prefix, recent, seeds, expanded


def token_row_fields() -> list[str]:
    return [
        "layer",
        "kv_head",
        "attention_head",
        "query_token",
        "visible_tokens",
        "candidate_count",
        "candidate_fraction",
        "prefix_count",
        "recent_count",
        "seed_count",
        "expanded_middle_count",
        "method_energy",
        "oracle_energy",
        "prefix_recent_energy",
        "energy_gap_to_oracle",
    ]


def summary_fields() -> list[str]:
    return [
        "layer",
        "kv_head",
        "attention_head",
        "query_count",
        "candidate_fraction_mean",
        "candidate_fraction_p95",
        "method_energy_mean",
        "method_energy_p05",
        "oracle_energy_mean",
        "prefix_recent_energy_mean",
        "energy_gap_to_oracle_mean",
    ]


def energy_for_indices(attention: torch.Tensor, indices: set[int]) -> float:
    if not indices:
        return 0.0
    index_tensor = torch.tensor(sorted(indices), dtype=torch.long)
    return float(attention[index_tensor].sum())


def index_tensor_for_indices(indices: set[int]) -> torch.Tensor | None:
    if not indices:
        return None
    return torch.tensor(sorted(indices), dtype=torch.long)


def energy_for_index_tensor(attention: torch.Tensor, index_tensor: torch.Tensor | None) -> float:
    if index_tensor is None:
        return 0.0
    return float(attention[index_tensor].sum())


def oracle_energy(attention: torch.Tensor, candidate_count: int, visible: int) -> float:
    if candidate_count <= 0:
        return 0.0
    k = min(candidate_count, visible)
    values, _ = torch.topk(attention[:visible], k=k, largest=True)
    return float(values.sum())


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["kv_head"]), int(row["attention_head"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (layer, kv_head, attention_head), group in sorted(grouped.items()):
        candidate_fraction = torch.tensor([float(row["candidate_fraction"]) for row in group], dtype=torch.float32)
        method_energy = torch.tensor([float(row["method_energy"]) for row in group], dtype=torch.float32)
        prefix_recent = torch.tensor([float(row["prefix_recent_energy"]) for row in group], dtype=torch.float32)
        oracle_values = [float(row["oracle_energy"]) for row in group if row["oracle_energy"] != ""]
        gap_values = [float(row["energy_gap_to_oracle"]) for row in group if row["energy_gap_to_oracle"] != ""]
        oracle = torch.tensor(oracle_values, dtype=torch.float32) if oracle_values else None
        gap = torch.tensor(gap_values, dtype=torch.float32) if gap_values else None
        summary.append(
            {
                "layer": layer,
                "kv_head": kv_head,
                "attention_head": attention_head,
                "query_count": len(group),
                "candidate_fraction_mean": float(candidate_fraction.mean()),
                "candidate_fraction_p95": float(torch.quantile(candidate_fraction, 0.95)),
                "method_energy_mean": float(method_energy.mean()),
                "method_energy_p05": float(torch.quantile(method_energy, 0.05)),
                "oracle_energy_mean": float(oracle.mean()) if oracle is not None else "",
                "prefix_recent_energy_mean": float(prefix_recent.mean()),
                "energy_gap_to_oracle_mean": float(gap.mean()) if gap is not None else "",
            }
        )
    return summary


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    half_left = (window - 1) // 2
    half_right = window // 2
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - half_left)
        end = min(len(values), index + half_right + 1)
        subset = [value for value in values[start:end] if math.isfinite(value)]
        result.append(sum(subset) / len(subset) if subset else float("nan"))
    return result


def save_energy_plot(
    tokens: list[int],
    method_energy: list[float],
    oracle_energy_values: list[float] | None,
    prefix_recent_energy: list[float],
    candidate_fraction: list[float],
    title: str,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 4), dpi=dpi)
    ax1.plot(tokens, method_energy, label="method energy", linewidth=1.0)
    if oracle_energy_values is not None:
        ax1.plot(tokens, oracle_energy_values, label="oracle top-s energy", linewidth=1.0)
    ax1.plot(tokens, prefix_recent_energy, label="prefix+recent energy", linewidth=1.0)
    ax1.set_xlabel("Query token index")
    ax1.set_ylabel("Attention energy")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(tokens, [100.0 * value for value in candidate_fraction], color="black", alpha=0.35, linewidth=0.8, label="candidate %")
    ax2.set_ylabel("Candidate set size (%)")
    ax2.set_ylim(0.0, max(5.0, max(100.0 * value for value in candidate_fraction) * 1.1))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_head_rows(
    rows_by_head: dict[tuple[int, int, int], list[dict[str, Any]]],
    output_dir: Path,
    dpi: int,
    smoothing_window: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")

    paths: list[str] = []
    for (layer, kv_head, attention_head), rows in sorted(rows_by_head.items()):
        plot_dir = output_dir / "plots" / f"layer_{layer:02d}" / f"kv_head_{kv_head:02d}" / f"attention_head_{attention_head:02d}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        tokens = [int(row["query_token"]) for row in rows]
        method_energy = [float(row["method_energy"]) for row in rows]
        oracle_values = None
        if any(row["oracle_energy"] != "" for row in rows):
            oracle_values = [float(row["oracle_energy"]) if row["oracle_energy"] != "" else float("nan") for row in rows]
        prefix_recent_energy = [float(row["prefix_recent_energy"]) for row in rows]
        candidate_fraction = [float(row["candidate_fraction"]) for row in rows]

        path = plot_dir / "energy_and_candidate_fraction_by_token.png"
        save_energy_plot(
            tokens,
            method_energy,
            oracle_values,
            prefix_recent_energy,
            candidate_fraction,
            f"Retrieval energy L{layer} KV{kv_head} AttnH{attention_head}",
            path,
            dpi,
        )
        paths.append(str(path))

        if smoothing_window > 1:
            smooth_path = plot_dir / f"energy_and_candidate_fraction_smoothed_w{smoothing_window}.png"
            save_energy_plot(
                tokens,
                moving_average(method_energy, smoothing_window),
                moving_average(oracle_values, smoothing_window) if oracle_values is not None else None,
                moving_average(prefix_recent_energy, smoothing_window),
                moving_average(candidate_fraction, smoothing_window),
                f"Retrieval energy L{layer} KV{kv_head} AttnH{attention_head} smoothed w={smoothing_window}",
                smooth_path,
                dpi,
            )
            paths.append(str(smooth_path))
    return paths


def main() -> None:
    args = parse_args()
    if args.boundary_fraction <= 0 or args.seed_fraction <= 0:
        raise ValueError("fractions must be positive.")
    if args.neighbor_count <= 0:
        raise ValueError("--neighbor_count must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
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
    attention_heads = int(getattr(model.config, "num_attention_heads"))
    kv_heads = int(getattr(model.config, "num_key_value_heads"))
    group_size = attention_heads // kv_heads
    layer_indices = parse_index_spec(args.layers, layer_count, "layers")
    kv_head_indices = parse_index_spec(args.kv_heads, kv_heads, "kv_heads")
    attention_heads_by_kv = {
        kv_head: [head for head in range(kv_head * group_size, (kv_head + 1) * group_size)]
        for kv_head in kv_head_indices
    }

    input_device = pick_input_device(model, requested_device)
    print("building K cache for L2 neighbor graphs", flush=True)
    past_key_values = build_k_cache(model, input_ids, args.chunk_size, input_device)
    key_tensors = extract_key_tensors(past_key_values)
    neighbor_graphs = build_neighbor_graphs(
        key_tensors,
        layer_indices,
        kv_head_indices,
        kv_heads,
        args.neighbor_count,
        resolve_knn_device(args.knn_device),
    )
    del past_key_values, key_tensors
    if input_device.type == "cuda":
        torch.cuda.empty_cache()

    total_tokens = int(input_ids.shape[1])
    total_chunks = math.ceil(total_tokens / args.chunk_size)
    prev_attention_by_head: dict[tuple[int, int], torch.Tensor] = {}
    prev_candidates_by_head: dict[tuple[int, int], set[int]] = {}
    rows_all: list[dict[str, Any]] = []
    rows_by_head: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    token_csv_path = output_dir / "retrieval_energy_by_token.csv"
    wrote_token_rows = False
    past_key_values = None

    selected_attention_heads = sorted({head for heads in attention_heads_by_kv.values() for head in heads})
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
        print(f"attention energy chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        with torch.inference_mode():
            outputs = model_forward(model, kwargs)
        if outputs.attentions is None:
            raise RuntimeError("Model did not return attentions. Use ATTN_IMPLEMENTATION=eager.")
        past_key_values = outputs.past_key_values
        attention_layers = {
            layer: outputs.attentions[layer][0].detach().float().cpu()
            for layer in layer_indices
        }

        chunk_rows: list[dict[str, Any]] = []
        for local_query in range(end - start):
            query_token = start + local_query
            visible = query_token + 1
            for layer in layer_indices:
                attention_layer = attention_layers[layer]
                for kv_head in kv_head_indices:
                    shared_heads = attention_heads_by_kv[kv_head]
                    candidates, prefix, recent, seeds, expanded = candidate_set_for_kv_head(
                        query_token,
                        layer,
                        kv_head,
                        shared_heads,
                        prev_attention_by_head,
                        prev_candidates_by_head,
                        neighbor_graphs,
                        args.boundary_fraction,
                        args.seed_fraction,
                    )
                    prefix_recent = prefix | recent
                    candidate_index_tensor = index_tensor_for_indices(candidates)
                    prefix_recent_index_tensor = index_tensor_for_indices(prefix_recent)
                    for attention_head in shared_heads:
                        attention = attention_layer[attention_head, local_query, :visible]
                        method = energy_for_index_tensor(attention, candidate_index_tensor)
                        oracle = oracle_energy(attention, len(candidates), visible) if args.compute_oracle_baseline else ""
                        pr_energy = energy_for_index_tensor(attention, prefix_recent_index_tensor)
                        row = {
                            "layer": layer,
                            "kv_head": kv_head,
                            "attention_head": attention_head,
                            "query_token": query_token,
                            "visible_tokens": visible,
                            "candidate_count": len(candidates),
                            "candidate_fraction": len(candidates) / visible,
                            "prefix_count": len(prefix),
                            "recent_count": len(recent),
                            "seed_count": len(seeds),
                            "expanded_middle_count": len(expanded),
                            "method_energy": method,
                            "oracle_energy": oracle,
                            "prefix_recent_energy": pr_energy,
                            "energy_gap_to_oracle": (oracle - method) if args.compute_oracle_baseline else "",
                        }
                        chunk_rows.append(row)
                        rows_all.append(row)
                        if args.make_plots:
                            rows_by_head[(layer, kv_head, attention_head)].append(row)
                        prev_candidates_by_head[(layer, attention_head)] = set(candidates)
                for attention_head in selected_attention_heads:
                    prev_attention_by_head[(layer, attention_head)] = attention_layer[
                        attention_head, local_query, :visible
                    ].clone()

        if args.save_token_rows and chunk_rows:
            append_rows(token_csv_path, chunk_rows, token_row_fields(), append=wrote_token_rows)
            wrote_token_rows = True
        del outputs, chunk, attention_layers
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = summarize_rows(rows_all)
    write_csv(output_dir / "summary_by_head.csv", summary_rows, summary_fields())
    plot_paths = plot_head_rows(rows_by_head, output_dir, args.plot_dpi, args.plot_smoothing_window) if args.make_plots else []
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "tokens": total_tokens,
                    "layers": layer_indices,
                    "kv_heads": kv_head_indices,
                    "attention_heads_by_kv": attention_heads_by_kv,
                },
                "paths": {
                    "retrieval_energy_by_token": str(token_csv_path) if args.save_token_rows else None,
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
