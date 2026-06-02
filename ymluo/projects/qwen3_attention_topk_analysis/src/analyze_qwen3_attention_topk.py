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
        description="Stream Qwen3 attentions and plot top-k attended key positions per query token."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/attention_topk")
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
    parser.add_argument("--heads", default="all")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--include_self",
        type=str2bool,
        default=True,
        help="If false, mask the current token itself before selecting top-k attention keys.",
    )
    parser.add_argument("--save_token_rows", type=str2bool, default=True)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--make_heatmaps", type=str2bool, default=True)
    parser.add_argument(
        "--heatmap_max_tokens",
        type=int,
        default=1500,
        help="Downsample attention heatmaps above this token count. Use 0 to plot all tokens.",
    )
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


def token_row_fields() -> list[str]:
    return [
        "layer",
        "head",
        "query_token",
        "rank",
        "key_token",
        "index_distance",
        "attention_weight",
    ]


def summary_fields() -> list[str]:
    return [
        "layer",
        "head",
        "query_count",
        "top_k",
        "include_self",
        "mean_index_distance",
        "median_index_distance",
        "p95_index_distance",
        "max_index_distance",
        "mean_attention_weight",
        "max_attention_weight",
    ]


class HeadAccumulator:
    def __init__(self) -> None:
        self.index_distances: list[float] = []
        self.attention_weights: list[float] = []
        self.query_tokens: set[int] = set()

    def update(self, query_token: int, index_distances: torch.Tensor, weights: torch.Tensor) -> None:
        self.query_tokens.add(int(query_token))
        self.index_distances.extend(float(item) for item in index_distances.tolist())
        self.attention_weights.extend(float(item) for item in weights.tolist())

    def row(self, layer: int, head: int, top_k: int, include_self: bool) -> dict[str, Any]:
        index_tensor = torch.tensor(self.index_distances, dtype=torch.float32)
        weight_tensor = torch.tensor(self.attention_weights, dtype=torch.float32)
        return {
            "layer": layer,
            "head": head,
            "query_count": len(self.query_tokens),
            "top_k": top_k,
            "include_self": include_self,
            "mean_index_distance": float(index_tensor.mean()) if index_tensor.numel() else 0.0,
            "median_index_distance": float(index_tensor.median()) if index_tensor.numel() else 0.0,
            "p95_index_distance": float(torch.quantile(index_tensor, 0.95)) if index_tensor.numel() else 0.0,
            "max_index_distance": float(index_tensor.max()) if index_tensor.numel() else 0.0,
            "mean_attention_weight": float(weight_tensor.mean()) if weight_tensor.numel() else 0.0,
            "max_attention_weight": float(weight_tensor.max()) if weight_tensor.numel() else 0.0,
        }


def select_topk_attention_rows(
    attention: torch.Tensor,
    layer: int,
    head_indices: list[int],
    query_start: int,
    top_k: int,
    include_self: bool,
    accumulators: dict[tuple[int, int], HeadAccumulator],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # attention: [heads, query_tokens, key_tokens]
    _, query_count, key_count = attention.shape
    for head in head_indices:
        head_attention = attention[head].float().cpu()
        for local_query in range(query_count):
            query_token = query_start + local_query
            scores = head_attention[local_query].clone()
            valid_key_count = min(key_count, query_token + 1)
            if valid_key_count < key_count:
                scores[valid_key_count:] = -float("inf")
            if not include_self and query_token < key_count:
                scores[query_token] = -float("inf")
            finite_count = int(torch.isfinite(scores).sum())
            if finite_count <= 0:
                continue
            k = min(top_k, finite_count)
            weights, key_indices = torch.topk(scores, k=k, largest=True)
            index_distances = (query_token - key_indices.long()).abs()
            accumulators[(layer, head)].update(query_token, index_distances, weights)
            for rank, (key_token, distance, weight) in enumerate(
                zip(key_indices.tolist(), index_distances.tolist(), weights.tolist()),
                start=1,
            ):
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "query_token": query_token,
                        "rank": rank,
                        "key_token": int(key_token),
                        "index_distance": int(distance),
                        "attention_weight": float(weight),
                    }
                )
    return rows


def plot_head_rows(
    rows_by_head: dict[tuple[int, int], list[dict[str, Any]]],
    output_dir: Path,
    dpi: int,
    alpha: float,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    for (layer, head), rows in sorted(rows_by_head.items()):
        plot_dir = output_dir / "plots" / f"layer_{layer:02d}" / f"head_{head:02d}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for metric, ylabel in (
            ("index_distance", "Index distance"),
            ("attention_weight", "Attention weight"),
        ):
            fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
            max_rank = max(int(row["rank"]) for row in rows) if rows else 0
            for rank in range(1, max_rank + 1):
                rank_rows = [row for row in rows if int(row["rank"]) == rank]
                ax.scatter(
                    [int(row["query_token"]) for row in rank_rows],
                    [float(row[metric]) for row in rank_rows],
                    s=2,
                    alpha=alpha,
                    linewidths=0,
                    label=f"top{rank}",
                )
            ax.set_title(f"Attention top-k L{layer} H{head}: {ylabel}")
            ax.set_xlabel("Query token index")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.2)
            ax.legend(markerscale=4, fontsize=8, ncol=min(max_rank, 5))
            fig.tight_layout()
            path = plot_dir / f"{metric}_by_rank_tokens.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(str(path))
    return paths


def plot_attention_heatmap(
    matrix: torch.Tensor,
    output_dir: Path,
    layer: int,
    head: int,
    total_tokens: int,
    stride: int,
    dpi: int,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / f"layer_{layer:02d}" / f"head_{head:02d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=dpi)
    image = ax.imshow(
        matrix.numpy(),
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="magma",
    )
    ax.set_title(f"Attention L{layer} H{head}: attention weight")
    ax.set_xlabel("Key token index")
    ax.set_ylabel("Query token index")
    if matrix.shape[0] > 1:
        tick_count = min(6, int(matrix.shape[0]))
        ticks = torch.linspace(0, int(matrix.shape[0]) - 1, tick_count).round().long().tolist()
        labels = [str(min(total_tokens - 1, tick * stride)) for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    fig.tight_layout()
    path = plot_dir / "attention_weight_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 1:
        raise ValueError("--max_tokens must be greater than 1.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = Path(args.text_path)
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
    head_count = int(getattr(model.config, "num_attention_heads"))
    layer_indices = parse_index_spec(args.layers, layer_count, "layers")
    head_indices = parse_index_spec(args.heads, head_count, "heads")

    input_device = pick_input_device(model, requested_device)
    past_key_values = None
    total_tokens = int(input_ids.shape[1])
    total_chunks = math.ceil(total_tokens / args.chunk_size)
    token_csv_path = output_dir / "attention_topk_by_token.csv"
    wrote_token_rows = False
    accumulators = {(layer, head): HeadAccumulator() for layer in layer_indices for head in head_indices}
    plot_rows_by_head: dict[tuple[int, int], list[dict[str, Any]]] = {
        (layer, head): [] for layer in layer_indices for head in head_indices
    }
    heatmap_stride = 1
    if args.heatmap_max_tokens > 0 and total_tokens > args.heatmap_max_tokens:
        heatmap_stride = math.ceil(total_tokens / args.heatmap_max_tokens)
    heatmap_tokens = math.ceil(total_tokens / heatmap_stride)
    heatmap_matrices: dict[tuple[int, int], torch.Tensor] = {}
    if args.make_heatmaps:
        heatmap_matrices = {
            (layer, head): torch.zeros((heatmap_tokens, heatmap_tokens), dtype=torch.float32)
            for layer in layer_indices
            for head in head_indices
        }
    timing_rows: list[dict[str, Any]] = []

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
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model_forward(model, kwargs)
        seconds = time.perf_counter() - started
        if outputs.attentions is None:
            raise RuntimeError("Model did not return attentions. Use ATTN_IMPLEMENTATION=eager if needed.")
        past_key_values = outputs.past_key_values
        timing_rows.append(
            {
                "chunk": chunk_idx,
                "start_token": start,
                "end_token_exclusive": end,
                "token_count": end - start,
                "seconds": seconds,
            }
        )

        chunk_rows: list[dict[str, Any]] = []
        for layer in layer_indices:
            attention = outputs.attentions[layer][0].detach()
            rows = select_topk_attention_rows(
                attention,
                layer,
                head_indices,
                start,
                args.top_k,
                args.include_self,
                accumulators,
            )
            chunk_rows.extend(rows)
            if args.make_plots:
                for row in rows:
                    plot_rows_by_head[(int(row["layer"]), int(row["head"]))].append(row)
            if args.make_heatmaps:
                attention_cpu = attention.float().cpu()
                for local_query in range(attention_cpu.shape[1]):
                    query_token = start + local_query
                    if query_token % heatmap_stride != 0:
                        continue
                    query_bin = query_token // heatmap_stride
                    key_values = attention_cpu[:, local_query, ::heatmap_stride]
                    for head in head_indices:
                        heatmap_matrices[(layer, head)][query_bin, : key_values.shape[1]] = key_values[head]

        if args.save_token_rows and chunk_rows:
            append_rows(token_csv_path, chunk_rows, token_row_fields(), append=wrote_token_rows)
            wrote_token_rows = True
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = [
        accumulators[(layer, head)].row(layer, head, args.top_k, args.include_self)
        for layer in layer_indices
        for head in head_indices
    ]
    write_csv(output_dir / "summary_by_head.csv", summary_rows, summary_fields())
    write_csv(
        output_dir / "profile_timings.csv",
        timing_rows,
        ["chunk", "start_token", "end_token_exclusive", "token_count", "seconds"],
    )

    plot_paths: list[str] = []
    if args.make_plots:
        plot_paths = plot_head_rows(plot_rows_by_head, output_dir, args.plot_dpi, args.point_alpha)
    if args.make_heatmaps:
        for (layer, head), matrix in sorted(heatmap_matrices.items()):
            plot_paths.append(
                plot_attention_heatmap(
                    matrix,
                    output_dir,
                    layer,
                    head,
                    total_tokens,
                    heatmap_stride,
                    args.plot_dpi,
                )
            )

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "tokens": total_tokens,
                    "layers": layer_indices,
                    "heads": head_indices,
                    "attention_kind": "post_softmax_attention_weight",
                },
                "paths": {
                    "attention_topk_by_token": str(token_csv_path) if args.save_token_rows else None,
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
