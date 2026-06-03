from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TreeLayout:
    leaf_size: int
    leaf_ranges: list[tuple[int, int]]
    mid_ranges: list[tuple[int, int]]
    mid_children: list[list[int]]
    big_ranges: list[tuple[int, int]]
    big_children: list[list[int]]


@dataclass(frozen=True)
class TreeCandidateResult:
    candidates: set[int]
    prefix: set[int]
    recent: set[int]
    selected_big_count: int
    selected_mid_count: int
    selected_leaf_count: int
    tree_middle_count: int


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical K-center tree retrieval by true attention-energy coverage."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/tree_retrieval_energy")
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
    parser.add_argument("--leaf_fraction", type=float, default=0.001)
    parser.add_argument("--leaf_size", type=int, default=0)
    parser.add_argument("--tree_fanout", type=int, default=10)
    parser.add_argument("--tree_branch_counts", default="5,5,5")
    parser.add_argument("--candidate_granularity", choices=["attention_head", "kv_head_union"], default="attention_head")
    parser.add_argument("--save_token_rows", type=str2bool, default=True)
    parser.add_argument("--make_plots", type=str2bool, default=True)
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


def parse_branch_counts(spec: str) -> tuple[int, int, int]:
    values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("--tree_branch_counts must be three positive integers, e.g. 5,5,5")
    return values[0], values[1], values[2]


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


def build_tree_layout(total_tokens: int, leaf_size: int, fanout: int) -> TreeLayout:
    leaf_ranges = [(start, min(start + leaf_size, total_tokens)) for start in range(0, total_tokens, leaf_size)]
    mid_ranges: list[tuple[int, int]] = []
    mid_children: list[list[int]] = []
    for start in range(0, len(leaf_ranges), fanout):
        children = list(range(start, min(start + fanout, len(leaf_ranges))))
        mid_children.append(children)
        mid_ranges.append((leaf_ranges[children[0]][0], leaf_ranges[children[-1]][1]))
    big_ranges: list[tuple[int, int]] = []
    big_children: list[list[int]] = []
    for start in range(0, len(mid_ranges), fanout):
        children = list(range(start, min(start + fanout, len(mid_ranges))))
        big_children.append(children)
        big_ranges.append((mid_ranges[children[0]][0], mid_ranges[children[-1]][1]))
    return TreeLayout(
        leaf_size=leaf_size,
        leaf_ranges=leaf_ranges,
        mid_ranges=mid_ranges,
        mid_children=mid_children,
        big_ranges=big_ranges,
        big_children=big_children,
    )


def build_key_prefix_sums(
    key_tensors: list[torch.Tensor],
    layer_indices: list[int],
    kv_head_indices: list[int],
    expected_kv_heads: int,
) -> dict[tuple[int, int], torch.Tensor]:
    prefix_sums: dict[tuple[int, int], torch.Tensor] = {}
    for layer in layer_indices:
        key_by_head = key_tensor_to_head_token_dim(key_tensors[layer], expected_kv_heads)
        for kv_head in kv_head_indices:
            vectors = key_by_head[kv_head].float().cpu()
            zeros = torch.zeros((1, vectors.shape[-1]), dtype=torch.float32)
            prefix_sums[(layer, kv_head)] = torch.cat([zeros, torch.cumsum(vectors, dim=0)], dim=0)
        del key_by_head
    return prefix_sums


def visible_regions(token: int, boundary_fraction: float) -> tuple[set[int], set[int], tuple[int, int]]:
    visible = token + 1
    boundary_count = max(1, math.ceil(boundary_fraction * visible))
    prefix = set(range(0, min(boundary_count, visible)))
    recent_start = max(0, visible - boundary_count)
    recent = set(range(recent_start, visible))
    middle_start = min(boundary_count, visible)
    middle_end = max(middle_start, recent_start)
    return prefix, recent, (middle_start, middle_end)


def range_center(
    prefix_sum: torch.Tensor,
    start: int,
    end: int,
    middle_range: tuple[int, int],
    visible: int,
) -> tuple[torch.Tensor | None, int, tuple[int, int]]:
    left = max(start, middle_range[0])
    right = min(end, middle_range[1], visible)
    count = right - left
    if count <= 0:
        return None, 0, (left, right)
    return (prefix_sum[right] - prefix_sum[left]) / count, count, (left, right)


def top_scored_indices(
    query: torch.Tensor,
    prefix_sum: torch.Tensor,
    ranges: list[tuple[int, int]],
    candidate_indices: list[int],
    middle_range: tuple[int, int],
    visible: int,
    top_count: int,
) -> list[int]:
    scored: list[tuple[float, int]] = []
    for index in candidate_indices:
        center, count, _ = range_center(prefix_sum, ranges[index][0], ranges[index][1], middle_range, visible)
        if center is None or count <= 0:
            continue
        scored.append((float(torch.dot(query, center)), index))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in scored[:top_count]]


def tree_candidate_set(
    token: int,
    query: torch.Tensor,
    prefix_sum: torch.Tensor,
    layout: TreeLayout,
    branch_counts: tuple[int, int, int],
    boundary_fraction: float,
) -> TreeCandidateResult:
    visible = token + 1
    prefix, recent, middle_range = visible_regions(token, boundary_fraction)
    if middle_range[1] <= middle_range[0]:
        candidates = {idx for idx in prefix | recent if idx <= token}
        return TreeCandidateResult(candidates, prefix, recent, 0, 0, 0, 0)

    big_indices = top_scored_indices(
        query,
        prefix_sum,
        layout.big_ranges,
        list(range(len(layout.big_ranges))),
        middle_range,
        visible,
        branch_counts[0],
    )
    mid_indices: list[int] = []
    for big_index in big_indices:
        mid_indices.extend(
            top_scored_indices(
                query,
                prefix_sum,
                layout.mid_ranges,
                layout.big_children[big_index],
                middle_range,
                visible,
                branch_counts[1],
            )
        )
    leaf_indices: list[int] = []
    for mid_index in mid_indices:
        leaf_indices.extend(
            top_scored_indices(
                query,
                prefix_sum,
                layout.leaf_ranges,
                layout.mid_children[mid_index],
                middle_range,
                visible,
                branch_counts[2],
            )
        )

    tree_tokens: set[int] = set()
    for leaf_index in sorted(set(leaf_indices)):
        _, count, valid_range = range_center(
            prefix_sum,
            layout.leaf_ranges[leaf_index][0],
            layout.leaf_ranges[leaf_index][1],
            middle_range,
            visible,
        )
        if count > 0:
            tree_tokens.update(range(valid_range[0], valid_range[1]))
    candidates = {idx for idx in prefix | recent | tree_tokens if 0 <= idx <= token}
    return TreeCandidateResult(
        candidates=candidates,
        prefix=prefix,
        recent=recent,
        selected_big_count=len(set(big_indices)),
        selected_mid_count=len(set(mid_indices)),
        selected_leaf_count=len(set(leaf_indices)),
        tree_middle_count=len(tree_tokens),
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_to_query(query_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query_states * cos) + (rotate_half(query_states) * sin)


def model_body(model: torch.nn.Module) -> torch.nn.Module:
    for attr_name in ("model", "transformer"):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
    return model


def layer_module(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    body = model_body(model)
    if hasattr(body, "layers"):
        return body.layers[layer]
    if hasattr(body, "h"):
        return body.h[layer]
    raise AttributeError("Could not find transformer layers on model.")


def compute_query_states(
    model: torch.nn.Module,
    layer: int,
    hidden_states: torch.Tensor,
    local_query: int,
    query_token: int,
    num_attention_heads: int,
) -> torch.Tensor:
    layer_obj = layer_module(model, layer)
    attn = getattr(layer_obj, "self_attn", None) or getattr(layer_obj, "attention", None)
    if attn is None or not hasattr(attn, "q_proj"):
        raise AttributeError("Could not find self_attn.q_proj for query-state extraction.")

    q_weight = getattr(attn.q_proj, "weight", None)
    q_device = q_weight.device if q_weight is not None else hidden_states.device
    q_dtype = q_weight.dtype if q_weight is not None and q_weight.is_floating_point() else hidden_states.dtype
    hidden = hidden_states[:, local_query : local_query + 1, :].to(device=q_device, dtype=q_dtype)
    projected = attn.q_proj(hidden)
    head_dim = int(getattr(attn, "head_dim", projected.shape[-1] // num_attention_heads))
    query_states = projected.view(projected.shape[0], projected.shape[1], num_attention_heads, head_dim)
    query_states = query_states.transpose(1, 2)
    if hasattr(attn, "q_norm"):
        query_states = attn.q_norm(query_states)

    position_ids = torch.tensor([[query_token]], dtype=torch.long, device=hidden.device)
    rotary = getattr(attn, "rotary_emb", None)
    if rotary is None:
        rotary = getattr(model_body(model), "rotary_emb", None)
    if rotary is not None:
        try:
            cos, sin = rotary(hidden, position_ids)
        except TypeError:
            cos, sin = rotary(query_states, position_ids)
        query_states = apply_rotary_to_query(query_states, cos, sin)
    return query_states[0, :, 0, :].detach().float().cpu()


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
        "selected_big_count",
        "selected_mid_count",
        "selected_leaf_count",
        "tree_middle_count",
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
        oracle = torch.tensor([float(row["oracle_energy"]) for row in group], dtype=torch.float32)
        prefix_recent = torch.tensor([float(row["prefix_recent_energy"]) for row in group], dtype=torch.float32)
        gap = torch.tensor([float(row["energy_gap_to_oracle"]) for row in group], dtype=torch.float32)
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
                "oracle_energy_mean": float(oracle.mean()),
                "prefix_recent_energy_mean": float(prefix_recent.mean()),
                "energy_gap_to_oracle_mean": float(gap.mean()),
            }
        )
    return summary


def plot_head_rows(rows_by_head: dict[tuple[int, int, int], list[dict[str, Any]]], output_dir: Path, dpi: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    for (layer, kv_head, attention_head), rows in sorted(rows_by_head.items()):
        plot_dir = output_dir / "plots" / f"layer_{layer:02d}" / f"kv_head_{kv_head:02d}" / f"attention_head_{attention_head:02d}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        tokens = [int(row["query_token"]) for row in rows]
        fig, ax1 = plt.subplots(figsize=(12, 4), dpi=dpi)
        ax1.plot(tokens, [float(row["method_energy"]) for row in rows], label="method energy", linewidth=1.0)
        ax1.plot(tokens, [float(row["oracle_energy"]) for row in rows], label="oracle top-s energy", linewidth=1.0)
        ax1.plot(tokens, [float(row["prefix_recent_energy"]) for row in rows], label="prefix+recent energy", linewidth=1.0)
        ax1.set_xlabel("Query token index")
        ax1.set_ylabel("Attention energy")
        ax1.set_ylim(0.0, 1.05)
        ax1.grid(True, alpha=0.2)
        ax2 = ax1.twinx()
        ax2.plot(tokens, [100.0 * float(row["candidate_fraction"]) for row in rows], color="black", alpha=0.35, linewidth=0.8, label="candidate %")
        ax2.set_ylabel("Candidate set size (%)")
        ax2.set_ylim(0.0, max(5.0, max(100.0 * float(row["candidate_fraction"]) for row in rows) * 1.1))
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
        ax1.set_title(f"Tree retrieval energy L{layer} KV{kv_head} AttnH{attention_head}")
        fig.tight_layout()
        path = plot_dir / "energy_and_candidate_fraction_by_token.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    if args.boundary_fraction <= 0 or args.leaf_fraction <= 0:
        raise ValueError("fractions must be positive.")
    if args.tree_fanout <= 1:
        raise ValueError("--tree_fanout must be greater than 1.")
    branch_counts = parse_branch_counts(args.tree_branch_counts)

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
    total_tokens = int(input_ids.shape[1])

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
    selected_attention_heads = sorted({head for heads in attention_heads_by_kv.values() for head in heads})

    leaf_size = args.leaf_size if args.leaf_size > 0 else max(1, math.ceil(args.leaf_fraction * total_tokens))
    layout = build_tree_layout(total_tokens, leaf_size, args.tree_fanout)

    input_device = pick_input_device(model, requested_device)
    print("building K cache for tree centers", flush=True)
    past_key_values = build_k_cache(model, input_ids, args.chunk_size, input_device)
    key_tensors = extract_key_tensors(past_key_values)
    key_prefix_sums = build_key_prefix_sums(key_tensors, layer_indices, kv_head_indices, kv_heads)
    del past_key_values, key_tensors
    if input_device.type == "cuda":
        torch.cuda.empty_cache()

    total_chunks = math.ceil(total_tokens / args.chunk_size)
    rows_all: list[dict[str, Any]] = []
    rows_by_head: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    token_csv_path = output_dir / "retrieval_energy_by_token.csv"
    wrote_token_rows = False
    past_key_values = None

    for chunk_idx, start in enumerate(range(0, total_tokens, args.chunk_size), start=1):
        end = min(start + args.chunk_size, total_tokens)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": True,
            "output_hidden_states": True,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        print(f"tree retrieval energy chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        with torch.inference_mode():
            outputs = model_forward(model, kwargs)
        if outputs.attentions is None:
            raise RuntimeError("Model did not return attentions. Use ATTN_IMPLEMENTATION=eager.")
        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        past_key_values = outputs.past_key_values

        chunk_rows: list[dict[str, Any]] = []
        for local_query in range(end - start):
            query_token = start + local_query
            visible = query_token + 1
            for layer in layer_indices:
                attention_layer = outputs.attentions[layer][0].detach().float().cpu()
                query_states = compute_query_states(
                    model,
                    layer,
                    outputs.hidden_states[layer],
                    local_query,
                    query_token,
                    attention_heads,
                )
                per_head_candidates: dict[int, TreeCandidateResult] = {}
                for kv_head in kv_head_indices:
                    prefix_sum = key_prefix_sums[(layer, kv_head)]
                    shared_heads = attention_heads_by_kv[kv_head]
                    for attention_head in shared_heads:
                        per_head_candidates[attention_head] = tree_candidate_set(
                            query_token,
                            query_states[attention_head],
                            prefix_sum,
                            layout,
                            branch_counts,
                            args.boundary_fraction,
                        )
                    if args.candidate_granularity == "kv_head_union":
                        union_candidates: set[int] = set()
                        selected_big = selected_mid = selected_leaf = tree_middle = 0
                        prefix: set[int] = set()
                        recent: set[int] = set()
                        for attention_head in shared_heads:
                            result = per_head_candidates[attention_head]
                            union_candidates.update(result.candidates)
                            prefix = result.prefix
                            recent = result.recent
                            selected_big = max(selected_big, result.selected_big_count)
                            selected_mid = max(selected_mid, result.selected_mid_count)
                            selected_leaf = max(selected_leaf, result.selected_leaf_count)
                            tree_middle = max(tree_middle, result.tree_middle_count)
                        union_result = TreeCandidateResult(
                            candidates=union_candidates,
                            prefix=prefix,
                            recent=recent,
                            selected_big_count=selected_big,
                            selected_mid_count=selected_mid,
                            selected_leaf_count=selected_leaf,
                            tree_middle_count=tree_middle,
                        )
                        for attention_head in shared_heads:
                            per_head_candidates[attention_head] = union_result

                    for attention_head in shared_heads:
                        result = per_head_candidates[attention_head]
                        attention = attention_layer[attention_head, local_query, :visible]
                        prefix_recent = result.prefix | result.recent
                        method = energy_for_indices(attention, result.candidates)
                        oracle = oracle_energy(attention, len(result.candidates), visible)
                        pr_energy = energy_for_indices(attention, prefix_recent)
                        row = {
                            "layer": layer,
                            "kv_head": kv_head,
                            "attention_head": attention_head,
                            "query_token": query_token,
                            "visible_tokens": visible,
                            "candidate_count": len(result.candidates),
                            "candidate_fraction": len(result.candidates) / visible,
                            "prefix_count": len(result.prefix),
                            "recent_count": len(result.recent),
                            "selected_big_count": result.selected_big_count,
                            "selected_mid_count": result.selected_mid_count,
                            "selected_leaf_count": result.selected_leaf_count,
                            "tree_middle_count": result.tree_middle_count,
                            "method_energy": method,
                            "oracle_energy": oracle,
                            "prefix_recent_energy": pr_energy,
                            "energy_gap_to_oracle": oracle - method,
                        }
                        chunk_rows.append(row)
                        rows_all.append(row)
                        if args.make_plots:
                            rows_by_head[(layer, kv_head, attention_head)].append(row)

        if args.save_token_rows and chunk_rows:
            append_rows(token_csv_path, chunk_rows, token_row_fields(), append=wrote_token_rows)
            wrote_token_rows = True
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = summarize_rows(rows_all)
    write_csv(output_dir / "summary_by_head.csv", summary_rows, summary_fields())
    plot_paths = plot_head_rows(rows_by_head, output_dir, args.plot_dpi) if args.make_plots else []
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "tokens": total_tokens,
                    "layers": layer_indices,
                    "kv_heads": kv_head_indices,
                    "attention_heads_by_kv": attention_heads_by_kv,
                    "leaf_size": leaf_size,
                    "tree_fanout": args.tree_fanout,
                    "tree_branch_counts": branch_counts,
                    "leaf_count": len(layout.leaf_ranges),
                    "mid_count": len(layout.mid_ranges),
                    "big_count": len(layout.big_ranges),
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
