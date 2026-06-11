from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
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
DEFAULT_TEXT_PATH = "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_MODULE_TO_LAYER: dict[int, int] = {}
_ACTIVE_CONTEXT: AnalysisContext | None = None


@dataclass
class AnalysisConfig:
    mode: str
    layers: set[int]
    heads: set[int]
    basis_source_tokens: int
    basis_sample_mode: str
    basis_sample_seed: int
    svd_components: int
    query_stride: int
    max_query_rows_per_layer_head: int
    max_basis_vectors_per_layer_head: int
    top_ratios: list[float]
    tail_ratios: list[float]


class VectorStore:
    def __init__(self, max_vectors: int) -> None:
        self.max_vectors = max_vectors
        self.total_seen = 0
        self.stored = 0
        self.chunks: list[torch.Tensor] = []

    def add(self, vectors: torch.Tensor) -> None:
        if vectors.numel() == 0:
            return
        vectors = vectors.detach().float().cpu().reshape(-1, vectors.shape[-1])
        self.total_seen += int(vectors.shape[0])
        remaining = self.max_vectors - self.stored
        if remaining <= 0:
            return
        kept = vectors[:remaining].contiguous()
        self.chunks.append(kept)
        self.stored += int(kept.shape[0])

    def tensor(self) -> torch.Tensor:
        if not self.chunks:
            return torch.empty(0, 0, dtype=torch.float32)
        return torch.cat(self.chunks, dim=0)


class ProjectionAccumulator:
    def __init__(self, component_count: int) -> None:
        self.component_count = component_count
        self.count = 0
        self.energy_sum = torch.zeros(component_count, dtype=torch.float64)
        self.abs_coord_sum = torch.zeros(component_count, dtype=torch.float64)
        self.output_norm_sum = 0.0

    def update(self, coords: torch.Tensor, output: torch.Tensor) -> None:
        coords = coords.detach().float().cpu().reshape(-1)
        if coords.numel() == 0:
            return
        energy = coords * coords
        total = float(energy.sum())
        if total <= 1e-20:
            return
        self.count += 1
        self.energy_sum[: coords.numel()] += (energy / total).double()
        self.abs_coord_sum[: coords.numel()] += coords.abs().double()
        self.output_norm_sum += float(torch.linalg.vector_norm(output.detach().float().cpu()))

    def row(self, layer: int, head: int, selection: str, component: int) -> dict[str, Any]:
        index = component - 1
        denom = max(1, self.count)
        return {
            "layer": layer,
            "head": head,
            "selection": selection,
            "component": component,
            "query_count": self.count,
            "energy_mean": float(self.energy_sum[index] / denom),
            "abs_coord_mean": float(self.abs_coord_sum[index] / denom),
            "output_norm_mean": self.output_norm_sum / denom,
        }


class AnalysisContext:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.basis_vectors: dict[tuple[int, int], VectorStore] = {}
        self.svd_mean: dict[tuple[int, int], torch.Tensor] = {}
        self.svd_basis: dict[tuple[int, int], torch.Tensor] = {}
        self.singular_values: dict[tuple[int, int], torch.Tensor] = {}
        self.query_counts: dict[tuple[int, int], int] = {}
        self.projections: dict[tuple[int, int, str], ProjectionAccumulator] = {}

    def vector_store(self, layer: int, head: int) -> VectorStore:
        key = (layer, head)
        if key not in self.basis_vectors:
            self.basis_vectors[key] = VectorStore(self.config.max_basis_vectors_per_layer_head)
        return self.basis_vectors[key]

    def should_collect_query(self, layer: int, head: int) -> bool:
        key = (layer, head)
        seen = self.query_counts.get(key, 0)
        if seen >= self.config.max_query_rows_per_layer_head:
            return False
        self.query_counts[key] = seen + 1
        return True

    def projection_accumulator(self, layer: int, head: int, selection: str) -> ProjectionAccumulator:
        key = (layer, head, selection)
        if key not in self.projections:
            self.projections[key] = ProjectionAccumulator(self.config.svd_components)
        return self.projections[key]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed per-layer/head V SVD basis, then project top/tail weighted V outputs."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/score_tail_svd_representation")
    parser.add_argument("--basis_tokens", type=int, default=5000)
    parser.add_argument(
        "--basis_sample_mode",
        choices=["random", "even"],
        default="random",
        help="Sample V basis vectors randomly or at evenly spaced positions from the basis token pool.",
    )
    parser.add_argument("--basis_sample_seed", type=int, default=0)
    parser.add_argument("--prefill_tokens", type=int, default=5000)
    parser.add_argument("--eval_tokens", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--svd_components", type=int, default=16)
    parser.add_argument("--query_stride", type=int, default=8)
    parser.add_argument("--max_query_rows_per_layer_head", type=int, default=512)
    parser.add_argument("--max_basis_vectors_per_layer_head", type=int, default=5000)
    parser.add_argument("--top_ratios", default="0.01,0.02,0.04,0.08,0.16,0.30,0.50,0.90")
    parser.add_argument("--tail_ratios", default="0.10,0.30,0.50")
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--make_head_plots", type=str2bool, default=True)
    parser.add_argument("--max_head_plots", type=int, default=0, help="0 means no limit.")
    args = parser.parse_args()
    if args.basis_tokens <= 0:
        raise ValueError("--basis_tokens must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    if args.svd_components <= 0:
        raise ValueError("--svd_components must be positive.")
    if args.query_stride <= 0:
        raise ValueError("--query_stride must be positive.")
    return args


def parse_ratios(spec: str, name: str) -> list[float]:
    ratios = [float(part.strip()) for part in spec.split(",") if part.strip()]
    invalid = [ratio for ratio in ratios if ratio <= 0.0 or ratio >= 1.0]
    if invalid:
        raise ValueError(f"{name} ratios must be in (0, 1): {invalid}")
    return ratios


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
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(part))
    invalid = sorted(index for index in selected if index < 0 or index >= max_count)
    if invalid:
        raise ValueError(f"{name} out of range 0..{max_count - 1}: {invalid}")
    return sorted(selected)


def ratio_label(prefix: str, ratio: float) -> str:
    value = ratio * 100.0
    if abs(value - round(value)) < 1e-8:
        return f"{prefix}{int(round(value))}"
    return f"{prefix}{value:g}".replace(".", "p")


def selection_labels(top_ratios: list[float], tail_ratios: list[float]) -> list[str]:
    return [ratio_label("top", ratio) for ratio in top_ratios] + [ratio_label("tail", ratio) for ratio in tail_ratios]


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def model_body(model: torch.nn.Module) -> torch.nn.Module:
    for attr_name in ("model", "transformer"):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
    return model


def register_attention_layers(model: torch.nn.Module) -> None:
    _MODULE_TO_LAYER.clear()
    layers = getattr(model_body(model), "layers", None)
    if layers is None:
        raise AttributeError("Could not find transformer layers on model.")
    for layer_idx, layer_obj in enumerate(layers):
        attn = getattr(layer_obj, "self_attn", None) or getattr(layer_obj, "attention", None)
        if attn is not None:
            _MODULE_TO_LAYER[id(attn)] = layer_idx


def repeat_kv_for_attention(states: torch.Tensor, attention_heads: int) -> torch.Tensor:
    if states.shape[1] == attention_heads:
        return states
    group_size = attention_heads // states.shape[1]
    return states.repeat_interleave(group_size, dim=1)


def basis_sample_indices(
    key_count: int,
    sample_count: int,
    layer: int,
    head: int,
    config: AnalysisConfig,
    device: torch.device,
) -> torch.Tensor:
    if config.basis_sample_mode == "even":
        return torch.linspace(0, key_count - 1, steps=sample_count, device=device).round().long()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.basis_sample_seed + layer * 1009 + head * 9176)
    indices = torch.randperm(key_count, generator=generator)[:sample_count].sort().values
    return indices.to(device=device)


def collect_basis_vectors(context: AnalysisContext, layer: int, value_states: torch.Tensor) -> None:
    config = context.config
    if layer not in config.layers:
        return
    _, head_count, key_count, _ = value_states.shape
    if key_count < config.basis_source_tokens:
        return
    for head in sorted(config.heads):
        if head >= head_count:
            continue
        store = context.vector_store(layer, head)
        if store.stored > 0:
            continue
        take = min(config.max_basis_vectors_per_layer_head, key_count)
        indices = basis_sample_indices(key_count, take, layer, head, config, value_states.device)
        store.add(value_states[0, head, indices])


def masked_weighted_output(
    scores: torch.Tensor,
    values: torch.Tensor,
    local_indices: torch.Tensor,
) -> torch.Tensor:
    selected_scores = scores[local_indices].float()
    selected_values = values[local_indices].float()
    weights = F.softmax(selected_scores, dim=-1)
    return torch.matmul(weights, selected_values)


def project_selected_outputs(
    context: AnalysisContext,
    layer: int,
    scores: torch.Tensor,
    attention_weights: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    config = context.config
    if layer not in config.layers:
        return
    _, head_count, query_count, _ = scores.shape
    query_indices = list(range(0, query_count, config.query_stride))
    for head in sorted(config.heads):
        if head >= head_count or (layer, head) not in context.svd_basis:
            continue
        mean = context.svd_mean[(layer, head)].to(scores.device)
        basis = context.svd_basis[(layer, head)].to(scores.device)
        for query_index in query_indices:
            if not context.should_collect_query(layer, head):
                break
            row_weights = attention_weights[0, head, query_index].float()
            valid_positions = torch.nonzero(row_weights > 0, as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue
            row_scores = scores[0, head, query_index, valid_positions].float()
            row_values = value_states[0, head, valid_positions].float()
            order_desc = torch.argsort(row_scores, descending=True)
            valid_count = int(order_desc.numel())
            for ratio in config.top_ratios:
                count = max(1, math.ceil(valid_count * ratio))
                label = ratio_label("top", ratio)
                output = masked_weighted_output(row_scores, row_values, order_desc[:count])
                coords = (output - mean) @ basis
                context.projection_accumulator(layer, head, label).update(coords, output)
            for ratio in config.tail_ratios:
                count = max(1, math.ceil(valid_count * ratio))
                label = ratio_label("tail", ratio)
                output = masked_weighted_output(row_scores, row_values, order_desc[-count:])
                coords = (output - mean) @ basis
                context.projection_accumulator(layer, head, label).update(coords, output)


def patched_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = _ACTIVE_CONTEXT
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    attention_heads = query_states.shape[1]
    key_states_for_attention = repeat_kv_for_attention(key_states, attention_heads)
    value_states_for_attention = repeat_kv_for_attention(value_states, attention_heads)
    scores = torch.matmul(query_states, key_states_for_attention.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    layer = _MODULE_TO_LAYER.get(id(module), -1)
    if context is not None and context.config.mode == "basis":
        collect_basis_vectors(context, layer, value_states_for_attention)
    elif context is not None and context.config.mode == "projection":
        project_selected_outputs(context, layer, scores, attention_weights, value_states_for_attention)
    attention_output = torch.matmul(attention_weights, value_states_for_attention)
    return attention_output.transpose(1, 2).contiguous(), attention_weights


def install_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is not None:
        return
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3.") from exc
    _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
    setattr(modeling_qwen3, "eager_attention_forward", patched_eager_attention_forward)
    if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
        modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = patched_eager_attention_forward


@contextmanager
def active_context(context: AnalysisContext | None):
    global _ACTIVE_CONTEXT
    previous = _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = context
    try:
        yield
    finally:
        _ACTIVE_CONTEXT = previous


@torch.inference_mode()
def run_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    start_token: int,
    token_count: int,
    chunk_size: int,
    input_device: torch.device,
    context: AnalysisContext | None,
    use_cache: bool = True,
) -> Any:
    past_key_values = None
    end_token = start_token + token_count
    total_chunks = math.ceil(token_count / chunk_size)
    with active_context(context):
        for chunk_idx, start in enumerate(range(start_token, end_token, chunk_size), start=1):
            end = min(start + chunk_size, end_token)
            chunk = input_ids[:, start:end].to(input_device)
            kwargs: dict[str, Any] = {
                "input_ids": chunk,
                "use_cache": use_cache,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "cache_position": torch.arange(start, end, device=input_device),
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            print(f"{context.config.mode if context else 'run'} chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
            outputs = model_forward(model, kwargs)
            past_key_values = outputs.past_key_values if use_cache else None
            del outputs, chunk
    return past_key_values


def build_svd_basis(context: AnalysisContext, layers: list[int], heads: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for head in heads:
            store = context.basis_vectors.get((layer, head))
            if store is None:
                continue
            matrix = store.tensor()
            if matrix.numel() == 0:
                continue
            mean = matrix.mean(dim=0)
            centered = matrix - mean.unsqueeze(0)
            _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            component_count = min(context.config.svd_components, int(vh.shape[0]))
            context.svd_mean[(layer, head)] = mean
            context.svd_basis[(layer, head)] = vh[:component_count].T.contiguous()
            context.singular_values[(layer, head)] = singular_values[:component_count].contiguous()
            total_energy = float((singular_values * singular_values).sum())
            cumulative = 0.0
            for component, singular_value in enumerate(singular_values[:component_count].tolist(), start=1):
                value = float(singular_value)
                ratio = (value * value) / max(total_energy, 1e-12)
                cumulative += ratio
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "component": component,
                        "singular_value": value,
                        "energy_ratio": ratio,
                        "cumulative_energy_ratio": cumulative,
                        "basis_vector_count": int(matrix.shape[0]),
                        "basis_total_seen": store.total_seen,
                    }
                )
    return rows


def projection_rows(
    context: AnalysisContext,
    layers: list[int],
    heads: list[int],
    selections: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for head in heads:
            for selection in selections:
                accumulator = context.projections.get((layer, head, selection))
                if accumulator is None:
                    continue
                for component in range(1, context.config.svd_components + 1):
                    rows.append(accumulator.row(layer, head, selection, component))
    return rows


def aggregate_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        bucket = grouped.setdefault(
            key,
            {
                **{field: row[field] for field in group_fields},
                "query_count": 0,
                "energy_weighted_sum": 0.0,
                "abs_coord_weighted_sum": 0.0,
                "output_norm_weighted_sum": 0.0,
            },
        )
        count = int(row["query_count"])
        bucket["query_count"] += count
        bucket["energy_weighted_sum"] += float(row["energy_mean"]) * count
        bucket["abs_coord_weighted_sum"] += float(row["abs_coord_mean"]) * count
        bucket["output_norm_weighted_sum"] += float(row["output_norm_mean"]) * count
    result: list[dict[str, Any]] = []
    for bucket in grouped.values():
        count = max(1, int(bucket["query_count"]))
        result.append(
            {
                **{field: bucket[field] for field in group_fields},
                "query_count": bucket["query_count"],
                "energy_mean": bucket["energy_weighted_sum"] / count,
                "abs_coord_mean": bucket["abs_coord_weighted_sum"] / count,
                "output_norm_mean": bucket["output_norm_weighted_sum"] / count,
            }
        )
    return sorted(result, key=lambda item: tuple(item[field] for field in group_fields))


def save_basis_pt(context: AnalysisContext, path: Path) -> None:
    payload = {
        "mean": context.svd_mean,
        "basis": context.svd_basis,
        "singular_values": context.singular_values,
    }
    torch.save(payload, path)


def color_for_selection(selection: str, selections: list[str]) -> Any:
    import matplotlib.pyplot as plt

    top = [item for item in selections if item.startswith("top")]
    tail = [item for item in selections if item.startswith("tail")]
    if selection.startswith("top"):
        idx = top.index(selection) if selection in top else 0
        denom = max(1, len(top) - 1)
        return plt.cm.Blues(0.35 + 0.60 * idx / denom)
    idx = tail.index(selection) if selection in tail else 0
    denom = max(1, len(tail) - 1)
    return plt.cm.Reds(0.35 + 0.60 * idx / denom)


def plot_energy_curves(
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
    selections: list[str],
    subset: list[str],
) -> None:
    import matplotlib.pyplot as plt

    by_selection: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_selection.setdefault(str(row["selection"]), []).append(row)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=180)
    for selection in subset:
        items = sorted(by_selection.get(selection, []), key=lambda item: int(item["component"]))
        if not items:
            continue
        xs = [int(item["component"]) for item in items]
        ys = [float(item["energy_mean"]) for item in items]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, color=color_for_selection(selection, selections), label=selection)
    ax.set_title(title)
    ax.set_xlabel("SVD projection index")
    ax.set_ylabel("Projection energy ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_singular_values(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    by_component: dict[int, list[float]] = {}
    for row in rows:
        by_component.setdefault(int(row["component"]), []).append(float(row["energy_ratio"]))
    components = sorted(by_component)
    values = [sum(by_component[component]) / len(by_component[component]) for component in components]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=180)
    ax.plot(components, values, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("SVD component index")
    ax.set_ylabel("Mean singular-value energy ratio")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_plots(
    output_dir: Path,
    projection_by_head: list[dict[str, Any]],
    projection_by_layer: list[dict[str, Any]],
    projection_global: list[dict[str, Any]],
    singular_rows: list[dict[str, Any]],
    selections: list[str],
    make_head_plots: bool,
    max_head_plots: int,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    subsets = {
        "all": selections,
        "top_sparse": [item for item in ("top1", "top4", "top16", "top50", "top90") if item in selections],
        "top_low": [item for item in ("top1", "top2", "top4", "top8", "top16") if item in selections],
        "tail_only": [item for item in ("tail10", "tail30", "tail50") if item in selections],
    }
    for name, subset in subsets.items():
        if subset:
            plot_energy_curves(projection_global, plots_dir / f"global_projection_energy_{name}.png", f"Global projection energy ({name})", selections, subset)
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in projection_by_layer:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    for layer, rows in by_layer.items():
        for name, subset in subsets.items():
            if subset:
                plot_energy_curves(rows, plots_dir / f"layer_{layer:02d}_projection_energy_{name}.png", f"Layer {layer} projection energy ({name})", selections, subset)
    if make_head_plots:
        by_head: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in projection_by_head:
            by_head.setdefault((int(row["layer"]), int(row["head"])), []).append(row)
        for index, ((layer, head), rows) in enumerate(sorted(by_head.items())):
            if max_head_plots > 0 and index >= max_head_plots:
                break
            for name, subset in subsets.items():
                if subset:
                    plot_energy_curves(
                        rows,
                        plots_dir / f"layer_{layer:02d}_head_{head:02d}_projection_energy_{name}.png",
                        f"Layer {layer} head {head} projection energy ({name})",
                        selections,
                        subset,
                    )
    plot_singular_values(singular_rows, plots_dir / "singular_value_energy_global.png", "Global SVD singular-value energy")
    by_svd_layer: dict[int, list[dict[str, Any]]] = {}
    for row in singular_rows:
        by_svd_layer.setdefault(int(row["layer"]), []).append(row)
    for layer, rows in by_svd_layer.items():
        plot_singular_values(rows, plots_dir / f"layer_{layer:02d}_singular_value_energy.png", f"Layer {layer} SVD singular-value energy")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_ratios = parse_ratios(args.top_ratios, "top")
    tail_ratios = parse_ratios(args.tail_ratios, "tail")
    selections = selection_labels(top_ratios, tail_ratios)

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    total_tokens_needed = max(args.basis_tokens, args.prefill_tokens + args.eval_tokens)
    if args.require_total_tokens and len(token_ids) < total_tokens_needed:
        raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than {total_tokens_needed}.")
    input_ids = torch.tensor(token_ids[:total_tokens_needed], dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(args.dtype, requested_device)}
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

    register_attention_layers(model)
    install_attention_patch()
    layer_count = int(getattr(model.config, "num_hidden_layers"))
    head_count = int(getattr(model.config, "num_attention_heads"))
    layers = parse_index_spec(args.layers, layer_count, "layers")
    heads = parse_index_spec(args.heads, head_count, "heads")

    basis_config = AnalysisConfig(
        mode="basis",
        layers=set(layers),
        heads=set(heads),
        basis_source_tokens=args.basis_tokens,
        basis_sample_mode=args.basis_sample_mode,
        basis_sample_seed=args.basis_sample_seed,
        svd_components=args.svd_components,
        query_stride=args.query_stride,
        max_query_rows_per_layer_head=args.max_query_rows_per_layer_head,
        max_basis_vectors_per_layer_head=args.max_basis_vectors_per_layer_head,
        top_ratios=top_ratios,
        tail_ratios=tail_ratios,
    )
    context = AnalysisContext(basis_config)
    print("stage 1: collecting V basis samples", flush=True)
    run_tokens(model, input_ids, 0, args.basis_tokens, args.chunk_size, input_device, context, use_cache=True)
    singular_rows = build_svd_basis(context, layers, heads)
    save_basis_pt(context, output_dir / "svd_basis.pt")

    context.config.mode = "projection"
    context.query_counts.clear()
    print("stage 2: running projection experiment", flush=True)
    run_tokens(model, input_ids, 0, args.prefill_tokens + args.eval_tokens, args.chunk_size, input_device, context, use_cache=True)

    projection_by_head = projection_rows(context, layers, heads, selections)
    projection_by_layer = aggregate_rows(projection_by_head, ["layer", "selection", "component"])
    projection_global = aggregate_rows(projection_by_head, ["selection", "component"])

    projection_fields = ["layer", "head", "selection", "component", "query_count", "energy_mean", "abs_coord_mean", "output_norm_mean"]
    projection_layer_fields = ["layer", "selection", "component", "query_count", "energy_mean", "abs_coord_mean", "output_norm_mean"]
    projection_global_fields = ["selection", "component", "query_count", "energy_mean", "abs_coord_mean", "output_norm_mean"]
    singular_fields = [
        "layer",
        "head",
        "component",
        "singular_value",
        "energy_ratio",
        "cumulative_energy_ratio",
        "basis_vector_count",
        "basis_total_seen",
    ]
    write_csv(output_dir / "projection_energy_by_layer_head.csv", projection_by_head, projection_fields)
    write_csv(output_dir / "projection_energy_by_layer.csv", projection_by_layer, projection_layer_fields)
    write_csv(output_dir / "projection_energy_global.csv", projection_global, projection_global_fields)
    write_csv(output_dir / "singular_value_energy.csv", singular_rows, singular_fields)

    if args.make_plots:
        make_plots(
            output_dir,
            projection_by_head,
            projection_by_layer,
            projection_global,
            singular_rows,
            selections,
            args.make_head_plots,
            args.max_head_plots,
        )

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "text_path": args.text_path,
        "basis_tokens": args.basis_tokens,
        "basis_sample_mode": args.basis_sample_mode,
        "basis_sample_seed": args.basis_sample_seed,
        "prefill_tokens": args.prefill_tokens,
        "eval_tokens": args.eval_tokens,
        "chunk_size": args.chunk_size,
        "layers": layers,
        "heads": heads,
        "svd_components": args.svd_components,
        "top_ratios": top_ratios,
        "tail_ratios": tail_ratios,
        "outputs": {
            "svd_basis": str(output_dir / "svd_basis.pt"),
            "singular_value_energy": str(output_dir / "singular_value_energy.csv"),
            "projection_energy_by_layer_head": str(output_dir / "projection_energy_by_layer_head.csv"),
            "projection_energy_by_layer": str(output_dir / "projection_energy_by_layer.csv"),
            "projection_energy_global": str(output_dir / "projection_energy_global.csv"),
            "plots": str(output_dir / "plots"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
