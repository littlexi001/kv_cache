from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from coselection_analysis import (  # noqa: E402
    edge_to_dict,
    extract_pair_edges,
    pair_metric_matrices,
    summarize_head,
)


def _install_torchvision_fake_registration_guard() -> None:
    register_fake = getattr(torch.library, "register_fake", None)
    if register_fake is None or getattr(register_fake, "_top2_coselection_guarded", False):
        return

    def guarded_register_fake(op_name: str, *args: Any, **kwargs: Any):
        decorator = register_fake(op_name, *args, **kwargs)

        def guarded_decorator(fn: Any) -> Any:
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if "operator torchvision::" in str(exc) and "does not exist" in str(exc):
                    return fn
                raise

        return guarded_decorator

    guarded_register_fake._top2_coselection_guarded = True
    torch.library.register_fake = guarded_register_fake


_install_torchvision_fake_registration_guard()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "ymluo/models/Qwen3-0.6B"
DEFAULT_TEXT_PATH = "ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt"

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "Top2CoselectionCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-head N x N Top-2% token co-selection matrices across query observations."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/top2_coselection")
    parser.add_argument("--context_tokens", type=int, default=1024)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--text_token_offset", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--query_stride", type=int, default=1)
    parser.add_argument("--max_query_samples", type=int, default=0)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--analysis_device", default="cuda")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--min_token_count", type=int, default=8)
    parser.add_argument("--min_pair_count", type=int, default=4)
    parser.add_argument("--fdr_alpha", type=float, default=0.01)
    parser.add_argument("--top_pairs_per_head", type=int, default=30)
    parser.add_argument("--top_tokens_per_head", type=int, default=30)
    parser.add_argument("--representative_heads", type=int, default=6)
    parser.add_argument("--heatmap_tokens", type=int, default=128)
    parser.add_argument("--save_selection_indices", type=str2bool, default=True)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def parse_index_spec(spec: str, count: int) -> list[int]:
    normalized = spec.strip().lower()
    if normalized in {"all", "*"}:
        return list(range(count))
    values: set[int] = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            end = int(right)
            values.update(range(start, end + 1))
        else:
            values.add(int(item))
    ordered = sorted(values)
    if not ordered or ordered[0] < 0 or ordered[-1] >= count:
        raise ValueError(f"Index specification {spec!r} is outside [0, {count}).")
    return ordered


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(max_chars) if max_chars > 0 else handle.read()


def pick_input_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def model_forward(model: torch.nn.Module, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" in kwargs and "cache_position" in str(exc):
            reduced = dict(kwargs)
            reduced.pop("cache_position")
            return model(**reduced)
        raise


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_query_tokens(context_tokens: int, eval_tokens: int, stride: int, maximum: int) -> list[int]:
    if stride <= 0:
        raise ValueError("query_stride must be positive.")
    queries = list(range(context_tokens, context_tokens + eval_tokens, stride))
    if maximum > 0 and len(queries) > maximum:
        if maximum == 1:
            return [queries[len(queries) // 2]]
        positions = np.linspace(0, len(queries) - 1, maximum)
        queries = [queries[int(round(position))] for position in positions]
    return sorted(set(queries))


def safe_token_piece(tokenizer: Any, token_id: int) -> tuple[str, str]:
    try:
        piece = str(tokenizer.convert_ids_to_tokens(int(token_id)))
    except Exception:
        piece = ""
    try:
        text = str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        text = ""
    return piece.replace("\n", "\\n").replace("\r", "\\r"), text.replace("\n", "\\n").replace("\r", "\\r")


class Top2CoselectionCollector:
    def __init__(
        self,
        selected_layers: list[int],
        selected_heads: list[int],
        query_tokens: list[int],
        context_tokens: int,
        budget: int,
    ) -> None:
        self.selected_layers = selected_layers
        self.selected_heads = selected_heads
        self.selected_layer_set = set(selected_layers)
        self.query_tokens = query_tokens
        self.query_slot = {query: index for index, query in enumerate(query_tokens)}
        self.context_tokens = context_tokens
        self.budget = budget
        self.layer_slot = {layer: index for index, layer in enumerate(selected_layers)}
        self.indices = np.full(
            (len(selected_layers), len(selected_heads), len(query_tokens), budget),
            -1,
            dtype=np.int32,
        )
        self.observed = np.zeros((len(selected_layers), len(query_tokens)), dtype=np.uint8)

    @torch.inference_mode()
    def observe(self, layer: int, query_token: int, scores: torch.Tensor, query_index: int) -> None:
        if layer not in self.selected_layer_set or query_token not in self.query_slot:
            return
        if scores.shape[0] != 1:
            raise ValueError("Co-selection collection currently requires batch size 1.")
        prefix_scores = scores[0, self.selected_heads, query_index, : self.context_tokens]
        if prefix_scores.shape[-1] != self.context_tokens:
            raise ValueError("The fixed context window is not fully visible to the sampled query.")
        top_indices = torch.topk(prefix_scores, k=self.budget, dim=-1, largest=True, sorted=False).indices
        layer_slot = self.layer_slot[layer]
        query_slot = self.query_slot[query_token]
        self.indices[layer_slot, :, query_slot, :] = top_indices.to("cpu", dtype=torch.int32).numpy()
        self.observed[layer_slot, query_slot] = 1

    def validate(self) -> None:
        if np.any(self.observed == 0):
            missing = np.argwhere(self.observed == 0)
            preview = [
                {
                    "layer": self.selected_layers[int(layer_slot)],
                    "query": self.query_tokens[int(query_slot)],
                }
                for layer_slot, query_slot in missing[:20]
            ]
            raise RuntimeError(f"Missing layer/query observations: {preview}")
        if np.any(self.indices < 0):
            raise RuntimeError("Selection index tensor contains unfilled entries.")


def _patched_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]

    if _ACTIVE_COLLECTOR is not None:
        layer = int(getattr(module, "layer_idx", 0))
        query_count = scores.shape[-2]
        key_count = scores.shape[-1]
        query_start = key_count - query_count
        for query_index in range(query_count):
            _ACTIVE_COLLECTOR.observe(layer, query_start + query_index, scores, query_index)

    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def install_qwen3_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3.") from exc
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", _patched_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _patched_eager_attention_forward


@contextmanager
def active_collector(collector: Top2CoselectionCollector):
    global _ACTIVE_COLLECTOR
    previous = _ACTIVE_COLLECTOR
    _ACTIVE_COLLECTOR = collector
    try:
        yield
    finally:
        _ACTIVE_COLLECTOR = previous


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    context_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> Any:
    past_key_values = None
    total_chunks = math.ceil(context_tokens / chunk_size)
    for chunk_number, start in enumerate(range(0, context_tokens, chunk_size), start=1):
        end = min(start + chunk_size, context_tokens)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(input_device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        print(f"prefill {chunk_number}/{total_chunks}: {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        del outputs
    return past_key_values


@torch.inference_mode()
def run_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    context_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: Top2CoselectionCollector,
) -> None:
    end_token = context_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    with active_collector(collector):
        for chunk_number, start in enumerate(range(context_tokens, end_token, chunk_size), start=1):
            end = min(start + chunk_size, end_token)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids[:, start:end].to(input_device),
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "cache_position": torch.arange(start, end, device=input_device),
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            print(f"eval {chunk_number}/{total_chunks}: {start}-{end - 1}", flush=True)
            outputs = model_forward(model, kwargs)
            past_key_values = outputs.past_key_values
            del outputs


def batched_cooccurrence(
    layer_indices: np.ndarray,
    token_count: int,
    analysis_device: torch.device,
) -> np.ndarray:
    # layer_indices: [heads, observations, budget]
    indices = torch.from_numpy(layer_indices.astype(np.int64, copy=False)).to(analysis_device)
    head_count, observations, _ = indices.shape
    dtype = torch.float16 if analysis_device.type == "cuda" else torch.float32
    incidence = torch.zeros((head_count, observations, token_count), dtype=dtype, device=analysis_device)
    incidence.scatter_(2, indices, 1.0)
    cooccurrence = torch.bmm(incidence.transpose(1, 2), incidence)
    result = cooccurrence.round().to(torch.int32).cpu().numpy()
    del indices, incidence, cooccurrence
    if analysis_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate_layer_rows(head_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in head_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    metrics = [
        "positive_excess_pair_mass_fraction",
        "significant_positive_pairs",
        "significant_fraction_of_coobserved_pairs",
        "significant_conditional_median",
        "significant_lift_median",
        "largest_component_tokens",
        "distance_le_16_enrichment",
        "cluster_score",
    ]
    rows: list[dict[str, Any]] = []
    for layer, members in sorted(by_layer.items()):
        row: dict[str, Any] = {
            "layer": layer,
            "heads": len(members),
            "heads_with_significant_pairs": sum(int(member["significant_positive_pairs"]) > 0 for member in members),
        }
        for metric in metrics:
            values = np.asarray([float(member[metric]) for member in members], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return rows


def choose_representatives(head_rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not head_rows:
        return []
    ordered = sorted(head_rows, key=lambda row: float(row["cluster_score"]))
    requested = min(count, len(ordered))
    candidate_indices: list[int] = []
    top_count = max(1, requested // 2)
    candidate_indices.extend(range(len(ordered) - top_count, len(ordered)))
    remaining = requested - len(candidate_indices)
    if remaining > 0:
        for position in np.linspace(0, len(ordered) - 1, remaining + 2)[1:-1]:
            candidate_indices.append(int(round(position)))
    chosen: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index in reversed(candidate_indices):
        row = ordered[index]
        key = (int(row["layer"]), int(row["head"]))
        if key not in seen:
            seen.add(key)
            chosen.append(row)
    for row in reversed(ordered):
        if len(chosen) >= requested:
            break
        key = (int(row["layer"]), int(row["head"]))
        if key not in seen:
            seen.add(key)
            chosen.append(row)
    return chosen[:requested]


def plot_head_heatmap(
    output_path: Path,
    layer: int,
    head: int,
    selection_count: np.ndarray,
    cooccurrence: np.ndarray,
    observations: int,
    maximum_tokens: int,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    active = np.flatnonzero(selection_count > 0)
    if active.size == 0:
        return
    if active.size > maximum_tokens:
        ranked = active[np.argsort(selection_count[active])[-maximum_tokens:]]
    else:
        ranked = active
    sub_count = selection_count[ranked]
    sub_coocc = cooccurrence[np.ix_(ranked, ranked)]
    metrics = pair_metric_matrices(sub_count, sub_coocc, observations)
    jaccard = metrics["jaccard"]
    if ranked.size >= 3 and np.any(jaccard > 0):
        distance = np.clip(1.0 - jaccard, 0.0, 1.0)
        np.fill_diagonal(distance, 0.0)
        order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    else:
        order = np.argsort(ranked)
    ranked = ranked[order]
    conditional = metrics["conditional"][np.ix_(order, order)]
    lift = metrics["lift"][np.ix_(order, order)]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    image0 = axes[0].imshow(conditional, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[0].set_title(f"L{layer} H{head}: P(B selected | A selected)")
    figure.colorbar(image0, ax=axes[0], fraction=0.046)
    log_lift = np.log2(np.clip(lift, 0.25, 16.0))
    image1 = axes[1].imshow(log_lift, cmap="coolwarm", vmin=-2.0, vmax=4.0, interpolation="nearest")
    axes[1].set_title("log2 lift over marginal independence")
    figure.colorbar(image1, ax=axes[1], fraction=0.046)
    tick_step = max(1, ranked.size // 12)
    ticks = np.arange(0, ranked.size, tick_step)
    labels = [str(int(ranked[index])) for index in ticks]
    for axis in axes:
        axis.set_xticks(ticks, labels=labels, rotation=90, fontsize=7)
        axis.set_yticks(ticks, labels=labels, fontsize=7)
        axis.set_xlabel("token B (cluster ordered)")
        axis.set_ylabel("token A (cluster ordered)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1].")
    if args.context_tokens <= 1 or args.eval_tokens <= 0:
        raise ValueError("context_tokens must exceed 1 and eval_tokens must be positive.")
    budget = max(1, int(math.ceil(args.context_tokens * args.top_fraction)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    required_tokens = args.context_tokens + args.eval_tokens
    required_end = args.text_token_offset + required_tokens
    if args.text_token_offset < 0:
        raise ValueError("text_token_offset must be non-negative.")
    if args.require_total_tokens and len(token_ids) < required_end:
        raise ValueError(f"Need tokens through index {required_end}, got {len(token_ids)}.")
    token_ids = token_ids[args.text_token_offset : required_end]
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    install_qwen3_attention_patch()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)
    layer_count = int(model.config.num_hidden_layers)
    head_count = int(model.config.num_attention_heads)
    selected_layers = parse_index_spec(args.layers, layer_count)
    selected_heads = parse_index_spec(args.heads, head_count)
    query_tokens = sample_query_tokens(
        args.context_tokens,
        args.eval_tokens,
        args.query_stride,
        args.max_query_samples,
    )
    collector = Top2CoselectionCollector(
        selected_layers,
        selected_heads,
        query_tokens,
        args.context_tokens,
        budget,
    )

    run_started = time.perf_counter()
    past = prefill_cache(model, input_ids, args.context_tokens, args.chunk_size, input_device)
    run_eval(
        model,
        input_ids,
        past,
        args.context_tokens,
        args.eval_tokens,
        args.chunk_size,
        input_device,
        collector,
    )
    collection_seconds = time.perf_counter() - run_started
    collector.validate()
    del past, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    context_token_ids = np.asarray(token_ids[: args.context_tokens], dtype=np.int64)
    token_pieces = [safe_token_piece(tokenizer, int(token_id)) for token_id in context_token_ids]
    if args.save_selection_indices:
        np.savez_compressed(
            output_dir / "selection_indices.npz",
            indices=collector.indices,
            selected_layers=np.asarray(selected_layers, dtype=np.int32),
            selected_heads=np.asarray(selected_heads, dtype=np.int32),
            query_tokens=np.asarray(query_tokens, dtype=np.int32),
            context_token_ids=context_token_ids,
            budget=np.asarray([budget], dtype=np.int32),
        )

    analysis_device = torch.device(
        args.analysis_device if args.analysis_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    head_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    analysis_started = time.perf_counter()
    for layer_slot, layer in enumerate(selected_layers):
        print(f"analyze layer {layer} ({layer_slot + 1}/{len(selected_layers)})", flush=True)
        layer_cooccurrence = batched_cooccurrence(
            collector.indices[layer_slot], args.context_tokens, analysis_device
        )
        for head_slot, head in enumerate(selected_heads):
            cooccurrence = layer_cooccurrence[head_slot]
            selection_count = np.diag(cooccurrence).astype(np.int64, copy=True)
            edges = extract_pair_edges(
                selection_count,
                cooccurrence,
                len(query_tokens),
                min_token_count=args.min_token_count,
                min_pair_count=args.min_pair_count,
                fdr_alpha=args.fdr_alpha,
            )
            summary = summarize_head(
                selection_count,
                cooccurrence,
                len(query_tokens),
                budget,
                edges,
            )
            head_rows.append({"layer": layer, "head": head, **summary})

            for rank, edge in enumerate(edges[: args.top_pairs_per_head], start=1):
                row = {"layer": layer, "head": head, "rank": rank, **edge_to_dict(edge)}
                piece_a, text_a = token_pieces[edge.token_a]
                piece_b, text_b = token_pieces[edge.token_b]
                row.update(
                    {
                        "token_distance": abs(edge.token_b - edge.token_a),
                        "token_a_id": int(context_token_ids[edge.token_a]),
                        "token_a_piece": piece_a,
                        "token_a_text": text_a,
                        "token_b_id": int(context_token_ids[edge.token_b]),
                        "token_b_piece": piece_b,
                        "token_b_text": text_b,
                    }
                )
                pair_rows.append(row)

            active_positions = np.flatnonzero(selection_count > 0)
            top_positions = active_positions[
                np.argsort(selection_count[active_positions])[-args.top_tokens_per_head :][::-1]
            ]
            for rank, position in enumerate(top_positions, start=1):
                piece, text = token_pieces[int(position)]
                token_rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "rank": rank,
                        "token_position": int(position),
                        "token_id": int(context_token_ids[position]),
                        "token_piece": piece,
                        "token_text": text,
                        "selected_count": int(selection_count[position]),
                        "selected_probability": float(selection_count[position] / len(query_tokens)),
                    }
                )
        del layer_cooccurrence

    layer_rows = aggregate_layer_rows(head_rows)
    write_csv(output_dir / "head_coselection_summary.csv", head_rows)
    write_csv(output_dir / "layer_coselection_summary.csv", layer_rows)
    write_csv(output_dir / "top_coselected_pairs.csv", pair_rows)
    write_csv(output_dir / "top_selected_tokens.csv", token_rows)

    representatives = choose_representatives(head_rows, args.representative_heads)
    representative_rows: list[dict[str, Any]] = []
    matrix_dir = output_dir / "representative_matrices"
    figure_dir = output_dir / "figures"
    for row in representatives:
        layer = int(row["layer"])
        head = int(row["head"])
        layer_slot = selected_layers.index(layer)
        head_slot = selected_heads.index(head)
        cooccurrence = batched_cooccurrence(
            collector.indices[layer_slot, head_slot : head_slot + 1],
            args.context_tokens,
            analysis_device,
        )[0]
        selection_count = np.diag(cooccurrence).astype(np.int64, copy=True)
        matrix_path = matrix_dir / f"layer{layer:02d}_head{head:02d}.npz"
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            matrix_path,
            cooccurrence=cooccurrence.astype(np.uint16),
            selection_count=selection_count.astype(np.uint16),
            query_tokens=np.asarray(query_tokens, dtype=np.int32),
            context_token_ids=context_token_ids,
            budget=np.asarray([budget], dtype=np.int32),
        )
        figure_path = figure_dir / f"layer{layer:02d}_head{head:02d}_coselection.png"
        if args.make_plots:
            plot_head_heatmap(
                figure_path,
                layer,
                head,
                selection_count,
                cooccurrence,
                len(query_tokens),
                args.heatmap_tokens,
                args.plot_dpi,
            )
        representative_rows.append(
            {
                "layer": layer,
                "head": head,
                "cluster_score": row["cluster_score"],
                "significant_positive_pairs": row["significant_positive_pairs"],
                "matrix_path": str(matrix_path),
                "figure_path": str(figure_path) if args.make_plots else "",
            }
        )
    write_csv(output_dir / "representative_heads.csv", representative_rows)

    analysis_seconds = time.perf_counter() - analysis_started
    significant_heads = [row for row in head_rows if int(row["significant_positive_pairs"]) > 0]
    summary = {
        "args": vars(args),
        "resolved": {
            "layer_count": layer_count,
            "head_count": head_count,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "query_tokens": query_tokens,
            "observations": len(query_tokens),
            "context_tokens": args.context_tokens,
            "top2_budget": budget,
            "uniform_fixed_budget_conditional": (budget - 1) / (args.context_tokens - 1),
            "collection_seconds": collection_seconds,
            "analysis_seconds": analysis_seconds,
            "heads_analyzed": len(head_rows),
            "heads_with_significant_pairs": len(significant_heads),
            "heads_with_significant_pairs_fraction": len(significant_heads) / len(head_rows),
            "median_significant_pairs": float(
                np.median([int(row["significant_positive_pairs"]) for row in head_rows])
            ),
            "median_positive_excess_pair_mass_fraction": float(
                np.median([float(row["positive_excess_pair_mass_fraction"]) for row in head_rows])
            ),
            "median_distance_le_16_enrichment": float(
                np.median([float(row["distance_le_16_enrichment"]) for row in head_rows])
            ),
            "top_cluster_heads": sorted(
                (
                    {
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "cluster_score": float(row["cluster_score"]),
                        "significant_positive_pairs": int(row["significant_positive_pairs"]),
                        "significant_conditional_median": float(row["significant_conditional_median"]),
                        "largest_component_tokens": int(row["largest_component_tokens"]),
                    }
                    for row in head_rows
                ),
                key=lambda item: item["cluster_score"],
                reverse=True,
            )[:20],
            "metric_definitions": {
                "conditional_b_given_a": "P(token B is Top-2% selected | token A is Top-2% selected) across sampled queries",
                "uniform_fixed_budget_conditional": "(k-1)/(N-1), the conditional probability under uniform random fixed-k selection",
                "lift": "observed conditional divided by B's marginal selection probability",
                "phi": "binary association after correcting both token marginals",
                "significant_positive_pairs": "positive hypergeometric association after BH correction over N choose 2 pairs",
                "positive_excess_pair_mass_fraction": "fraction of observed co-selection event mass above marginal-independence expectation",
                "distance_enrichment": "near-position co-selection share divided by the same share under marginal independence",
            },
        },
        "paths": {
            "head_summary": str(output_dir / "head_coselection_summary.csv"),
            "layer_summary": str(output_dir / "layer_coselection_summary.csv"),
            "top_pairs": str(output_dir / "top_coselected_pairs.csv"),
            "top_tokens": str(output_dir / "top_selected_tokens.csv"),
            "selection_indices": str(output_dir / "selection_indices.npz") if args.save_selection_indices else None,
            "representatives": str(output_dir / "representative_heads.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["resolved"], indent=2), flush=True)


if __name__ == "__main__":
    main()
