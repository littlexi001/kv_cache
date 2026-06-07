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
DEFAULT_TEXT_PATH = (
    "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"
)

_TREE_CONFIG: TreeMaskConfig | None = None
_TREE_ENABLED = False
_MODULE_TO_LAYER: dict[int, int] = {}
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None


@dataclass(frozen=True)
class TreeLayout:
    leaf_size: int
    leaf_ranges: list[tuple[int, int]]
    mid_ranges: list[tuple[int, int]]
    mid_children: list[list[int]]
    big_ranges: list[tuple[int, int]]
    big_children: list[list[int]]


@dataclass(frozen=True)
class TreeMaskConfig:
    layers: set[int]
    kv_heads: set[int]
    boundary_fraction: float
    leaf_fraction: float
    leaf_size: int
    tree_fanout: int
    branch_counts: tuple[int, int, int]
    candidate_granularity: str


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and tree-retrieval-masked Qwen3 PPL.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/ppl_only")
    parser.add_argument("--prefill_tokens", type=int, default=5000)
    parser.add_argument("--eval_tokens", type=int, default=5000)
    parser.add_argument(
        "--eval_last_tokens_only",
        type=str2bool,
        default=False,
        help="Use the whole tokenized text as context and evaluate only the last eval_tokens tokens.",
    )
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--compute_tree_ppl", type=str2bool, default=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--kv_heads", default="all")
    parser.add_argument("--boundary_fraction", type=float, default=0.01)
    parser.add_argument("--leaf_fraction", type=float, default=0.001)
    parser.add_argument("--leaf_size", type=int, default=0)
    parser.add_argument("--tree_fanout", type=int, default=10)
    parser.add_argument("--tree_branch_counts", default="5,5,5")
    parser.add_argument("--candidate_granularity", choices=["attention_head", "kv_head_union"], default="attention_head")
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


def model_body(model: torch.nn.Module) -> torch.nn.Module:
    for attr_name in ("model", "transformer"):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
    return model


def register_attention_layers(model: torch.nn.Module) -> None:
    _MODULE_TO_LAYER.clear()
    body = model_body(model)
    layers = getattr(body, "layers", None) or getattr(body, "h", None)
    if layers is None:
        raise AttributeError("Could not find transformer layers on model.")
    for layer_idx, layer_obj in enumerate(layers):
        attn = getattr(layer_obj, "self_attn", None) or getattr(layer_obj, "attention", None)
        if attn is not None:
            _MODULE_TO_LAYER[id(attn)] = layer_idx


def ppl_fields() -> list[str]:
    return [
        "mode",
        "loss",
        "ppl",
        "token_count",
        "layers",
        "kv_heads",
        "boundary_fraction",
        "leaf_fraction",
        "leaf_size",
        "tree_fanout",
        "tree_branch_counts",
        "candidate_granularity",
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor | None]:
    past_key_values = None
    last_logits: torch.Tensor | None = None
    total_chunks = math.ceil(prefill_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(0, prefill_tokens, chunk_size), start=1):
        end = min(start + chunk_size, prefill_tokens)
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
        print(f"prefill chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :].detach()
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return past_key_values, last_logits


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
    return TreeLayout(leaf_size, leaf_ranges, mid_ranges, mid_children, big_ranges, big_children)


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


def tree_candidate_indices(
    token: int,
    query: torch.Tensor,
    prefix_sum: torch.Tensor,
    layout: TreeLayout,
    config: TreeMaskConfig,
) -> set[int]:
    visible = token + 1
    prefix, recent, middle_range = visible_regions(token, config.boundary_fraction)
    if middle_range[1] <= middle_range[0]:
        return {idx for idx in prefix | recent if idx <= token}

    big_indices = top_scored_indices(
        query,
        prefix_sum,
        layout.big_ranges,
        list(range(len(layout.big_ranges))),
        middle_range,
        visible,
        config.branch_counts[0],
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
                config.branch_counts[1],
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
                config.branch_counts[2],
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
    return {idx for idx in prefix | recent | tree_tokens if 0 <= idx <= token}


def tree_keep_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    layer: int,
    scores: torch.Tensor,
    config: TreeMaskConfig,
) -> torch.Tensor:
    batch, attention_heads, query_count, _ = query_states.shape
    _, kv_heads, key_count, _ = key_states.shape
    if batch != 1:
        raise ValueError("Tree PPL currently supports batch size 1.")
    if layer not in config.layers:
        return torch.isfinite(scores)

    group_size = attention_heads // kv_heads
    active_kv_heads = {kv_head for kv_head in range(kv_heads) if kv_head in config.kv_heads}
    if not active_kv_heads:
        return torch.isfinite(scores)

    leaf_size = config.leaf_size if config.leaf_size > 0 else max(1, math.ceil(config.leaf_fraction * key_count))
    layout = build_tree_layout(key_count, leaf_size, config.tree_fanout)
    prefix_sums = []
    for kv_head in range(kv_heads):
        vectors = key_states[0, kv_head].detach().float()
        zeros = torch.zeros((1, vectors.shape[-1]), dtype=torch.float32, device=vectors.device)
        prefix_sums.append(torch.cat([zeros, torch.cumsum(vectors, dim=0)], dim=0))

    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep |= ~torch.isfinite(scores)
    keep.logical_not_()
    keep.fill_(False)

    for local_query in range(query_count):
        token = key_count - query_count + local_query
        if token < 0:
            continue
        per_head_candidates: dict[int, set[int]] = {}
        for attention_head in range(attention_heads):
            kv_head = attention_head // group_size
            if kv_head not in active_kv_heads:
                keep[:, attention_head, local_query, : token + 1] = True
                continue
            query = query_states[0, attention_head, local_query].detach().float()
            per_head_candidates[attention_head] = tree_candidate_indices(
                token, query, prefix_sums[kv_head], layout, config
            )
        if config.candidate_granularity == "kv_head_union":
            for kv_head in active_kv_heads:
                union_candidates: set[int] = set()
                heads = range(kv_head * group_size, min((kv_head + 1) * group_size, attention_heads))
                for attention_head in heads:
                    union_candidates.update(per_head_candidates.get(attention_head, set()))
                for attention_head in heads:
                    if union_candidates:
                        idx = torch.tensor(sorted(union_candidates), dtype=torch.long, device=scores.device)
                        keep[:, attention_head, local_query, idx] = True
        else:
            for attention_head, candidates in per_head_candidates.items():
                if candidates:
                    idx = torch.tensor(sorted(candidates), dtype=torch.long, device=scores.device)
                    keep[:, attention_head, local_query, idx] = True
    return keep


def _tree_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = _TREE_CONFIG
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    original_key_states = key_states
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states_for_attention = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    else:
        key_states_for_attention = key_states
    scores = torch.matmul(query_states, key_states_for_attention.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    if _TREE_ENABLED and config is not None:
        layer = _MODULE_TO_LAYER.get(id(module), -1)
        keep = tree_keep_mask(query_states, original_key_states, layer, scores, config)
        scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def install_qwen3_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is not None:
        return
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3 for tree PPL.") from exc
    _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
    setattr(modeling_qwen3, "eager_attention_forward", _tree_eager_attention_forward)
    if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
        modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _tree_eager_attention_forward


@contextmanager
def tree_mask_enabled(enabled: bool):
    global _TREE_ENABLED
    previous = _TREE_ENABLED
    _TREE_ENABLED = enabled
    try:
        yield
    finally:
        _TREE_ENABLED = previous


@torch.inference_mode()
def compute_eval_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    use_tree_mask: bool,
) -> tuple[float, float, int]:
    with tree_mask_enabled(use_tree_mask):
        past_key_values, prev_logits = prefill_cache(model, input_ids, prefill_tokens, chunk_size, input_device)
    if prev_logits is None:
        raise RuntimeError("Prefill did not return last logits.")
    total_loss = 0.0
    total_count = 0
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, chunk_size), start=1):
        end = min(start + chunk_size, eval_end)
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
        label = "tree" if use_tree_mask else "baseline"
        print(f"ppl {label} chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        with tree_mask_enabled(use_tree_mask):
            outputs = model_forward(model, kwargs)
        logits = outputs.logits
        shifted_logits = torch.cat([prev_logits.unsqueeze(1), logits[:, :-1, :]], dim=1)
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
            chunk.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss)
        total_count += int(chunk.numel())
        prev_logits = logits[:, -1, :].detach()
        past_key_values = outputs.past_key_values
        del outputs, chunk, logits, shifted_logits, loss
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    mean_loss = total_loss / max(1, total_count)
    return mean_loss, math.exp(mean_loss), total_count


def tree_metadata_row(args: argparse.Namespace, layer_indices: list[int], kv_head_indices: list[int]) -> dict[str, Any]:
    return {
        "layers": ",".join(str(index) for index in layer_indices),
        "kv_heads": ",".join(str(index) for index in kv_head_indices),
        "boundary_fraction": args.boundary_fraction,
        "leaf_fraction": args.leaf_fraction,
        "leaf_size": args.leaf_size,
        "tree_fanout": args.tree_fanout,
        "tree_branch_counts": args.tree_branch_counts,
        "candidate_granularity": args.candidate_granularity,
    }


def main() -> None:
    global _TREE_CONFIG
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

    if args.eval_last_tokens_only:
        if len(token_ids) <= args.eval_tokens:
            raise ValueError(
                f"Tokenization produced {len(token_ids)} tokens, "
                f"not enough to evaluate the last {args.eval_tokens} tokens with a non-empty prefill."
            )
        prefill_tokens = len(token_ids) - args.eval_tokens
    else:
        prefill_tokens = args.prefill_tokens
        total_tokens_needed = prefill_tokens + args.eval_tokens
        if args.require_total_tokens and len(token_ids) < total_tokens_needed:
            raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than {total_tokens_needed}.")
        token_ids = token_ids[:total_tokens_needed]
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
    kv_head_count = int(getattr(model.config, "num_key_value_heads"))
    layer_indices = parse_index_spec(args.layers, layer_count, "layers")
    kv_head_indices = parse_index_spec(args.kv_heads, kv_head_count, "kv_heads")
    _TREE_CONFIG = TreeMaskConfig(
        layers=set(layer_indices),
        kv_heads=set(kv_head_indices),
        boundary_fraction=args.boundary_fraction,
        leaf_fraction=args.leaf_fraction,
        leaf_size=args.leaf_size,
        tree_fanout=args.tree_fanout,
        branch_counts=branch_counts,
        candidate_granularity=args.candidate_granularity,
    )
    register_attention_layers(model)
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, requested_device)

    metadata = tree_metadata_row(args, layer_indices, kv_head_indices)
    rows: list[dict[str, Any]] = []
    baseline_loss, baseline_ppl, baseline_count = compute_eval_loss(
        model, input_ids, prefill_tokens, args.eval_tokens, args.chunk_size, input_device, False
    )
    rows.append({"mode": "baseline", "loss": baseline_loss, "ppl": baseline_ppl, "token_count": baseline_count, **metadata})
    if args.compute_tree_ppl:
        tree_loss, tree_ppl, tree_count = compute_eval_loss(
            model, input_ids, prefill_tokens, args.eval_tokens, args.chunk_size, input_device, True
        )
        rows.append({"mode": "tree", "loss": tree_loss, "ppl": tree_ppl, "token_count": tree_count, **metadata})

    ppl_path = output_dir / "ppl_by_tree.csv"
    write_csv(ppl_path, rows, ppl_fields())
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "prefill_tokens": prefill_tokens,
                    "eval_tokens": args.eval_tokens,
                    "eval_last_tokens_only": args.eval_last_tokens_only,
                    "total_tokenized_tokens_used": int(input_ids.numel()),
                    "layers": layer_indices,
                    "kv_heads": kv_head_indices,
                    "tree_branch_counts": branch_counts,
                },
                "paths": {"ppl_by_tree": str(ppl_path)},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote tree PPL outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
