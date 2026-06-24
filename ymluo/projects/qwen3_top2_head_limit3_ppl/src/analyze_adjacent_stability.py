from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
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


DEFAULT_MODEL_PATH = "ymluo/models/Qwen3-0.6B"
DEFAULT_TEXT_PATH = "external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt"

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "AdjacentStabilityCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure adjacent decode-step stability for q top-|abs| channels and true full-QK top2 tokens."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/adjacent_stability")
    parser.add_argument("--total_tokens", type=int, default=2048)
    parser.add_argument("--prefill_tokens", type=int, default=1536)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--dim_counts", default="8,16,32,64")
    parser.add_argument(
        "--max_pair_samples",
        type=int,
        default=128,
        help="Maximum adjacent eval pairs to analyze. Use <=0 for all eval-adjacent pairs.",
    )
    parser.add_argument("--pair_stride", type=int, default=0)
    parser.add_argument("--write_layer_head", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]


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


@dataclass
class StabilityAccumulator:
    cases: int = 0
    intersection_sum: float = 0.0
    current_size_sum: float = 0.0
    previous_size_sum: float = 0.0
    union_size_sum: float = 0.0
    current_mass_sum: float = 0.0
    previous_set_current_mass_sum: float = 0.0
    intersection_current_mass_sum: float = 0.0

    def add(
        self,
        intersection: int,
        current_size: int,
        previous_size: int,
        union_size: int,
        current_mass: float = 0.0,
        previous_set_current_mass: float = 0.0,
        intersection_current_mass: float = 0.0,
    ) -> None:
        self.cases += 1
        self.intersection_sum += intersection
        self.current_size_sum += current_size
        self.previous_size_sum += previous_size
        self.union_size_sum += union_size
        self.current_mass_sum += current_mass
        self.previous_set_current_mass_sum += previous_set_current_mass
        self.intersection_current_mass_sum += intersection_current_mass

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            **extra,
            "cases": self.cases,
            "mean_intersection": self.intersection_sum / self.cases if self.cases else 0.0,
            "mean_current_size": self.current_size_sum / self.cases if self.cases else 0.0,
            "mean_previous_size": self.previous_size_sum / self.cases if self.cases else 0.0,
            "overlap_fraction_current": self.intersection_sum / self.current_size_sum
            if self.current_size_sum
            else 0.0,
            "overlap_fraction_previous": self.intersection_sum / self.previous_size_sum
            if self.previous_size_sum
            else 0.0,
            "jaccard": self.intersection_sum / self.union_size_sum if self.union_size_sum else 0.0,
            "current_attention_mass_mean": self.current_mass_sum / self.cases if self.cases else 0.0,
            "previous_set_current_attention_mass_mean": self.previous_set_current_mass_sum / self.cases
            if self.cases
            else 0.0,
            "intersection_current_attention_mass_mean": self.intersection_current_mass_sum / self.cases
            if self.cases
            else 0.0,
        }


class AdjacentStabilityCollector:
    def __init__(
        self,
        layer_count: int,
        head_count: int,
        dim_counts: list[int],
        top_fraction: float,
        target_pair_ends: set[int],
        write_layer_head: bool,
    ) -> None:
        self.layer_count = layer_count
        self.head_count = head_count
        self.dim_counts = dim_counts
        self.top_fraction = top_fraction
        self.target_pair_ends = target_pair_ends
        self.write_layer_head = write_layer_head
        self.previous_by_layer_head: dict[tuple[int, int], dict[str, Any]] = {}
        self.q_by_dim: dict[int, StabilityAccumulator] = defaultdict(StabilityAccumulator)
        self.q_by_layer_dim: dict[tuple[int, int], StabilityAccumulator] = defaultdict(StabilityAccumulator)
        self.q_by_layer_head_dim: dict[tuple[int, int, int], StabilityAccumulator] = defaultdict(StabilityAccumulator)
        self.top2_overall = StabilityAccumulator()
        self.top2_by_layer: dict[int, StabilityAccumulator] = defaultdict(StabilityAccumulator)
        self.top2_by_layer_head: dict[tuple[int, int], StabilityAccumulator] = defaultdict(StabilityAccumulator)
        self.observed_pair_ends: set[int] = set()

    def observe(
        self,
        layer: int,
        query_token: int,
        query_states: torch.Tensor,
        scores: torch.Tensor,
        query_index: int,
    ) -> None:
        finite = torch.isfinite(scores[:, :, query_index, :])
        valid_count = int(finite[0, 0].sum().item())
        if valid_count <= 1:
            return
        history_count = valid_count - 1
        true_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
        row_scores = scores[0, :, query_index, :history_count].detach().float()
        attention_weights = F.softmax(scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        q = query_states[0, :, query_index, :].detach().float()
        true_top_indices = torch.topk(row_scores, k=true_count, dim=-1, largest=True).indices

        should_compare = query_token in self.target_pair_ends
        if should_compare:
            self.observed_pair_ends.add(query_token)

        for head in range(self.head_count):
            current_top_dims: dict[int, set[int]] = {}
            for dim in self.dim_counts:
                d = min(dim, q.shape[-1])
                current_top_dims[dim] = set(torch.topk(q[head].abs(), k=d, largest=True).indices.cpu().tolist())
            current_top2 = set(true_top_indices[head].cpu().tolist())

            previous = self.previous_by_layer_head.get((layer, head))
            if should_compare and previous is not None and previous["query_token"] == query_token - 1:
                for dim in self.dim_counts:
                    prev_dims = previous["top_dims"][dim]
                    curr_dims = current_top_dims[dim]
                    intersection = len(curr_dims & prev_dims)
                    union = len(curr_dims | prev_dims)
                    self.q_by_dim[dim].add(intersection, len(curr_dims), len(prev_dims), union)
                    self.q_by_layer_dim[(layer, dim)].add(intersection, len(curr_dims), len(prev_dims), union)
                    if self.write_layer_head:
                        self.q_by_layer_head_dim[(layer, head, dim)].add(
                            intersection,
                            len(curr_dims),
                            len(prev_dims),
                            union,
                        )

                prev_top2 = previous["top2"]
                intersection_tokens = current_top2 & prev_top2
                union_tokens = current_top2 | prev_top2
                current_mass = float(attention_weights[head, list(current_top2)].sum().item()) if current_top2 else 0.0
                prev_valid = [token for token in prev_top2 if token < history_count]
                prev_mass = float(attention_weights[head, prev_valid].sum().item()) if prev_valid else 0.0
                inter_valid = [token for token in intersection_tokens if token < history_count]
                inter_mass = float(attention_weights[head, inter_valid].sum().item()) if inter_valid else 0.0
                self.top2_overall.add(
                    len(intersection_tokens),
                    len(current_top2),
                    len(prev_top2),
                    len(union_tokens),
                    current_mass,
                    prev_mass,
                    inter_mass,
                )
                self.top2_by_layer[layer].add(
                    len(intersection_tokens),
                    len(current_top2),
                    len(prev_top2),
                    len(union_tokens),
                    current_mass,
                    prev_mass,
                    inter_mass,
                )
                if self.write_layer_head:
                    self.top2_by_layer_head[(layer, head)].add(
                        len(intersection_tokens),
                        len(current_top2),
                        len(prev_top2),
                        len(union_tokens),
                        current_mass,
                        prev_mass,
                        inter_mass,
                    )

            self.previous_by_layer_head[(layer, head)] = {
                "query_token": query_token,
                "top_dims": current_top_dims,
                "top2": current_top2,
            }

    def q_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        by_dim = [self.q_by_dim[dim].row({"dim_count": dim}) for dim in self.dim_counts]
        by_layer = [
            acc.row({"layer": layer, "dim_count": dim})
            for (layer, dim), acc in sorted(self.q_by_layer_dim.items())
        ]
        by_layer_head = [
            acc.row({"layer": layer, "head": head, "dim_count": dim})
            for (layer, head, dim), acc in sorted(self.q_by_layer_head_dim.items())
        ]
        return by_dim, by_layer, by_layer_head

    def top2_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        overall = [self.top2_overall.row({"scope": "overall"})]
        by_layer = [acc.row({"scope": "layer", "layer": layer}) for layer, acc in sorted(self.top2_by_layer.items())]
        by_layer_head = [
            acc.row({"scope": "layer_head", "layer": layer, "head": head})
            for (layer, head), acc in sorted(self.top2_by_layer_head.items())
        ]
        return overall, by_layer, by_layer_head


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
        chunk_query_start = key_count - query_count
        for query_index in range(query_count):
            query_token = chunk_query_start + query_index
            _ACTIVE_COLLECTOR.observe(layer, query_token, query_states, scores, query_index)

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
def active_collector(collector: AdjacentStabilityCollector):
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
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> Any:
    past_key_values = None
    total_chunks = math.ceil(prefill_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(0, prefill_tokens, chunk_size), start=1):
        end = min(start + chunk_size, prefill_tokens)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(input_device),
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
        del outputs
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return past_key_values


def build_pair_ends(prefill_tokens: int, eval_tokens: int, pair_stride: int, max_pair_samples: int) -> list[int]:
    # Pair end token t compares eval query t with immediately previous query t-1.
    pair_ends = list(range(prefill_tokens + 1, prefill_tokens + eval_tokens))
    if pair_stride > 0:
        pair_ends = pair_ends[::pair_stride]
    if max_pair_samples > 0 and len(pair_ends) > max_pair_samples:
        if max_pair_samples == 1:
            return [pair_ends[len(pair_ends) // 2]]
        step = (len(pair_ends) - 1) / (max_pair_samples - 1)
        indices = sorted({round(i * step) for i in range(max_pair_samples)})
        pair_ends = [pair_ends[index] for index in indices]
    return pair_ends


@torch.inference_mode()
def run_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: AdjacentStabilityCollector,
) -> None:
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    with active_collector(collector):
        for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, chunk_size), start=1):
            end = min(start + chunk_size, eval_end)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids[:, start:end].to(input_device),
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "cache_position": torch.arange(start, end, device=input_device),
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            print(f"eval chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
            outputs = model_forward(model, kwargs)
            past_key_values = outputs.past_key_values
            del outputs
            if input_device.type == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    dim_counts = [int(part) for part in args.dim_counts.split(",") if part.strip()]
    if not dim_counts or any(dim <= 0 for dim in dim_counts):
        raise ValueError("--dim_counts must contain positive integers.")
    if args.prefill_tokens + args.eval_tokens > args.total_tokens:
        raise ValueError("--prefill_tokens + --eval_tokens must be <= --total_tokens.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    if args.require_total_tokens and len(token_ids) < args.total_tokens:
        raise ValueError(f"Need {args.total_tokens} tokens, got {len(token_ids)}.")
    input_ids = torch.tensor(token_ids[: args.total_tokens], dtype=torch.long).view(1, -1)

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

    pair_ends = build_pair_ends(args.prefill_tokens, args.eval_tokens, args.pair_stride, args.max_pair_samples)
    collector = AdjacentStabilityCollector(
        layer_count=layer_count,
        head_count=head_count,
        dim_counts=dim_counts,
        top_fraction=args.top_fraction,
        target_pair_ends=set(pair_ends),
        write_layer_head=args.write_layer_head,
    )

    started = time.perf_counter()
    past = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    run_eval(model, input_ids, past, args.prefill_tokens, args.eval_tokens, args.chunk_size, input_device, collector)
    seconds = time.perf_counter() - started

    q_by_dim, q_by_layer, q_by_layer_head = collector.q_rows()
    top2_overall, top2_by_layer, top2_by_layer_head = collector.top2_rows()

    q_fields = [
        "dim_count",
        "cases",
        "mean_intersection",
        "mean_current_size",
        "mean_previous_size",
        "overlap_fraction_current",
        "overlap_fraction_previous",
        "jaccard",
        "current_attention_mass_mean",
        "previous_set_current_attention_mass_mean",
        "intersection_current_attention_mass_mean",
    ]
    top2_fields = [
        "scope",
        "cases",
        "mean_intersection",
        "mean_current_size",
        "mean_previous_size",
        "overlap_fraction_current",
        "overlap_fraction_previous",
        "jaccard",
        "current_attention_mass_mean",
        "previous_set_current_attention_mass_mean",
        "intersection_current_attention_mass_mean",
    ]
    write_csv(output_dir / "q_channel_adjacent_by_dim.csv", q_by_dim, q_fields)
    write_csv(output_dir / "q_channel_adjacent_by_layer.csv", q_by_layer, ["layer"] + q_fields)
    if args.write_layer_head:
        write_csv(output_dir / "q_channel_adjacent_by_layer_head.csv", q_by_layer_head, ["layer", "head"] + q_fields)
    write_csv(output_dir / "top2_token_adjacent_overall.csv", top2_overall, top2_fields)
    write_csv(output_dir / "top2_token_adjacent_by_layer.csv", top2_by_layer, ["layer"] + top2_fields)
    if args.write_layer_head:
        write_csv(output_dir / "top2_token_adjacent_by_layer_head.csv", top2_by_layer_head, ["layer", "head"] + top2_fields)

    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens_loaded": int(input_ids.numel()),
            "layer_count": layer_count,
            "head_count": head_count,
            "sampled_pair_ends_requested": pair_ends,
            "sampled_pair_ends_observed": sorted(collector.observed_pair_ends),
            "seconds": seconds,
            "metric_definitions": {
                "q_channel_overlap_fraction_current": "intersection(topD q channels at t, topD q channels at t-1) / D",
                "top2_overlap_fraction_current": "intersection(true top2 tokens at t, true top2 tokens at t-1) / current true top2 size",
                "previous_set_current_attention_mass_mean": "current query's full attention mass on the previous query's true top2 token set",
            },
        },
        "paths": {
            "q_channel_adjacent_by_dim": str(output_dir / "q_channel_adjacent_by_dim.csv"),
            "top2_token_adjacent_overall": str(output_dir / "top2_token_adjacent_overall.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": seconds, "q_by_dim": q_by_dim, "top2": top2_overall}, indent=2))


if __name__ == "__main__":
    main()
