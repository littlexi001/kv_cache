from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _install_torchvision_fake_registration_guard() -> None:
    register_fake = getattr(torch.library, "register_fake", None)
    if register_fake is None or getattr(register_fake, "_top2_count_guarded", False):
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

    guarded_register_fake._top2_count_guarded = True
    torch.library.register_fake = guarded_register_fake


_install_torchvision_fake_registration_guard()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "ymluo/models/Qwen3-0.6B"
DEFAULT_TEXT_PATH = "external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt"

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "Top2TokenSelectionCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count how often each historical token is selected by true full-QK top-fraction "
            "attention for every Qwen3 layer/head during eval forward."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/top2_token_selection_counts")
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
    parser.add_argument(
        "--remote_only",
        type=str2bool,
        default=False,
        help="If true, count only selected tokens outside sink and recent windows.",
    )
    parser.add_argument("--exclude_sink_tokens", type=int, default=0)
    parser.add_argument("--exclude_recent_tokens", type=int, default=0)
    parser.add_argument("--max_query_samples", type=int, default=0, help="Use <=0 to analyze all eval queries.")
    parser.add_argument("--query_stride", type=int, default=0)
    parser.add_argument("--include_token_text", type=str2bool, default=True)
    parser.add_argument("--write_zero_count_tokens", type=str2bool, default=True)
    parser.add_argument("--write_zero_count_layer_rows", type=str2bool, default=False)
    parser.add_argument("--write_layer_head_token_counts", type=str2bool, default=True)
    parser.add_argument(
        "--layer_head_min_count",
        type=int,
        default=1,
        help="Minimum count for rows in token_selection_counts_by_layer_head.csv.",
    )
    parser.add_argument(
        "--top_tokens_per_layer_head",
        type=int,
        default=100,
        help="Write this many highest-count tokens per layer/head to top_tokens_by_layer_head.csv.",
    )
    parser.add_argument(
        "--accumulate_selected_attention_mass",
        type=str2bool,
        default=False,
        help="Also accumulate full-softmax attention mass for selected token events.",
    )
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


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_token_text(tokenizer: Any, token_id: int) -> tuple[str, str]:
    piece = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    return str(piece).replace("\n", "\\n").replace("\r", "\\r"), text.replace("\n", "\\n").replace("\r", "\\r")


def build_query_samples(prefill_tokens: int, eval_tokens: int, query_stride: int, max_query_samples: int) -> list[int]:
    queries = list(range(prefill_tokens, prefill_tokens + eval_tokens))
    if query_stride > 0:
        queries = queries[::query_stride]
    if max_query_samples > 0 and len(queries) > max_query_samples:
        if max_query_samples == 1:
            return [queries[len(queries) // 2]]
        step = (len(queries) - 1) / (max_query_samples - 1)
        indices = sorted({round(i * step) for i in range(max_query_samples)})
        queries = [queries[index] for index in indices]
    return queries


class Top2TokenSelectionCollector:
    def __init__(
        self,
        layer_count: int,
        head_count: int,
        total_tokens: int,
        query_tokens: set[int],
        top_fraction: float,
        remote_only: bool,
        exclude_sink_tokens: int,
        exclude_recent_tokens: int,
        accumulate_attention_mass: bool,
    ) -> None:
        self.layer_count = layer_count
        self.head_count = head_count
        self.total_tokens = total_tokens
        self.query_tokens = query_tokens
        self.top_fraction = top_fraction
        self.remote_only = remote_only
        self.exclude_sink_tokens = exclude_sink_tokens
        self.exclude_recent_tokens = exclude_recent_tokens
        self.accumulate_attention_mass = accumulate_attention_mass

        self.total_counts = torch.zeros(total_tokens, dtype=torch.long)
        self.layer_counts = torch.zeros((layer_count, total_tokens), dtype=torch.long)
        self.layer_head_counts = torch.zeros((layer_count, head_count, total_tokens), dtype=torch.long)
        self.total_attention_mass = torch.zeros(total_tokens, dtype=torch.float64) if accumulate_attention_mass else None
        self.layer_attention_mass = (
            torch.zeros((layer_count, total_tokens), dtype=torch.float64) if accumulate_attention_mass else None
        )
        self.layer_head_attention_mass = (
            torch.zeros((layer_count, head_count, total_tokens), dtype=torch.float64)
            if accumulate_attention_mass
            else None
        )

        self.observed_query_tokens: set[int] = set()
        self.topk_by_query: dict[int, int] = {}
        self.total_selection_events = 0
        self.layer_selection_events = torch.zeros(layer_count, dtype=torch.long)
        self.layer_head_selection_events = torch.zeros((layer_count, head_count), dtype=torch.long)

    def observe(self, layer: int, query_token: int, scores: torch.Tensor, query_index: int) -> None:
        if query_token not in self.query_tokens:
            return
        finite = torch.isfinite(scores[:, :, query_index, :])
        valid_count = min(int(finite[0, 0].sum().item()), query_token + 1)
        if valid_count <= 1:
            return
        history_count = valid_count - 1
        top_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
        row_scores = scores[0, :, query_index, :history_count].detach().float()
        top_indices = torch.topk(row_scores, k=top_count, dim=-1, largest=True).indices
        if self.remote_only:
            remote_end = max(0, history_count - self.exclude_recent_tokens)
            selected_mask = (top_indices >= self.exclude_sink_tokens) & (top_indices < remote_end)
        else:
            selected_mask = torch.ones_like(top_indices, dtype=torch.bool)
        selected = top_indices.detach().cpu().long()
        selected_mask_cpu = selected_mask.detach().cpu()
        flat_selected = selected.reshape(-1)
        flat_selected_mask = selected_mask_cpu.reshape(-1)
        flat_selected = flat_selected[flat_selected_mask]

        self.observed_query_tokens.add(query_token)
        self.topk_by_query[query_token] = top_count
        self.total_selection_events += int(flat_selected.numel())
        self.layer_selection_events[layer] += int(flat_selected.numel())
        head_event_counts = selected_mask_cpu.sum(dim=-1).long()
        self.layer_head_selection_events[layer, :] += head_event_counts

        if flat_selected.numel() > 0:
            total_increment = torch.bincount(flat_selected, minlength=self.total_tokens)
            self.total_counts += total_increment
            self.layer_counts[layer] += total_increment
        for head in range(self.head_count):
            selected_head = selected[head, selected_mask_cpu[head]]
            if selected_head.numel() > 0:
                self.layer_head_counts[layer, head].index_add_(
                    0,
                    selected_head,
                    torch.ones(selected_head.numel(), dtype=torch.long),
                )

        if self.accumulate_attention_mass:
            attention_weights = F.softmax(scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[
                :, :history_count
            ]
            selected_mass = torch.gather(attention_weights, dim=-1, index=top_indices).detach().cpu().double()
            flat_mass = selected_mass.reshape(-1)[flat_selected_mask]
            assert self.total_attention_mass is not None
            assert self.layer_attention_mass is not None
            assert self.layer_head_attention_mass is not None
            if flat_selected.numel() > 0:
                self.total_attention_mass.index_add_(0, flat_selected, flat_mass)
                self.layer_attention_mass[layer].index_add_(0, flat_selected, flat_mass)
            for head in range(self.head_count):
                selected_head = selected[head, selected_mask_cpu[head]]
                selected_mass_head = selected_mass[head, selected_mask_cpu[head]]
                if selected_head.numel() > 0:
                    self.layer_head_attention_mass[layer, head].index_add_(0, selected_head, selected_mass_head)


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
            _ACTIVE_COLLECTOR.observe(layer, query_token, scores, query_index)

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
def active_collector(collector: Top2TokenSelectionCollector):
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


@torch.inference_mode()
def run_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: Top2TokenSelectionCollector,
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


def eligible_query_count(
    observed_queries: list[int],
    token_index: int,
    remote_only: bool = False,
    exclude_sink_tokens: int = 0,
    exclude_recent_tokens: int = 0,
) -> int:
    if remote_only and token_index < exclude_sink_tokens:
        return 0
    if remote_only:
        return len(observed_queries) - bisect.bisect_right(observed_queries, token_index + exclude_recent_tokens)
    return len(observed_queries) - bisect.bisect_right(observed_queries, token_index)


def token_row_prefix(
    tokenizer: Any,
    token_ids: list[int],
    token_index: int,
    include_token_text: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "token_index": token_index,
        "token_id": int(token_ids[token_index]),
    }
    if include_token_text:
        token_piece, token_text = safe_token_text(tokenizer, token_ids[token_index])
        row["token_piece"] = token_piece
        row["token_text"] = token_text
    return row


def make_summary_row(
    counts: torch.Tensor,
    scope: dict[str, Any],
    total_events: int,
    tokenizer: Any,
    token_ids: list[int],
    include_token_text: bool,
) -> dict[str, Any]:
    nonzero = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if nonzero.numel() == 0:
        row: dict[str, Any] = {
            **scope,
            "selection_events": total_events,
            "unique_selected_tokens": 0,
            "top_token_index": "",
            "top_token_id": "",
            "top_token_count": 0,
            "top_token_event_fraction": 0.0,
        }
        if include_token_text:
            row["top_token_piece"] = ""
            row["top_token_text"] = ""
        return row
    top_count, top_position = torch.max(counts, dim=0)
    top_token = int(top_position.item())
    row = {
        **scope,
        "selection_events": total_events,
        "unique_selected_tokens": int(nonzero.numel()),
        "top_token_index": top_token,
        "top_token_id": int(token_ids[top_token]),
        "top_token_count": int(top_count.item()),
        "top_token_event_fraction": int(top_count.item()) / total_events if total_events else 0.0,
    }
    if include_token_text:
        token_piece, token_text = safe_token_text(tokenizer, token_ids[top_token])
        row["top_token_piece"] = token_piece
        row["top_token_text"] = token_text
    return row


def main() -> None:
    args = parse_args()
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top_fraction must be in (0, 1].")
    if args.exclude_sink_tokens < 0 or args.exclude_recent_tokens < 0:
        raise ValueError("--exclude_sink_tokens and --exclude_recent_tokens must be non-negative.")
    if args.prefill_tokens + args.eval_tokens > args.total_tokens:
        raise ValueError("--prefill_tokens + --eval_tokens must be <= --total_tokens.")
    if args.layer_head_min_count < 0:
        raise ValueError("--layer_head_min_count must be non-negative.")

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
    token_ids = token_ids[: args.total_tokens]
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

    query_samples = build_query_samples(args.prefill_tokens, args.eval_tokens, args.query_stride, args.max_query_samples)
    collector = Top2TokenSelectionCollector(
        layer_count=layer_count,
        head_count=head_count,
        total_tokens=args.total_tokens,
        query_tokens=set(query_samples),
        top_fraction=args.top_fraction,
        remote_only=args.remote_only,
        exclude_sink_tokens=args.exclude_sink_tokens,
        exclude_recent_tokens=args.exclude_recent_tokens,
        accumulate_attention_mass=args.accumulate_selected_attention_mass,
    )

    started = time.perf_counter()
    past = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    run_eval(model, input_ids, past, args.prefill_tokens, args.eval_tokens, args.chunk_size, input_device, collector)
    seconds = time.perf_counter() - started

    observed_queries = sorted(collector.observed_query_tokens)
    max_observed_query = max(observed_queries) if observed_queries else args.prefill_tokens
    token_indices = range(0, min(args.total_tokens, max_observed_query))

    token_fields = ["token_index", "token_id"]
    if args.include_token_text:
        token_fields += ["token_piece", "token_text"]
    token_fields += [
        "eligible_queries",
        "selected_count",
        "max_possible_layer_head_query_count",
        "selection_rate",
    ]
    if args.accumulate_selected_attention_mass:
        token_fields += ["selected_attention_mass_sum"]

    token_rows: list[dict[str, Any]] = []
    for token_index in token_indices:
        count = int(collector.total_counts[token_index].item())
        eligible = eligible_query_count(
            observed_queries,
            token_index,
            args.remote_only,
            args.exclude_sink_tokens,
            args.exclude_recent_tokens,
        )
        if count == 0 and not args.write_zero_count_tokens:
            continue
        possible = eligible * layer_count * head_count
        row = token_row_prefix(tokenizer, token_ids, token_index, args.include_token_text)
        row.update(
            {
                "eligible_queries": eligible,
                "selected_count": count,
                "max_possible_layer_head_query_count": possible,
                "selection_rate": count / possible if possible else 0.0,
            }
        )
        if args.accumulate_selected_attention_mass:
            assert collector.total_attention_mass is not None
            row["selected_attention_mass_sum"] = float(collector.total_attention_mass[token_index].item())
        token_rows.append(row)
    write_csv(output_dir / "token_selection_counts.csv", token_rows, token_fields)

    layer_fields = ["layer"] + token_fields
    layer_rows: list[dict[str, Any]] = []
    for layer in range(layer_count):
        for token_index in token_indices:
            count = int(collector.layer_counts[layer, token_index].item())
            if count == 0 and not args.write_zero_count_layer_rows:
                continue
            eligible = eligible_query_count(
                observed_queries,
                token_index,
                args.remote_only,
                args.exclude_sink_tokens,
                args.exclude_recent_tokens,
            )
            possible = eligible * head_count
            row = token_row_prefix(tokenizer, token_ids, token_index, args.include_token_text)
            row.update(
                {
                    "layer": layer,
                    "eligible_queries": eligible,
                    "selected_count": count,
                    "max_possible_layer_head_query_count": possible,
                    "selection_rate": count / possible if possible else 0.0,
                }
            )
            if args.accumulate_selected_attention_mass:
                assert collector.layer_attention_mass is not None
                row["selected_attention_mass_sum"] = float(collector.layer_attention_mass[layer, token_index].item())
            layer_rows.append(row)
    write_csv(output_dir / "token_selection_counts_by_layer.csv", layer_rows, layer_fields)

    layer_head_fields = ["layer", "head"] + token_fields
    if args.write_layer_head_token_counts:
        layer_head_rows: list[dict[str, Any]] = []
        for layer in range(layer_count):
            for head in range(head_count):
                counts = collector.layer_head_counts[layer, head]
                selected_tokens = torch.nonzero(counts >= args.layer_head_min_count, as_tuple=False).flatten()
                for token_tensor in selected_tokens:
                    token_index = int(token_tensor.item())
                    if token_index >= max_observed_query:
                        continue
                    count = int(counts[token_index].item())
                    eligible = eligible_query_count(
                        observed_queries,
                        token_index,
                        args.remote_only,
                        args.exclude_sink_tokens,
                        args.exclude_recent_tokens,
                    )
                    row = token_row_prefix(tokenizer, token_ids, token_index, args.include_token_text)
                    row.update(
                        {
                            "layer": layer,
                            "head": head,
                            "eligible_queries": eligible,
                            "selected_count": count,
                            "max_possible_layer_head_query_count": eligible,
                            "selection_rate": count / eligible if eligible else 0.0,
                        }
                    )
                    if args.accumulate_selected_attention_mass:
                        assert collector.layer_head_attention_mass is not None
                        row["selected_attention_mass_sum"] = float(
                            collector.layer_head_attention_mass[layer, head, token_index].item()
                        )
                    layer_head_rows.append(row)
        write_csv(output_dir / "token_selection_counts_by_layer_head.csv", layer_head_rows, layer_head_fields)

    top_fields = ["layer", "head", "rank"] + token_fields
    top_rows: list[dict[str, Any]] = []
    top_n = max(0, args.top_tokens_per_layer_head)
    if top_n > 0:
        for layer in range(layer_count):
            for head in range(head_count):
                counts = collector.layer_head_counts[layer, head]
                k = min(top_n, int((counts > 0).sum().item()))
                if k <= 0:
                    continue
                values, indices = torch.topk(counts, k=k, largest=True)
                for rank, (value_tensor, index_tensor) in enumerate(zip(values, indices), start=1):
                    token_index = int(index_tensor.item())
                    count = int(value_tensor.item())
                    eligible = eligible_query_count(
                        observed_queries,
                        token_index,
                        args.remote_only,
                        args.exclude_sink_tokens,
                        args.exclude_recent_tokens,
                    )
                    row = token_row_prefix(tokenizer, token_ids, token_index, args.include_token_text)
                    row.update(
                        {
                            "layer": layer,
                            "head": head,
                            "rank": rank,
                            "eligible_queries": eligible,
                            "selected_count": count,
                            "max_possible_layer_head_query_count": eligible,
                            "selection_rate": count / eligible if eligible else 0.0,
                        }
                    )
                    if args.accumulate_selected_attention_mass:
                        assert collector.layer_head_attention_mass is not None
                        row["selected_attention_mass_sum"] = float(
                            collector.layer_head_attention_mass[layer, head, token_index].item()
                        )
                    top_rows.append(row)
    write_csv(output_dir / "top_tokens_by_layer_head.csv", top_rows, top_fields)

    summary_fields = [
        "scope",
        "layer",
        "head",
        "selection_events",
        "unique_selected_tokens",
        "top_token_index",
        "top_token_id",
    ]
    if args.include_token_text:
        summary_fields += ["top_token_piece", "top_token_text"]
    summary_fields += ["top_token_count", "top_token_event_fraction"]
    summary_rows = [
        make_summary_row(
            collector.total_counts,
            {"scope": "overall", "layer": "", "head": ""},
            collector.total_selection_events,
            tokenizer,
            token_ids,
            args.include_token_text,
        )
    ]
    for layer in range(layer_count):
        summary_rows.append(
            make_summary_row(
                collector.layer_counts[layer],
                {"scope": "layer", "layer": layer, "head": ""},
                int(collector.layer_selection_events[layer].item()),
                tokenizer,
                token_ids,
                args.include_token_text,
            )
        )
    for layer in range(layer_count):
        for head in range(head_count):
            summary_rows.append(
                make_summary_row(
                    collector.layer_head_counts[layer, head],
                    {"scope": "layer_head", "layer": layer, "head": head},
                    int(collector.layer_head_selection_events[layer, head].item()),
                    tokenizer,
                    token_ids,
                    args.include_token_text,
                )
            )
    write_csv(output_dir / "selection_count_summary.csv", summary_rows, summary_fields)

    count_hist = torch.bincount(collector.total_counts)
    hist_rows = [
        {
            "selected_count": count,
            "token_cases": int(case_count.item()),
        }
        for count, case_count in enumerate(count_hist)
        if int(case_count.item()) > 0
    ]
    write_csv(output_dir / "overall_count_histogram.csv", hist_rows, ["selected_count", "token_cases"])

    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens_loaded": int(input_ids.numel()),
            "layer_count": layer_count,
            "head_count": head_count,
            "sampled_query_tokens_requested": query_samples,
            "sampled_query_tokens_observed": observed_queries,
            "topk_by_query": {str(key): value for key, value in sorted(collector.topk_by_query.items())},
            "total_selection_events": collector.total_selection_events,
            "seconds": seconds,
            "metric_definitions": {
                "selected_count": (
                    "Number of times this historical token was in the true full-QK top_fraction set "
                    "across sampled eval queries and the row scope. If remote_only is true, selected "
                    "top_fraction tokens inside excluded sink/recent windows are not counted."
                ),
                "eligible_queries": (
                    "Sampled eval queries whose query_token is greater than token_index. If remote_only is true, "
                    "also requires token_index >= exclude_sink_tokens and query_token > token_index + exclude_recent_tokens."
                ),
                "selection_rate": "selected_count divided by the maximum possible count in the row scope.",
            },
        },
        "paths": {
            "token_selection_counts": str(output_dir / "token_selection_counts.csv"),
            "token_selection_counts_by_layer": str(output_dir / "token_selection_counts_by_layer.csv"),
            "token_selection_counts_by_layer_head": str(output_dir / "token_selection_counts_by_layer_head.csv")
            if args.write_layer_head_token_counts
            else None,
            "top_tokens_by_layer_head": str(output_dir / "top_tokens_by_layer_head.csv"),
            "selection_count_summary": str(output_dir / "selection_count_summary.csv"),
            "overall_count_histogram": str(output_dir / "overall_count_histogram.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": seconds, "total_selection_events": collector.total_selection_events}, indent=2))


if __name__ == "__main__":
    main()
