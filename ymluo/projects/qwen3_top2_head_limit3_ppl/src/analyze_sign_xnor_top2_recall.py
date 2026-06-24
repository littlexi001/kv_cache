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


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-0.6B"
DEFAULT_TEXT_PATH = "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "SignXnorRecallCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how well sign-XNOR/popcount candidates recover true QK top attention tokens "
            "on sampled Qwen3 attention rows."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/sign_xnor_top2_recall_10k")
    parser.add_argument("--total_tokens", type=int, default=10_000)
    parser.add_argument("--prefill_tokens", type=int, default=9_000)
    parser.add_argument("--eval_tokens", type=int, default=1_000)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--max_chars", type=int, default=20_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument(
        "--candidate_fractions",
        default="0.02,0.05,0.10,0.20",
        help="Comma-separated fractions of historical tokens requested as sign-XNOR candidates.",
    )
    parser.add_argument(
        "--max_query_samples",
        type=int,
        default=32,
        help="Maximum eval query positions to sample. Use <=0 to sample all eval positions.",
    )
    parser.add_argument(
        "--query_stride",
        type=int,
        default=0,
        help="If >0, sample every N eval queries before applying max_query_samples.",
    )
    parser.add_argument("--write_per_query", type=str2bool, default=True)
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
class RecallAccumulator:
    cases: int = 0
    history_tokens: int = 0
    true_top_tokens: int = 0
    requested_candidates: int = 0
    actual_candidates: int = 0
    true_hits: int = 0
    candidate_attention_mass_sum: float = 0.0
    true_top_attention_mass_sum: float = 0.0
    sign_threshold_sum: float = 0.0

    def add(
        self,
        history_tokens: int,
        true_top_tokens: int,
        requested_candidates: int,
        actual_candidates: int,
        true_hits: int,
        candidate_attention_mass: float,
        true_top_attention_mass: float,
        sign_threshold: float,
    ) -> None:
        self.cases += 1
        self.history_tokens += history_tokens
        self.true_top_tokens += true_top_tokens
        self.requested_candidates += requested_candidates
        self.actual_candidates += actual_candidates
        self.true_hits += true_hits
        self.candidate_attention_mass_sum += candidate_attention_mass
        self.true_top_attention_mass_sum += true_top_attention_mass
        self.sign_threshold_sum += sign_threshold

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        recall = self.true_hits / self.true_top_tokens if self.true_top_tokens else 0.0
        actual_fraction = self.actual_candidates / self.history_tokens if self.history_tokens else 0.0
        requested_fraction = self.requested_candidates / self.history_tokens if self.history_tokens else 0.0
        return {
            **extra,
            "cases": self.cases,
            "history_tokens": self.history_tokens,
            "true_top_tokens": self.true_top_tokens,
            "requested_candidates": self.requested_candidates,
            "actual_candidates": self.actual_candidates,
            "requested_candidate_fraction": requested_fraction,
            "actual_candidate_fraction": actual_fraction,
            "true_top_recall": recall,
            "candidate_attention_mass_mean": self.candidate_attention_mass_sum / self.cases if self.cases else 0.0,
            "true_top_attention_mass_mean": self.true_top_attention_mass_sum / self.cases if self.cases else 0.0,
            "mean_sign_threshold": self.sign_threshold_sum / self.cases if self.cases else 0.0,
        }


class SignXnorRecallCollector:
    def __init__(
        self,
        layer_count: int,
        head_count: int,
        query_tokens: set[int],
        top_fraction: float,
        candidate_fractions: list[float],
        write_per_query: bool,
    ) -> None:
        self.layer_count = layer_count
        self.head_count = head_count
        self.query_tokens = query_tokens
        self.top_fraction = top_fraction
        self.candidate_fractions = candidate_fractions
        self.write_per_query = write_per_query
        self.by_fraction: dict[float, RecallAccumulator] = defaultdict(RecallAccumulator)
        self.by_layer_fraction: dict[tuple[int, float], RecallAccumulator] = defaultdict(RecallAccumulator)
        self.by_layer_head_fraction: dict[tuple[int, int, float], RecallAccumulator] = defaultdict(RecallAccumulator)
        self.per_query_rows: list[dict[str, Any]] = []
        self.observed_query_tokens: set[int] = set()

    def observe(
        self,
        layer: int,
        query_token: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        scores: torch.Tensor,
        query_index: int,
    ) -> None:
        if query_token not in self.query_tokens:
            return
        finite = torch.isfinite(scores[:, :, query_index, :])
        valid_count = int(finite[0, 0].sum().item())
        if valid_count <= 1:
            return
        history_count = valid_count - 1
        true_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
        self.observed_query_tokens.add(query_token)

        q = query_states[0, :, query_index, :].detach()
        k = key_states[0, :, :history_count, :].detach()
        q_sign = torch.signbit(q)
        k_sign = torch.signbit(k)
        same_counts = (k_sign == q_sign[:, None, :]).sum(dim=-1)

        row_scores = scores[0, :, query_index, :history_count].detach().float()
        attention_weights = F.softmax(scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        true_top_indices = torch.topk(row_scores, k=true_count, dim=-1, largest=True).indices

        for head in range(self.head_count):
            true_idx = true_top_indices[head]
            true_top_mass = float(attention_weights[head, true_idx].sum().item())
            sign_scores = same_counts[head]
            for candidate_fraction in self.candidate_fractions:
                requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
                threshold = int(torch.topk(sign_scores, k=requested, largest=True).values[-1].item())
                candidate_mask = sign_scores >= threshold
                actual = int(candidate_mask.sum().item())
                hits = int(candidate_mask[true_idx].sum().item())
                candidate_mass = float(attention_weights[head, candidate_mask].sum().item())
                self.by_fraction[candidate_fraction].add(
                    history_count,
                    true_count,
                    requested,
                    actual,
                    hits,
                    candidate_mass,
                    true_top_mass,
                    float(threshold),
                )
                self.by_layer_fraction[(layer, candidate_fraction)].add(
                    history_count,
                    true_count,
                    requested,
                    actual,
                    hits,
                    candidate_mass,
                    true_top_mass,
                    float(threshold),
                )
                self.by_layer_head_fraction[(layer, head, candidate_fraction)].add(
                    history_count,
                    true_count,
                    requested,
                    actual,
                    hits,
                    candidate_mass,
                    true_top_mass,
                    float(threshold),
                )
                if self.write_per_query:
                    self.per_query_rows.append(
                        {
                            "query_token": query_token,
                            "layer": layer,
                            "head": head,
                            "candidate_fraction": candidate_fraction,
                            "history_tokens": history_count,
                            "true_top_tokens": true_count,
                            "requested_candidates": requested,
                            "actual_candidates": actual,
                            "actual_candidate_fraction": actual / history_count,
                            "true_hits": hits,
                            "true_top_recall": hits / true_count,
                            "candidate_attention_mass": candidate_mass,
                            "true_top_attention_mass": true_top_mass,
                            "sign_threshold": threshold,
                        }
                    )

    def fraction_rows(self) -> list[dict[str, Any]]:
        return [
            self.by_fraction[fraction].row({"candidate_fraction": fraction})
            for fraction in sorted(self.by_fraction)
        ]

    def layer_rows(self) -> list[dict[str, Any]]:
        rows = []
        for layer in range(self.layer_count):
            for fraction in self.candidate_fractions:
                acc = self.by_layer_fraction.get((layer, fraction))
                if acc is not None:
                    rows.append(acc.row({"layer": layer, "candidate_fraction": fraction}))
        return rows

    def layer_head_rows(self) -> list[dict[str, Any]]:
        rows = []
        for layer in range(self.layer_count):
            for head in range(self.head_count):
                for fraction in self.candidate_fractions:
                    acc = self.by_layer_head_fraction.get((layer, head, fraction))
                    if acc is not None:
                        rows.append(acc.row({"layer": layer, "head": head, "candidate_fraction": fraction}))
        return rows


def _sign_xnor_eager_attention_forward(
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
            _ACTIVE_COLLECTOR.observe(layer, query_token, query_states, key_states, scores, query_index)

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
        setattr(modeling_qwen3, "eager_attention_forward", _sign_xnor_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _sign_xnor_eager_attention_forward


@contextmanager
def active_collector(collector: SignXnorRecallCollector):
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


@torch.inference_mode()
def run_eval_samples(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: SignXnorRecallCollector,
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
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top_fraction must be in (0, 1].")
    candidate_fractions = [float(part) for part in args.candidate_fractions.split(",") if part.strip()]
    if any(fraction <= 0.0 or fraction > 1.0 for fraction in candidate_fractions):
        raise ValueError("--candidate_fractions must all be in (0, 1].")
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

    query_samples = build_query_samples(args.prefill_tokens, args.eval_tokens, args.query_stride, args.max_query_samples)
    collector = SignXnorRecallCollector(
        layer_count=layer_count,
        head_count=head_count,
        query_tokens=set(query_samples),
        top_fraction=args.top_fraction,
        candidate_fractions=candidate_fractions,
        write_per_query=args.write_per_query,
    )

    started = time.perf_counter()
    past = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    run_eval_samples(
        model,
        input_ids,
        past,
        args.prefill_tokens,
        args.eval_tokens,
        args.chunk_size,
        input_device,
        collector,
    )
    seconds = time.perf_counter() - started

    fraction_rows = collector.fraction_rows()
    layer_rows = collector.layer_rows()
    layer_head_rows = collector.layer_head_rows()
    common_fields = [
        "candidate_fraction",
        "cases",
        "history_tokens",
        "true_top_tokens",
        "requested_candidates",
        "actual_candidates",
        "requested_candidate_fraction",
        "actual_candidate_fraction",
        "true_top_recall",
        "candidate_attention_mass_mean",
        "true_top_attention_mass_mean",
        "mean_sign_threshold",
    ]
    write_csv(output_dir / "recall_by_candidate_fraction.csv", fraction_rows, common_fields)
    write_csv(output_dir / "recall_by_layer.csv", layer_rows, ["layer"] + common_fields)
    write_csv(output_dir / "recall_by_layer_head.csv", layer_head_rows, ["layer", "head"] + common_fields)
    if args.write_per_query:
        write_csv(
            output_dir / "per_query_recall.csv",
            collector.per_query_rows,
            [
                "query_token",
                "layer",
                "head",
                "candidate_fraction",
                "history_tokens",
                "true_top_tokens",
                "requested_candidates",
                "actual_candidates",
                "actual_candidate_fraction",
                "true_hits",
                "true_top_recall",
                "candidate_attention_mass",
                "true_top_attention_mass",
                "sign_threshold",
            ],
        )

    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens_loaded": int(input_ids.numel()),
            "layer_count": layer_count,
            "head_count": head_count,
            "sampled_query_tokens_requested": query_samples,
            "sampled_query_tokens_observed": sorted(collector.observed_query_tokens),
            "seconds": seconds,
            "rule": (
                "For each sampled layer/head/query row, true positives are QK-score top_fraction historical keys. "
                "Sign-XNOR candidates are all keys whose sign-match popcount is at least the bucket threshold "
                "needed to request candidate_fraction of historical keys; ties are included, so actual candidate "
                "fraction can exceed the requested fraction."
            ),
        },
        "paths": {
            "recall_by_candidate_fraction": str(output_dir / "recall_by_candidate_fraction.csv"),
            "recall_by_layer": str(output_dir / "recall_by_layer.csv"),
            "recall_by_layer_head": str(output_dir / "recall_by_layer_head.csv"),
            "per_query_recall": str(output_dir / "per_query_recall.csv") if args.write_per_query else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": seconds, "fraction_rows": fraction_rows}, indent=2))


if __name__ == "__main__":
    main()
