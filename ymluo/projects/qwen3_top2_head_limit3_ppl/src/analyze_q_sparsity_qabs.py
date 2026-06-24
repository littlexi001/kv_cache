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
_ACTIVE_COLLECTOR: "QSparsityCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether qabs partial-QK works because Q vectors are sparse or magnitude-concentrated."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/q_sparsity_qabs")
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
    parser.add_argument("--candidate_fractions", default="0.20,0.40")
    parser.add_argument("--max_query_samples", type=int, default=32)
    parser.add_argument("--query_stride", type=int, default=0)
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
class QStatsAccumulator:
    cases: int = 0
    zero_1e_8_sum: float = 0.0
    zero_1e_6_sum: float = 0.0
    zero_1e_4_sum: float = 0.0
    small_1pct_max_sum: float = 0.0
    effective_l2_dim_sum: float = 0.0
    max_over_mean_abs_sum: float = 0.0
    top_l1_sum: dict[int, float] | None = None
    top_l2_sum: dict[int, float] | None = None

    def add(self, q: torch.Tensor, dim_counts: list[int]) -> None:
        q_abs = q.abs()
        q2 = q * q
        head_dim = q.numel()
        if self.top_l1_sum is None:
            self.top_l1_sum = {dim: 0.0 for dim in dim_counts}
            self.top_l2_sum = {dim: 0.0 for dim in dim_counts}
        self.cases += 1
        self.zero_1e_8_sum += float((q_abs < 1e-8).float().mean().item())
        self.zero_1e_6_sum += float((q_abs < 1e-6).float().mean().item())
        self.zero_1e_4_sum += float((q_abs < 1e-4).float().mean().item())
        max_abs = float(q_abs.max().item())
        self.small_1pct_max_sum += float((q_abs < 0.01 * max_abs).float().mean().item()) if max_abs > 0 else 1.0
        l1_total = float(q_abs.sum().item())
        l2_total = float(q2.sum().item())
        q4_total = float((q2 * q2).sum().item())
        self.effective_l2_dim_sum += (l2_total * l2_total / q4_total) if q4_total > 0 else 0.0
        self.max_over_mean_abs_sum += max_abs / float(q_abs.mean().item()) if float(q_abs.mean().item()) > 0 else 0.0
        sorted_abs = torch.sort(q_abs, descending=True).values
        sorted_q2 = sorted_abs * sorted_abs
        for dim in dim_counts:
            d = min(dim, head_dim)
            self.top_l1_sum[dim] += float(sorted_abs[:d].sum().item()) / l1_total if l1_total > 0 else 0.0
            self.top_l2_sum[dim] += float(sorted_q2[:d].sum().item()) / l2_total if l2_total > 0 else 0.0

    def row(self, extra: dict[str, Any], dim_counts: list[int]) -> dict[str, Any]:
        row = {
            **extra,
            "cases": self.cases,
            "exact_zero_fraction_abs_lt_1e_8": self.zero_1e_8_sum / self.cases if self.cases else 0.0,
            "near_zero_fraction_abs_lt_1e_6": self.zero_1e_6_sum / self.cases if self.cases else 0.0,
            "near_zero_fraction_abs_lt_1e_4": self.zero_1e_4_sum / self.cases if self.cases else 0.0,
            "fraction_abs_lt_1pct_row_max": self.small_1pct_max_sum / self.cases if self.cases else 0.0,
            "effective_l2_dimension_mean": self.effective_l2_dim_sum / self.cases if self.cases else 0.0,
            "max_over_mean_abs_mean": self.max_over_mean_abs_sum / self.cases if self.cases else 0.0,
        }
        for dim in dim_counts:
            row[f"top{dim}_l1_mass"] = (self.top_l1_sum or {}).get(dim, 0.0) / self.cases if self.cases else 0.0
            row[f"top{dim}_l2_energy"] = (self.top_l2_sum or {}).get(dim, 0.0) / self.cases if self.cases else 0.0
        return row


@dataclass
class CandidateAccumulator:
    cases: int = 0
    history_tokens: int = 0
    true_top_tokens: int = 0
    requested_candidates: int = 0
    actual_candidates: int = 0
    true_hits: int = 0
    candidate_attention_mass_sum: float = 0.0
    selected_attention_mass_sum: float = 0.0
    true_top_attention_mass_sum: float = 0.0

    def add(
        self,
        history_tokens: int,
        true_top_tokens: int,
        requested_candidates: int,
        actual_candidates: int,
        true_hits: int,
        candidate_attention_mass: float,
        selected_attention_mass: float,
        true_top_attention_mass: float,
    ) -> None:
        self.cases += 1
        self.history_tokens += history_tokens
        self.true_top_tokens += true_top_tokens
        self.requested_candidates += requested_candidates
        self.actual_candidates += actual_candidates
        self.true_hits += true_hits
        self.candidate_attention_mass_sum += candidate_attention_mass
        self.selected_attention_mass_sum += selected_attention_mass
        self.true_top_attention_mass_sum += true_top_attention_mass

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            **extra,
            "cases": self.cases,
            "requested_candidate_fraction": (
                self.requested_candidates / self.history_tokens if self.history_tokens else 0.0
            ),
            "actual_candidate_fraction": self.actual_candidates / self.history_tokens if self.history_tokens else 0.0,
            "true_top_overlap": self.true_hits / self.true_top_tokens if self.true_top_tokens else 0.0,
            "candidate_attention_mass_mean": (
                self.candidate_attention_mass_sum / self.cases if self.cases else 0.0
            ),
            "selected_attention_mass_mean": self.selected_attention_mass_sum / self.cases if self.cases else 0.0,
            "true_top_attention_mass_mean": (
                self.true_top_attention_mass_sum / self.cases if self.cases else 0.0
            ),
        }


class QSparsityCollector:
    def __init__(
        self,
        layer_count: int,
        head_count: int,
        query_tokens: set[int],
        top_fraction: float,
        dim_counts: list[int],
        candidate_fractions: list[float],
        write_layer_head: bool,
        seed: int,
    ) -> None:
        self.layer_count = layer_count
        self.head_count = head_count
        self.query_tokens = query_tokens
        self.top_fraction = top_fraction
        self.dim_counts = dim_counts
        self.candidate_fractions = candidate_fractions
        self.write_layer_head = write_layer_head
        self.random_generator = torch.Generator(device="cpu")
        self.random_generator.manual_seed(seed)
        self.q_summary = QStatsAccumulator()
        self.q_by_layer: dict[int, QStatsAccumulator] = defaultdict(QStatsAccumulator)
        self.q_by_layer_head: dict[tuple[int, int], QStatsAccumulator] = defaultdict(QStatsAccumulator)
        self.candidate_by_rule: dict[tuple[str, int, float], CandidateAccumulator] = defaultdict(CandidateAccumulator)
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

        q = query_states[0, :, query_index, :].detach().float()
        k = key_states[0, :, :history_count, :].detach().float()
        row_scores = scores[0, :, query_index, :history_count].detach().float()
        attention_weights = F.softmax(scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        true_top_indices = torch.topk(row_scores, k=true_count, dim=-1, largest=True).indices
        head_dim = q.shape[-1]

        for head in range(self.head_count):
            q_head = q[head]
            self.q_summary.add(q_head, self.dim_counts)
            self.q_by_layer[layer].add(q_head, self.dim_counts)
            if self.write_layer_head:
                self.q_by_layer_head[(layer, head)].add(q_head, self.dim_counts)

            true_idx = true_top_indices[head]
            true_top_mass = float(attention_weights[head, true_idx].sum().item())
            for dim in self.dim_counts:
                d = min(dim, head_dim)
                qabs_indices = torch.topk(q_head.abs(), k=d, dim=-1, largest=True).indices
                random_indices = torch.randperm(head_dim, generator=self.random_generator)[:d].to(q_head.device)
                for rule_name, dim_indices in {"qabs": qabs_indices, "random": random_indices}.items():
                    q_selected = q_head[dim_indices]
                    k_selected = k[head, :, dim_indices]
                    partial_scores = (k_selected * q_selected[None, :]).sum(dim=-1)
                    for candidate_fraction in self.candidate_fractions:
                        requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
                        threshold = torch.topk(partial_scores, k=requested, largest=True).values[-1]
                        candidate_mask = partial_scores >= threshold
                        exact_candidate_scores = row_scores[head].masked_fill(
                            ~candidate_mask,
                            torch.finfo(row_scores.dtype).min,
                        )
                        selected_idx = torch.topk(exact_candidate_scores, k=true_count, largest=True).indices
                        selected_mask = torch.zeros(history_count, dtype=torch.bool, device=row_scores.device)
                        selected_mask[selected_idx] = True
                        hits = int(selected_mask[true_idx].sum().item())
                        self.candidate_by_rule[(rule_name, dim, candidate_fraction)].add(
                            history_tokens=history_count,
                            true_top_tokens=true_count,
                            requested_candidates=requested,
                            actual_candidates=int(candidate_mask.sum().item()),
                            true_hits=hits,
                            candidate_attention_mass=float(attention_weights[head, candidate_mask].sum().item()),
                            selected_attention_mass=float(attention_weights[head, selected_idx].sum().item()),
                            true_top_attention_mass=true_top_mass,
                        )

    def q_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        overall = [self.q_summary.row({"scope": "overall"}, self.dim_counts)]
        by_layer = [
            acc.row({"scope": "layer", "layer": layer}, self.dim_counts)
            for layer, acc in sorted(self.q_by_layer.items())
        ]
        by_layer_head = [
            acc.row({"scope": "layer_head", "layer": layer, "head": head}, self.dim_counts)
            for (layer, head), acc in sorted(self.q_by_layer_head.items())
        ]
        return overall, by_layer, by_layer_head

    def candidate_rows(self) -> list[dict[str, Any]]:
        rows = []
        for (rule_name, dim, candidate_fraction), acc in sorted(self.candidate_by_rule.items()):
            rows.append(
                acc.row(
                    {
                        "dimension_rule": rule_name,
                        "dim_count": dim,
                        "candidate_fraction": candidate_fraction,
                    }
                )
            )
        return rows


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
        setattr(modeling_qwen3, "eager_attention_forward", _patched_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _patched_eager_attention_forward


@contextmanager
def active_collector(collector: QSparsityCollector):
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
    collector: QSparsityCollector,
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
    candidate_fractions = [float(part) for part in args.candidate_fractions.split(",") if part.strip()]
    if not dim_counts or any(dim <= 0 for dim in dim_counts):
        raise ValueError("--dim_counts must contain positive integers.")
    if not candidate_fractions or any(frac <= 0.0 or frac > 1.0 for frac in candidate_fractions):
        raise ValueError("--candidate_fractions must be in (0, 1].")
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
    head_dim = int(getattr(model.config, "head_dim", 0) or (model.config.hidden_size // head_count))

    query_samples = build_query_samples(args.prefill_tokens, args.eval_tokens, args.query_stride, args.max_query_samples)
    collector = QSparsityCollector(
        layer_count=layer_count,
        head_count=head_count,
        query_tokens=set(query_samples),
        top_fraction=args.top_fraction,
        dim_counts=dim_counts,
        candidate_fractions=candidate_fractions,
        write_layer_head=args.write_layer_head,
        seed=args.seed,
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

    q_overall, q_by_layer, q_by_layer_head = collector.q_rows()
    q_fields = list(q_overall[0].keys())
    write_csv(output_dir / "q_sparsity_summary.csv", q_overall, q_fields)
    write_csv(output_dir / "q_sparsity_by_layer.csv", q_by_layer, list(q_by_layer[0].keys()) if q_by_layer else q_fields)
    if args.write_layer_head and q_by_layer_head:
        write_csv(output_dir / "q_sparsity_by_layer_head.csv", q_by_layer_head, list(q_by_layer_head[0].keys()))

    candidate_rows = collector.candidate_rows()
    write_csv(
        output_dir / "candidate_rule_comparison.csv",
        candidate_rows,
        [
            "dimension_rule",
            "dim_count",
            "candidate_fraction",
            "cases",
            "requested_candidate_fraction",
            "actual_candidate_fraction",
            "true_top_overlap",
            "candidate_attention_mass_mean",
            "selected_attention_mass_mean",
            "true_top_attention_mass_mean",
        ],
    )

    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens_loaded": int(input_ids.numel()),
            "layer_count": layer_count,
            "head_count": head_count,
            "head_dim": head_dim,
            "sampled_query_tokens_requested": query_samples,
            "sampled_query_tokens_observed": sorted(collector.observed_query_tokens),
            "seconds": seconds,
            "metric_definitions": {
                "exact_zero_fraction_abs_lt_1e_8": "fraction of q dimensions with abs(q_i) < 1e-8",
                "effective_l2_dimension_mean": "(sum q_i^2)^2 / sum q_i^4; lower means more concentrated energy",
                "topD_l2_energy": "fraction of q L2 energy in the largest D absolute dimensions",
                "qabs": "candidate dimensions chosen by largest |q_i| per query/head",
                "random": "candidate dimensions chosen randomly per query/head for comparison",
            },
        },
        "paths": {
            "q_sparsity_summary": str(output_dir / "q_sparsity_summary.csv"),
            "candidate_rule_comparison": str(output_dir / "candidate_rule_comparison.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": seconds, "q_summary": q_overall, "candidate_rows": candidate_rows}, indent=2))


if __name__ == "__main__":
    main()
