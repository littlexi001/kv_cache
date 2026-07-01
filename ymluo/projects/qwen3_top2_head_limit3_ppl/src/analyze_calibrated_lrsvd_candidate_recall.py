from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    model_forward,
    pick_input_device,
    resolve_dtype,
)
from run_qabs_downstream_task_suite import BUILDERS  # noqa: E402


_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "CalibratedLrsvdRecallCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate calibrated V_r low-rank candidate selection recall against true full-QK top2 tokens."
        )
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="compact_kv,json_kv,needle_sentence,topic_table")
    parser.add_argument("--tasks_per_variant", type=int, default=8)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026063005)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--candidate_fractions", default="0.02,0.05,0.08")
    parser.add_argument("--calib_samples", default="128,256,512")
    parser.add_argument("--ranks", default="16,32,64")
    parser.add_argument("--layers", default="0,4,8,13,20,27")
    parser.add_argument("--heads", default="0,4,8,12")
    parser.add_argument("--max_query_tokens_per_task", type=int, default=2)
    parser.add_argument("--svd_device", default="cuda")
    parser.add_argument("--svd_dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--center_k", type=str2bool, default=True)
    parser.add_argument("--write_per_query", type=str2bool, default=False)
    parser.add_argument("--log_every", type=int, default=2)
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
    if not selected:
        raise ValueError(f"No {name} selected from spec {spec!r}")
    return sorted(selected)


def parse_positive_ints(value: str, name: str, max_value: int | None = None) -> list[int]:
    values = sorted({int(part) for part in value.split(",") if part.strip()})
    values = [item for item in values if item > 0 and (max_value is None or item <= max_value)]
    if not values:
        raise ValueError(f"No positive {name} parsed from {value!r}")
    return values


def parse_floats(value: str, name: str) -> list[float]:
    values = sorted({float(part) for part in value.split(",") if part.strip()})
    if not values or any(item <= 0.0 or item > 1.0 for item in values):
        raise ValueError(f"{name} must contain values in (0, 1], got {value!r}")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_query_tokens(prefill_tokens: int, query_tokens: int, max_query_tokens: int) -> list[int]:
    positions = list(range(prefill_tokens, prefill_tokens + query_tokens))
    if max_query_tokens <= 0 or len(positions) <= max_query_tokens:
        return positions
    return positions[-max_query_tokens:]


@dataclass
class RecallAccumulator:
    rows: int = 0
    history_tokens: int = 0
    true_top_tokens: int = 0
    requested_candidates: int = 0
    hits: int = 0
    candidate_attention_mass: float = 0.0
    true_top_attention_mass: float = 0.0
    hit_attention_mass: float = 0.0

    def add(
        self,
        *,
        history_count: int,
        true_count: int,
        candidate_count: int,
        hits: int,
        candidate_mass: float,
        true_top_mass: float,
        hit_mass: float,
    ) -> None:
        self.rows += 1
        self.history_tokens += history_count
        self.true_top_tokens += true_count
        self.requested_candidates += candidate_count
        self.hits += hits
        self.candidate_attention_mass += candidate_mass
        self.true_top_attention_mass += true_top_mass
        self.hit_attention_mass += hit_mass

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            **extra,
            "rows": self.rows,
            "history_tokens": self.history_tokens,
            "true_top_tokens": self.true_top_tokens,
            "requested_candidates": self.requested_candidates,
            "actual_candidate_fraction": self.requested_candidates / self.history_tokens if self.history_tokens else 0.0,
            "top2_recall": self.hits / self.true_top_tokens if self.true_top_tokens else 0.0,
            "top2_attention_mass_recall": (
                self.hit_attention_mass / self.true_top_attention_mass if self.true_top_attention_mass > 0.0 else 0.0
            ),
            "candidate_attention_mass_mean": self.candidate_attention_mass / self.rows if self.rows else 0.0,
            "true_top_attention_mass_mean": self.true_top_attention_mass / self.rows if self.rows else 0.0,
        }


class CalibratedLrsvdRecallCollector:
    def __init__(
        self,
        *,
        selected_layers: list[int],
        selected_heads: list[int],
        ranks: list[int],
        calib_samples: list[int],
        candidate_fractions: list[float],
        top_fraction: float,
        svd_device: torch.device,
        svd_dtype: torch.dtype,
        center_k: bool,
        write_per_query: bool,
    ) -> None:
        self.selected_layers = set(selected_layers)
        self.selected_heads = set(selected_heads)
        self.ranks = ranks
        self.calib_samples = calib_samples
        self.candidate_fractions = candidate_fractions
        self.top_fraction = top_fraction
        self.svd_device = svd_device
        self.svd_dtype = svd_dtype
        self.center_k = center_k
        self.write_per_query = write_per_query

        self.variant = ""
        self.task_id = -1
        self.query_tokens: set[int] = set()
        self.basis_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        self.accumulators: dict[tuple[str, Any, Any, int, int, float], RecallAccumulator] = defaultdict(
            RecallAccumulator
        )
        self.per_query_rows: list[dict[str, Any]] = []
        self.skipped_svd = 0
        self.observed_rows = 0

    def set_task(self, *, variant: str, task_id: int, query_tokens: set[int]) -> None:
        self.variant = variant
        self.task_id = int(task_id)
        self.query_tokens = query_tokens
        self.basis_cache = {}

    def _basis_for(
        self,
        *,
        layer: int,
        head: int,
        calib: int,
        key_history: torch.Tensor,
        max_rank: int,
    ) -> torch.Tensor | None:
        cache_key = (layer, head, calib)
        cached = self.basis_cache.get(cache_key)
        if cached is not None and cached.shape[0] >= max_rank:
            return cached[:max_rank]
        if key_history.shape[0] < calib:
            return None
        matrix = key_history[:calib].to(device=self.svd_device, dtype=self.svd_dtype)
        if self.center_k:
            matrix = matrix - matrix.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        except RuntimeError:
            self.skipped_svd += 1
            return None
        self.basis_cache[cache_key] = vh[:max_rank].detach()
        return self.basis_cache[cache_key]

    def observe(
        self,
        layer: int,
        query_token: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        scores: torch.Tensor,
        query_index: int,
    ) -> None:
        if layer not in self.selected_layers or query_token not in self.query_tokens:
            return
        finite = torch.isfinite(scores[:, :, query_index, :])
        valid_count = int(finite[0, 0].sum().item())
        if valid_count <= 2:
            return
        history_count = valid_count - 1
        true_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
        row_scores = scores[0, :, query_index, :history_count].detach().float()
        attention_weights = F.softmax(scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        true_top_indices = torch.topk(row_scores, k=true_count, dim=-1, largest=True).indices.detach()

        for head in range(query_states.shape[1]):
            if head not in self.selected_heads:
                continue
            self.observed_rows += 1
            key_history = key_states[0, head, :history_count, :].detach()
            query = query_states[0, head, query_index, :].detach()
            true_idx = true_top_indices[head].detach().cpu().long()
            true_set = set(int(item) for item in true_idx.tolist())
            attn = attention_weights[head].detach().cpu().float()
            true_top_mass = float(attn[true_idx].sum().item())
            max_rank = min(max(self.ranks), key_history.shape[-1])
            for calib in self.calib_samples:
                basis = self._basis_for(layer=layer, head=head, calib=calib, key_history=key_history, max_rank=max_rank)
                if basis is None:
                    continue
                for rank in self.ranks:
                    if rank > basis.shape[0]:
                        continue
                    basis_rank = basis[:rank].to(device=key_history.device, dtype=torch.float32).transpose(0, 1)
                    q_proj = query.float() @ basis_rank
                    k_proj = key_history.float() @ basis_rank
                    lowrank_scores = (k_proj * q_proj.unsqueeze(0)).sum(dim=-1).detach().cpu().float()
                    for candidate_fraction in self.candidate_fractions:
                        candidate_count = min(
                            history_count,
                            max(1, math.ceil(candidate_fraction * history_count)),
                        )
                        candidate_idx = torch.topk(lowrank_scores, k=candidate_count, largest=True).indices.long()
                        candidate_set = set(int(item) for item in candidate_idx.tolist())
                        hit_items = sorted(true_set & candidate_set)
                        hits = len(hit_items)
                        candidate_mass = float(attn[candidate_idx].sum().item())
                        hit_mass = float(attn[torch.tensor(hit_items, dtype=torch.long)].sum().item()) if hit_items else 0.0
                        for key in [
                            ("overall", "all", "all", calib, rank, candidate_fraction),
                            ("variant", self.variant, "all", calib, rank, candidate_fraction),
                            ("layer", layer, "all", calib, rank, candidate_fraction),
                            ("layer_head", layer, head, calib, rank, candidate_fraction),
                        ]:
                            self.accumulators[key].add(
                                history_count=history_count,
                                true_count=true_count,
                                candidate_count=candidate_count,
                                hits=hits,
                                candidate_mass=candidate_mass,
                                true_top_mass=true_top_mass,
                                hit_mass=hit_mass,
                            )
                        if self.write_per_query:
                            self.per_query_rows.append(
                                {
                                    "variant": self.variant,
                                    "task_id": self.task_id,
                                    "query_token": query_token,
                                    "layer": layer,
                                    "head": head,
                                    "calib_samples": calib,
                                    "rank": rank,
                                    "candidate_fraction": candidate_fraction,
                                    "history_tokens": history_count,
                                    "true_top_tokens": true_count,
                                    "candidate_tokens": candidate_count,
                                    "hits": hits,
                                    "top2_recall": hits / true_count if true_count else 0.0,
                                    "top2_attention_mass_recall": hit_mass / true_top_mass if true_top_mass > 0 else 0.0,
                                }
                            )

    def summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for (scope, layer_or_variant, head, calib, rank, candidate_fraction), acc in sorted(
            self.accumulators.items(), key=lambda item: str(item[0])
        ):
            rows.append(
                acc.row(
                    {
                        "scope": scope,
                        "variant": layer_or_variant if scope == "variant" else "",
                        "layer": layer_or_variant if scope in {"layer", "layer_head"} else "",
                        "head": head if scope == "layer_head" else "",
                        "calib_samples": calib,
                        "rank": rank,
                        "candidate_fraction": candidate_fraction,
                    }
                )
            )
        return rows


def _calibrated_lrsvd_eager_attention_forward(
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
        setattr(modeling_qwen3, "eager_attention_forward", _calibrated_lrsvd_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _calibrated_lrsvd_eager_attention_forward


@contextmanager
def active_collector(collector: CalibratedLrsvdRecallCollector):
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
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        del outputs
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"prefill chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
    return past_key_values


@torch.inference_mode()
def run_query(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: CalibratedLrsvdRecallCollector,
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
            outputs = model_forward(model, kwargs)
            past_key_values = outputs.past_key_values
            del outputs
            if input_device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"eval chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top_fraction must be in (0, 1].")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(BUILDERS)}")
    candidate_fractions = parse_floats(args.candidate_fractions, "candidate_fractions")
    calib_samples = parse_positive_ints(args.calib_samples, "calib_samples")

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
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
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
    selected_layers = parse_index_spec(args.layers, layer_count, "layers")
    selected_heads = parse_index_spec(args.heads, head_count, "heads")
    ranks = parse_positive_ints(args.ranks, "ranks", max_value=head_dim)
    svd_device = torch.device(args.svd_device if torch.cuda.is_available() else "cpu")
    svd_dtype = torch.float64 if args.svd_dtype == "float64" else torch.float32
    collector = CalibratedLrsvdRecallCollector(
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        ranks=ranks,
        calib_samples=calib_samples,
        candidate_fractions=candidate_fractions,
        top_fraction=args.top_fraction,
        svd_device=svd_device,
        svd_dtype=svd_dtype,
        center_k=args.center_k,
        write_per_query=args.write_per_query,
    )

    task_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for variant_index, variant in enumerate(variants):
        rng = random.Random(args.seed + 1009 * variant_index)
        tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(args.tasks_per_variant)]
        for task_number, task in enumerate(tasks, start=1):
            if task_number == 1 or task_number == len(tasks) or task_number % args.log_every == 0:
                print(f"{variant} task {task_number}/{len(tasks)}", flush=True)
            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            query_ids = tokenizer(task["query"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            input_ids = torch.cat([context_ids, query_ids], dim=-1)
            prefill_tokens = int(context_ids.shape[-1])
            eval_tokens = int(query_ids.shape[-1])
            query_tokens = set(choose_query_tokens(prefill_tokens, eval_tokens, args.max_query_tokens_per_task))
            collector.set_task(variant=variant, task_id=task["task_id"], query_tokens=query_tokens)
            task_rows.append(
                {
                    "variant": variant,
                    "task_id": task["task_id"],
                    "prefill_tokens": prefill_tokens,
                    "query_tokens": eval_tokens,
                    "sampled_query_tokens": " ".join(str(item) for item in sorted(query_tokens)),
                    "target_key": task["target_key"],
                    "target_label": task["target_label"],
                }
            )
            past = prefill_cache(model, input_ids, prefill_tokens, args.chunk_size, input_device)
            run_query(model, input_ids, past, prefill_tokens, eval_tokens, args.chunk_size, input_device, collector)
            del past
            if input_device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = collector.summary_rows()
    summary_fields = [
        "scope",
        "variant",
        "layer",
        "head",
        "calib_samples",
        "rank",
        "candidate_fraction",
        "rows",
        "history_tokens",
        "true_top_tokens",
        "requested_candidates",
        "actual_candidate_fraction",
        "top2_recall",
        "top2_attention_mass_recall",
        "candidate_attention_mass_mean",
        "true_top_attention_mass_mean",
    ]
    write_csv(output_dir / "candidate_recall_summary.csv", summary_rows, summary_fields)
    write_csv(
        output_dir / "tasks.csv",
        task_rows,
        [
            "variant",
            "task_id",
            "prefill_tokens",
            "query_tokens",
            "sampled_query_tokens",
            "target_key",
            "target_label",
        ],
    )
    if args.write_per_query:
        write_csv(
            output_dir / "candidate_recall_per_query.csv",
            collector.per_query_rows,
            [
                "variant",
                "task_id",
                "query_token",
                "layer",
                "head",
                "calib_samples",
                "rank",
                "candidate_fraction",
                "history_tokens",
                "true_top_tokens",
                "candidate_tokens",
                "hits",
                "top2_recall",
                "top2_attention_mass_recall",
            ],
        )

    seconds = time.perf_counter() - started
    summary = {
        "args": vars(args),
        "resolved": {
            "layer_count": layer_count,
            "head_count": head_count,
            "head_dim": head_dim,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "ranks": ranks,
            "calib_samples": calib_samples,
            "candidate_fractions": candidate_fractions,
            "tasks": len(task_rows),
            "observed_layer_head_query_rows": collector.observed_rows,
            "skipped_svd": collector.skipped_svd,
            "seconds": seconds,
            "method": (
                "For each task/layer/head/calib size, estimate V_r once from the first calib_samples "
                "historical K tokens. For sampled query rows, score all historical tokens with projected "
                "lowrank q-k dot, take top candidate_fraction, and compare against true full-QK top_fraction."
            ),
        },
        "paths": {
            "candidate_recall_summary": str(output_dir / "candidate_recall_summary.csv"),
            "tasks": str(output_dir / "tasks.csv"),
            "candidate_recall_per_query": str(output_dir / "candidate_recall_per_query.csv")
            if args.write_per_query
            else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": seconds, "summary_rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
