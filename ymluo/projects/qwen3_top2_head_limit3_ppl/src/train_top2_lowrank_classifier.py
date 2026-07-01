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
_ACTIVE_COLLECTOR: "LowRankTop2Collector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate low-rank query-key classifiers for recovering true full-QK "
            "top-fraction historical tokens."
        )
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="compact_kv,json_kv,needle_sentence,topic_table")
    parser.add_argument("--train_tasks_per_variant", type=int, default=2)
    parser.add_argument("--eval_tasks_per_variant", type=int, default=1)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026063002)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--layers", default="0,4,8,13,20,27")
    parser.add_argument("--heads", default="0,4,8,12")
    parser.add_argument("--ranks", default="4,8,16,32,64")
    parser.add_argument("--max_query_tokens_per_task", type=int, default=2)
    parser.add_argument("--svd_device", default="cuda")
    parser.add_argument("--svd_dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--negative_per_positive", type=int, default=8)
    parser.add_argument("--train_epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--standardize_features", type=str2bool, default=True)
    parser.add_argument("--save_model_weights", type=str2bool, default=False)
    parser.add_argument("--log_every", type=int, default=1)
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


def parse_ranks(value: str, max_rank: int) -> list[int]:
    ranks = sorted({int(part) for part in value.split(",") if part.strip()})
    ranks = [rank for rank in ranks if 1 <= rank <= max_rank]
    if not ranks:
        raise ValueError(f"No ranks in 1..{max_rank} parsed from {value!r}")
    return ranks


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
class RowRecord:
    split: str
    variant: str
    task_id: int
    layer: int
    head: int
    query_token: int
    history_count: int
    top_count: int
    features: torch.Tensor
    labels: torch.Tensor
    attention_mass: torch.Tensor


class LowRankTop2Collector:
    def __init__(
        self,
        selected_layers: list[int],
        selected_heads: list[int],
        max_rank: int,
        top_fraction: float,
        svd_device: torch.device,
        svd_dtype: torch.dtype,
    ) -> None:
        self.selected_layers = set(selected_layers)
        self.selected_heads = set(selected_heads)
        self.max_rank = max_rank
        self.top_fraction = top_fraction
        self.svd_device = svd_device
        self.svd_dtype = svd_dtype
        self.split = "train"
        self.variant = ""
        self.task_id = -1
        self.query_tokens: set[int] = set()
        self.rows: list[RowRecord] = []
        self.skipped_svd_rows = 0

    def set_task(self, *, split: str, variant: str, task_id: int, query_tokens: set[int]) -> None:
        self.split = split
        self.variant = variant
        self.task_id = int(task_id)
        self.query_tokens = query_tokens

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
        top_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
        row_scores = scores[0, :, query_index, :history_count].detach().float()
        attention_weights = F.softmax(scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        top_indices = torch.topk(row_scores, k=top_count, dim=-1, largest=True).indices.detach()

        for head in self.selected_heads:
            key_matrix = key_states[0, head, :history_count, :].detach()
            query_vector = query_states[0, head, query_index, :].detach()
            working = key_matrix.to(device=self.svd_device, dtype=self.svd_dtype)
            centered = working - working.mean(dim=0, keepdim=True)
            try:
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            except RuntimeError:
                self.skipped_svd_rows += 1
                continue
            rank = min(self.max_rank, int(vh.shape[0]))
            basis = vh[:rank].transpose(0, 1)
            q_coeff = query_vector.to(device=self.svd_device, dtype=self.svd_dtype).view(1, -1) @ basis
            k_coeff = key_matrix.to(device=self.svd_device, dtype=self.svd_dtype) @ basis
            features = (k_coeff * q_coeff).detach().cpu().float()
            labels = torch.zeros(history_count, dtype=torch.float32)
            labels[top_indices[head].detach().cpu().long()] = 1.0
            self.rows.append(
                RowRecord(
                    split=self.split,
                    variant=self.variant,
                    task_id=self.task_id,
                    layer=layer,
                    head=head,
                    query_token=query_token,
                    history_count=history_count,
                    top_count=top_count,
                    features=features,
                    labels=labels,
                    attention_mass=attention_weights[head].detach().cpu().float(),
                )
            )


def _lowrank_classifier_eager_attention_forward(
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
        setattr(modeling_qwen3, "eager_attention_forward", _lowrank_classifier_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _lowrank_classifier_eager_attention_forward


@contextmanager
def active_collector(collector: LowRankTop2Collector):
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
    for start in range(0, prefill_tokens, chunk_size):
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
    return past_key_values


@torch.inference_mode()
def run_query_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: LowRankTop2Collector,
) -> None:
    eval_end = prefill_tokens + eval_tokens
    with active_collector(collector):
        for start in range(prefill_tokens, eval_end, chunk_size):
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


def collect_rows(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    collector: LowRankTop2Collector,
    variants: list[str],
    split: str,
    tasks_per_variant: int,
    seed_offset: int,
) -> list[dict[str, Any]]:
    task_rows: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants):
        rng = random.Random(args.seed + seed_offset + 1009 * variant_index)
        tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(tasks_per_variant)]
        for task_number, task in enumerate(tasks, start=1):
            if task_number == 1 or task_number == len(tasks) or task_number % args.log_every == 0:
                print(f"{split} {variant} task {task_number}/{len(tasks)}", flush=True)
            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            query_ids = tokenizer(task["query"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            input_ids = torch.cat([context_ids, query_ids], dim=-1)
            prefill_tokens = int(context_ids.shape[-1])
            eval_tokens = int(query_ids.shape[-1])
            query_tokens = set(choose_query_tokens(prefill_tokens, eval_tokens, args.max_query_tokens_per_task))
            collector.set_task(split=split, variant=variant, task_id=task["task_id"], query_tokens=query_tokens)
            task_rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "task_id": task["task_id"],
                    "prefill_tokens": prefill_tokens,
                    "query_tokens": eval_tokens,
                    "sampled_query_tokens": " ".join(str(token) for token in sorted(query_tokens)),
                    "target_key": task["target_key"],
                    "target_label": task["target_label"],
                }
            )
            past = prefill_cache(model, input_ids, prefill_tokens, args.chunk_size, input_device)
            run_query_eval(
                model,
                input_ids,
                past,
                prefill_tokens,
                eval_tokens,
                args.chunk_size,
                input_device,
                collector,
            )
            del past
            if input_device.type == "cuda":
                torch.cuda.empty_cache()
    return task_rows


def sample_training_examples(
    rows: list[RowRecord],
    rank: int,
    negative_per_positive: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for row in rows:
        labels = row.labels
        pos = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
        neg = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        neg_count = min(int(neg.numel()), max(1, int(pos.numel()) * negative_per_positive))
        perm = torch.randperm(int(neg.numel()), generator=generator)[:neg_count]
        idx = torch.cat([pos, neg[perm]], dim=0)
        xs.append(row.features[idx, :rank])
        ys.append(labels[idx])
    if not xs:
        return torch.empty(0, rank), torch.empty(0)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_linear_classifier(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    standardize: bool,
) -> tuple[torch.nn.Linear | None, torch.Tensor, torch.Tensor, float]:
    if x.numel() == 0 or y.numel() == 0 or float(y.sum().item()) <= 0.0:
        return None, torch.zeros(x.shape[-1]), torch.ones(x.shape[-1]), 0.0
    mean = x.mean(dim=0) if standardize else torch.zeros(x.shape[-1])
    std = x.std(dim=0).clamp_min(1e-6) if standardize else torch.ones(x.shape[-1])
    x_train = (x - mean) / std
    model = torch.nn.Linear(x.shape[-1], 1)
    pos = float(y.sum().item())
    neg = float(y.numel() - y.sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train).flatten()
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(x_train).flatten()
        loss = float(F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight).item())
    return model, mean, std, loss


@dataclass
class MetricAccumulator:
    rows: int = 0
    history_tokens: int = 0
    positive_tokens: int = 0
    hits: int = 0
    true_mass: float = 0.0
    hit_mass: float = 0.0
    score_positive_sum: float = 0.0
    score_negative_sum: float = 0.0

    def add(self, row: RowRecord, scores: torch.Tensor) -> None:
        k = row.top_count
        selected = torch.topk(scores, k=k, largest=True).indices
        labels = row.labels
        hit_mask = labels[selected] > 0.5
        self.rows += 1
        self.history_tokens += row.history_count
        self.positive_tokens += k
        self.hits += int(hit_mask.sum().item())
        true_mask = labels > 0.5
        self.true_mass += float(row.attention_mass[true_mask].sum().item())
        self.hit_mass += float(row.attention_mass[selected[hit_mask]].sum().item())
        if true_mask.any():
            self.score_positive_sum += float(scores[true_mask].mean().item())
        neg_mask = ~true_mask
        if neg_mask.any():
            self.score_negative_sum += float(scores[neg_mask].mean().item())

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            **extra,
            "rows": self.rows,
            "history_tokens": self.history_tokens,
            "positive_tokens": self.positive_tokens,
            "hits": self.hits,
            "recall_at_top_fraction": self.hits / self.positive_tokens if self.positive_tokens else 0.0,
            "attention_mass_recall": self.hit_mass / self.true_mass if self.true_mass > 0.0 else 0.0,
            "true_attention_mass": self.true_mass / self.rows if self.rows else 0.0,
            "mean_positive_score": self.score_positive_sum / self.rows if self.rows else 0.0,
            "mean_negative_score": self.score_negative_sum / self.rows if self.rows else 0.0,
        }


def evaluate_method(
    rows: list[RowRecord],
    rank: int,
    method: str,
    model: torch.nn.Linear | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    accumulators: dict[tuple[str, int | str, int | str], MetricAccumulator] = defaultdict(MetricAccumulator)
    for row in rows:
        x = row.features[:, :rank]
        if method == "lowrank_dot":
            scores = x.sum(dim=1)
        elif method == "trained_linear":
            if model is None or mean is None or std is None:
                continue
            with torch.no_grad():
                scores = model((x - mean) / std).flatten()
        else:
            raise ValueError(f"unknown method: {method}")
        for key in [("overall", "all", "all"), ("layer", row.layer, "all"), ("layer_head", row.layer, row.head)]:
            accumulators[key].add(row, scores)
    out: list[dict[str, Any]] = []
    for (scope, layer, head), acc in sorted(accumulators.items(), key=lambda item: str(item[0])):
        out.append(
            acc.row(
                {
                    "method": method,
                    "rank": rank,
                    "scope": scope,
                    "layer": "" if layer == "all" else layer,
                    "head": "" if head == "all" else head,
                }
            )
        )
    return out


def add_trained_scores(
    accumulators: dict[tuple[str, int | str, int | str], MetricAccumulator],
    row: RowRecord,
    scores: torch.Tensor,
) -> None:
    for key in [("overall", "all", "all"), ("layer", row.layer, "all"), ("layer_head", row.layer, row.head)]:
        accumulators[key].add(row, scores)


def metric_rows_from_accumulators(
    accumulators: dict[tuple[str, int | str, int | str], MetricAccumulator],
    *,
    method: str,
    rank: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scope, layer, head), acc in sorted(accumulators.items(), key=lambda item: str(item[0])):
        rows.append(
            acc.row(
                {
                    "method": method,
                    "rank": rank,
                    "scope": scope,
                    "layer": "" if layer == "all" else layer,
                    "head": "" if head == "all" else head,
                }
            )
        )
    return rows


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
    ranks = parse_ranks(args.ranks, head_dim)
    max_rank = max(ranks)
    svd_device = torch.device(args.svd_device if torch.cuda.is_available() else "cpu")
    svd_dtype = torch.float64 if args.svd_dtype == "float64" else torch.float32
    collector = LowRankTop2Collector(
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        max_rank=max_rank,
        top_fraction=args.top_fraction,
        svd_device=svd_device,
        svd_dtype=svd_dtype,
    )

    started = time.perf_counter()
    task_rows = []
    task_rows += collect_rows(
        args=args,
        model=model,
        tokenizer=tokenizer,
        input_device=input_device,
        collector=collector,
        variants=variants,
        split="train",
        tasks_per_variant=args.train_tasks_per_variant,
        seed_offset=0,
    )
    task_rows += collect_rows(
        args=args,
        model=model,
        tokenizer=tokenizer,
        input_device=input_device,
        collector=collector,
        variants=variants,
        split="eval",
        tasks_per_variant=args.eval_tasks_per_variant,
        seed_offset=100_000,
    )
    collect_seconds = time.perf_counter() - started

    train_rows_by_lh: dict[tuple[int, int], list[RowRecord]] = defaultdict(list)
    eval_rows_by_lh: dict[tuple[int, int], list[RowRecord]] = defaultdict(list)
    for row in collector.rows:
        target = train_rows_by_lh if row.split == "train" else eval_rows_by_lh
        target[(row.layer, row.head)].append(row)

    metric_rows: list[dict[str, Any]] = []
    train_rows_out: list[dict[str, Any]] = []
    weight_state: dict[str, Any] = {}
    train_started = time.perf_counter()
    for rank in ranks:
        eval_rows_all = [row for rows in eval_rows_by_lh.values() for row in rows]
        metric_rows.extend(evaluate_method(eval_rows_all, rank, "lowrank_dot"))
        trained_accumulators: dict[tuple[str, int | str, int | str], MetricAccumulator] = defaultdict(
            MetricAccumulator
        )
        for layer in selected_layers:
            for head in selected_heads:
                train_rows = train_rows_by_lh.get((layer, head), [])
                eval_rows = eval_rows_by_lh.get((layer, head), [])
                x, y = sample_training_examples(
                    train_rows,
                    rank,
                    args.negative_per_positive,
                    args.seed + 17 * rank + 1000 * layer + head,
                )
                clf, mean, std, loss = train_linear_classifier(
                    x,
                    y,
                    epochs=args.train_epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    standardize=args.standardize_features,
                )
                train_rows_out.append(
                    {
                        "layer": layer,
                        "head": head,
                        "rank": rank,
                        "train_rows": len(train_rows),
                        "eval_rows": len(eval_rows),
                        "train_examples": int(y.numel()),
                        "train_positives": int(y.sum().item()) if y.numel() else 0,
                        "final_train_loss": loss,
                    }
                )
                if clf is None:
                    continue
                for row in eval_rows:
                    x_eval = row.features[:, :rank]
                    with torch.no_grad():
                        scores = clf((x_eval - mean) / std).flatten()
                    add_trained_scores(trained_accumulators, row, scores)
                if args.save_model_weights:
                    weight_state[f"layer{layer}_head{head}_rank{rank}"] = {
                        "weight": clf.weight.detach().cpu().tolist(),
                        "bias": clf.bias.detach().cpu().tolist(),
                        "mean": mean.tolist(),
                        "std": std.tolist(),
                    }
        metric_rows.extend(metric_rows_from_accumulators(trained_accumulators, method="trained_linear", rank=rank))
    train_seconds = time.perf_counter() - train_started

    metric_fields = [
        "method",
        "rank",
        "scope",
        "layer",
        "head",
        "rows",
        "history_tokens",
        "positive_tokens",
        "hits",
        "recall_at_top_fraction",
        "attention_mass_recall",
        "true_attention_mass",
        "mean_positive_score",
        "mean_negative_score",
    ]
    write_csv(output_dir / "eval_recall_by_rank.csv", metric_rows, metric_fields)
    write_csv(
        output_dir / "classifier_train_summary.csv",
        train_rows_out,
        [
            "layer",
            "head",
            "rank",
            "train_rows",
            "eval_rows",
            "train_examples",
            "train_positives",
            "final_train_loss",
        ],
    )
    write_csv(
        output_dir / "tasks.csv",
        task_rows,
        [
            "split",
            "variant",
            "task_id",
            "prefill_tokens",
            "query_tokens",
            "sampled_query_tokens",
            "target_key",
            "target_label",
        ],
    )
    if args.save_model_weights:
        (output_dir / "model_weights.json").write_text(json.dumps(weight_state), encoding="utf-8")

    total_seconds = time.perf_counter() - started
    summary = {
        "args": vars(args),
        "resolved": {
            "layer_count": layer_count,
            "head_count": head_count,
            "head_dim": head_dim,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "ranks": ranks,
            "rows_collected": len(collector.rows),
            "train_rows": sum(len(rows) for rows in train_rows_by_lh.values()),
            "eval_rows": sum(len(rows) for rows in eval_rows_by_lh.values()),
            "skipped_svd_rows": collector.skipped_svd_rows,
            "collect_seconds": collect_seconds,
            "train_eval_seconds": train_seconds,
            "total_seconds": total_seconds,
            "feature_definition": (
                "For each row, SVD is fit on centered historical K. Features are per-direction "
                "products (q projected on V_i) * (k projected on V_i). lowrank_dot sums the first "
                "r products. trained_linear learns a per-layer/head weighted sum over the same r products."
            ),
        },
        "paths": {
            "eval_recall_by_rank": str(output_dir / "eval_recall_by_rank.csv"),
            "classifier_train_summary": str(output_dir / "classifier_train_summary.csv"),
            "tasks": str(output_dir / "tasks.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows_collected": len(collector.rows),
                "eval_rows": summary["resolved"]["eval_rows"],
                "total_seconds": total_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
