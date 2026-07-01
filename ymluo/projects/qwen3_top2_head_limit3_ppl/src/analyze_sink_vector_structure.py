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
from run_qabs_evidence_span_coverage import evidence_spans  # noqa: E402


_ORIGINAL_QWEN3_ATTENTION_FORWARD: Any | None = None
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "SinkVectorCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze why sink K vectors align with many Q vectors.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="compact_kv,json_kv,needle_sentence,topic_table")
    parser.add_argument("--tasks_per_variant", type=int, default=4)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026070101)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--layers", default="0,4,8,13,20,27")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--sink_tokens", type=int, default=16)
    parser.add_argument("--recent_tokens", type=int, default=16)
    parser.add_argument("--other_sample_tokens", type=int, default=32)
    parser.add_argument("--max_query_tokens_per_task", type=int, default=2)
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


def safe_unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-12)


def token_set_from_span(span: tuple[int, int] | None) -> set[int]:
    if span is None:
        return set()
    return set(range(span[0], span[1]))


def build_groups(
    *,
    history_count: int,
    spans: dict[str, tuple[int, int]],
    sink_tokens: int,
    recent_tokens: int,
    other_sample_tokens: int,
) -> dict[str, list[int]]:
    groups: dict[str, set[int]] = {
        "sink_first2": set(range(0, min(2, history_count))),
        "sink_rest": set(range(min(2, history_count), min(sink_tokens, history_count))),
        "sink_all": set(range(0, min(sink_tokens, history_count))),
        "front_positions": set(range(0, min(sink_tokens, history_count))),
        "recent": set(range(max(0, history_count - recent_tokens), history_count)),
        "evidence_key": token_set_from_span(spans.get("key")),
        "evidence_label": token_set_from_span(spans.get("label")),
        "evidence_record": token_set_from_span(spans.get("record")),
    }
    evidence_any = set()
    for name, tokens in groups.items():
        if name.startswith("evidence_"):
            evidence_any.update(tokens)
    groups["evidence_any"] = evidence_any
    covered = set().union(*groups.values()) if groups else set()
    other = [idx for idx in range(history_count) if idx not in covered]
    groups["other_sample"] = set(other[: max(0, other_sample_tokens)])
    return {name: sorted(idx for idx in tokens if 0 <= idx < history_count) for name, tokens in groups.items() if tokens}


class SinkVectorCollector:
    def __init__(
        self,
        *,
        selected_layers: list[int],
        selected_heads: list[int],
        sink_tokens: int,
        recent_tokens: int,
        other_sample_tokens: int,
    ) -> None:
        self.selected_layers = set(selected_layers)
        self.selected_heads = set(selected_heads)
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.other_sample_tokens = other_sample_tokens
        self.query_tokens: set[int] = set()
        self.spans: dict[str, tuple[int, int]] = {}
        self.pre_k_history: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.current_pre_q: dict[int, torch.Tensor] = {}
        self.pair_events: dict[tuple[str, int, int, str], list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        self.q_events: dict[tuple[str, int, int], list[torch.Tensor]] = defaultdict(list)
        self.observed_rows = 0

    def set_task(self, *, query_tokens: set[int], spans: dict[str, tuple[int, int]]) -> None:
        self.query_tokens = query_tokens
        self.spans = spans
        self.pre_k_history.clear()
        self.current_pre_q.clear()

    def store_pre_states(self, layer: int, q_pre: torch.Tensor, k_pre: torch.Tensor) -> None:
        if layer not in self.selected_layers:
            return
        if k_pre.shape[1] != q_pre.shape[1]:
            repeat_groups = q_pre.shape[1] // k_pre.shape[1]
            k_pre = k_pre.repeat_interleave(repeat_groups, dim=1)
        self.pre_k_history[layer].append(k_pre.detach().cpu())
        self.current_pre_q[layer] = q_pre.detach().cpu()

    def observe_post(
        self,
        *,
        layer: int,
        query_token: int,
        query_index: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        masked_scores: torch.Tensor,
    ) -> None:
        if layer not in self.selected_layers or query_token not in self.query_tokens:
            return
        finite = torch.isfinite(masked_scores[:, :, query_index, :])
        valid_count = int(finite[0, 0].sum().item())
        if valid_count <= 2:
            return
        history_count = valid_count - 1
        groups = build_groups(
            history_count=history_count,
            spans=self.spans,
            sink_tokens=self.sink_tokens,
            recent_tokens=self.recent_tokens,
            other_sample_tokens=self.other_sample_tokens,
        )
        post_q = query_states[0, :, query_index, :].detach().cpu().float()
        post_k = key_states[0, :, :history_count, :].detach().cpu().float()
        pre_q_chunk = self.current_pre_q.get(layer)
        pre_k_parts = self.pre_k_history.get(layer)
        if pre_q_chunk is None or not pre_k_parts:
            return
        pre_q = pre_q_chunk[0, :, query_index, :].float()
        pre_k = torch.cat(pre_k_parts, dim=2)[0, :, :history_count, :].float()
        self.observed_rows += 1

        for head in range(post_q.shape[0]):
            if head not in self.selected_heads:
                continue
            self.q_events[("post", layer, head)].append(post_q[head].clone())
            self.q_events[("pre", layer, head)].append(pre_q[head].clone())
            for group, indices in groups.items():
                idx = torch.tensor(indices, dtype=torch.long)
                for token_vector in post_k[head, idx, :]:
                    self.pair_events[("post", layer, head, group)].append((post_q[head].clone(), token_vector.clone()))
                for token_vector in pre_k[head, idx, :]:
                    self.pair_events[("pre", layer, head, group)].append((pre_q[head].clone(), token_vector.clone()))

    def summarize(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        layer_head_rows: list[dict[str, Any]] = []
        for (space, layer, head, group), pairs in sorted(self.pair_events.items(), key=lambda item: str(item[0])):
            q_matrix = torch.stack([q for q, _ in pairs], dim=0)
            k_matrix = torch.stack([k for _, k in pairs], dim=0)
            q_unique = torch.stack(self.q_events[(space, layer, head)], dim=0)
            row = summarize_pairs(space, layer, head, group, q_matrix, k_matrix, q_unique)
            layer_head_rows.append(row)

        aggregate: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in layer_head_rows:
            aggregate[(row["space"], row["group"])].append(row)
        overall_rows: list[dict[str, Any]] = []
        metric_fields = [
            "events",
            "q_mean_norm",
            "q_pc1_energy",
            "full_cos_mean",
            "full_cos_std",
            "residual_cos_mean",
            "residual_cos_std",
            "k_norm_mean",
            "q_norm_mean",
            "k_cos_qmean",
            "k_cos_qpc1",
            "k_common_energy_frac",
            "logit_mean",
            "logit_std",
        ]
        for (space, group), rows in sorted(aggregate.items()):
            out: dict[str, Any] = {"scope": "overall", "space": space, "group": group, "layer": "", "head": ""}
            weights = [float(row["events"]) for row in rows]
            total = sum(weights) or 1.0
            for field in metric_fields:
                if field == "events":
                    out[field] = int(sum(int(row[field]) for row in rows))
                else:
                    out[field] = sum(float(row[field]) * weight for row, weight in zip(rows, weights)) / total
            overall_rows.append(out)
        return layer_head_rows, overall_rows


def summarize_pairs(
    space: str,
    layer: int,
    head: int,
    group: str,
    q_matrix: torch.Tensor,
    k_matrix: torch.Tensor,
    q_unique: torch.Tensor,
) -> dict[str, Any]:
    q_unit = safe_unit(q_matrix)
    k_unit = safe_unit(k_matrix)
    full_cos = (q_unit * k_unit).sum(dim=-1)
    q_norm = torch.linalg.vector_norm(q_matrix.float(), dim=-1)
    k_norm = torch.linalg.vector_norm(k_matrix.float(), dim=-1)
    logit = (q_matrix.float() * k_matrix.float()).sum(dim=-1) / math.sqrt(q_matrix.shape[-1])

    q_unique_unit = safe_unit(q_unique)
    q_mean = q_unique_unit.mean(dim=0)
    q_mean_norm = float(torch.linalg.vector_norm(q_mean).item())
    q_mean_unit = safe_unit(q_mean.view(1, -1)).flatten()

    centered_q = q_unique_unit - q_unique_unit.mean(dim=0, keepdim=True)
    if centered_q.shape[0] >= 2:
        try:
            _, singular_values, vh = torch.linalg.svd(centered_q, full_matrices=False)
            energy = singular_values.square()
            q_pc1_energy = float((energy[0] / energy.sum()).item()) if float(energy.sum().item()) > 0.0 else 0.0
            q_pc1 = vh[0]
        except RuntimeError:
            q_pc1_energy = 0.0
            q_pc1 = q_mean_unit
    else:
        q_pc1_energy = 0.0
        q_pc1 = q_mean_unit

    q_res = q_unit - (q_unit @ q_mean_unit).unsqueeze(-1) * q_mean_unit
    k_res = k_unit - (k_unit @ q_mean_unit).unsqueeze(-1) * q_mean_unit
    residual_cos = (safe_unit(q_res) * safe_unit(k_res)).sum(dim=-1)
    k_cos_qmean = k_unit @ q_mean_unit
    k_cos_qpc1 = k_unit @ safe_unit(q_pc1.view(1, -1)).flatten()
    k_common_energy_frac = k_cos_qmean.square()

    return {
        "scope": "layer_head",
        "space": space,
        "group": group,
        "layer": layer,
        "head": head,
        "events": int(q_matrix.shape[0]),
        "q_mean_norm": q_mean_norm,
        "q_pc1_energy": q_pc1_energy,
        "full_cos_mean": float(full_cos.mean().item()),
        "full_cos_std": float(full_cos.std(unbiased=False).item()),
        "residual_cos_mean": float(residual_cos.mean().item()),
        "residual_cos_std": float(residual_cos.std(unbiased=False).item()),
        "k_norm_mean": float(k_norm.mean().item()),
        "q_norm_mean": float(q_norm.mean().item()),
        "k_cos_qmean": float(k_cos_qmean.mean().item()),
        "k_cos_qpc1": float(k_cos_qpc1.mean().item()),
        "k_common_energy_frac": float(k_common_energy_frac.mean().item()),
        "logit_mean": float(logit.mean().item()),
        "logit_std": float(logit.std(unbiased=False).item()),
    }


def _sink_structure_eager_attention_forward(
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
            _ACTIVE_COLLECTOR.observe_post(
                layer=layer,
                query_token=chunk_query_start + query_index,
                query_index=query_index,
                query_states=query_states,
                key_states=key_states,
                masked_scores=scores,
            )

    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def install_qwen3_attention_patch() -> None:
    global _ORIGINAL_QWEN3_ATTENTION_FORWARD, _ORIGINAL_EAGER_ATTENTION_FORWARD
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3.") from exc
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", _sink_structure_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _sink_structure_eager_attention_forward
    if _ORIGINAL_QWEN3_ATTENTION_FORWARD is None:
        _ORIGINAL_QWEN3_ATTENTION_FORWARD = modeling_qwen3.Qwen3Attention.forward

        def patched_forward(
            self: torch.nn.Module,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_value: Any = None,
            cache_position: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, Any]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            if _ACTIVE_COLLECTOR is not None:
                _ACTIVE_COLLECTOR.store_pre_states(int(getattr(self, "layer_idx", 0)), query_states, key_states)
            cos, sin = position_embeddings
            query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            attention_interface = modeling_qwen3.eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = modeling_qwen3.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        modeling_qwen3.Qwen3Attention.forward = patched_forward


@contextmanager
def active_collector(collector: SinkVectorCollector):
    global _ACTIVE_COLLECTOR
    previous = _ACTIVE_COLLECTOR
    _ACTIVE_COLLECTOR = collector
    try:
        yield
    finally:
        _ACTIVE_COLLECTOR = previous


@torch.inference_mode()
def run_sequence(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> None:
    past_key_values = None
    total_tokens = prefill_tokens + eval_tokens
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
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


def main() -> None:
    args = parse_args()
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
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    selected_layers = parse_index_spec(args.layers, layer_count, "layers")
    selected_heads = parse_index_spec(args.heads, head_count, "heads")

    collector = SinkVectorCollector(
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
        other_sample_tokens=args.other_sample_tokens,
    )

    rng = random.Random(args.seed)
    started = time.perf_counter()
    task_rows: list[dict[str, Any]] = []
    with active_collector(collector):
        for variant in variants:
            tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(args.tasks_per_variant)]
            for task_number, task in enumerate(tasks, start=1):
                if task_number == 1 or task_number == len(tasks) or task_number % args.log_every == 0:
                    print(f"{variant} task {task_number}/{len(tasks)}", flush=True)
                context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
                query_ids = tokenizer(task["query"], return_tensors="pt", add_special_tokens=False)["input_ids"]
                input_ids = torch.cat([context_ids, query_ids], dim=1)
                prefill_tokens = int(context_ids.shape[-1])
                eval_tokens = int(query_ids.shape[-1])
                query_tokens = set(choose_query_tokens(prefill_tokens, eval_tokens, args.max_query_tokens_per_task))
                spans = evidence_spans(tokenizer, task)
                collector.set_task(query_tokens=query_tokens, spans=spans)
                run_sequence(model, input_ids, prefill_tokens, eval_tokens, args.chunk_size, input_device)
                task_rows.append(
                    {
                        "variant": variant,
                        "task_id": task["task_id"],
                        "prefill_tokens": prefill_tokens,
                        "query_tokens": eval_tokens,
                        "sampled_query_tokens": " ".join(str(token) for token in sorted(query_tokens)),
                    }
                )

    layer_head_rows, overall_rows = collector.summarize()
    fields = [
        "scope",
        "space",
        "group",
        "layer",
        "head",
        "events",
        "q_mean_norm",
        "q_pc1_energy",
        "full_cos_mean",
        "full_cos_std",
        "residual_cos_mean",
        "residual_cos_std",
        "k_norm_mean",
        "q_norm_mean",
        "k_cos_qmean",
        "k_cos_qpc1",
        "k_common_energy_frac",
        "logit_mean",
        "logit_std",
    ]
    write_csv(output_dir / "sink_vector_structure_overall.csv", overall_rows, fields)
    write_csv(output_dir / "sink_vector_structure_layer_head.csv", layer_head_rows, fields)
    write_csv(
        output_dir / "tasks.csv",
        task_rows,
        ["variant", "task_id", "prefill_tokens", "query_tokens", "sampled_query_tokens"],
    )
    summary = {
        "seconds": time.perf_counter() - started,
        "resolved": {
            "layer_count": layer_count,
            "head_count": head_count,
            "head_dim": head_dim,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "tasks": len(task_rows),
            "observed_rows": collector.observed_rows,
            "groups": sorted({key[3] for key in collector.pair_events}),
        },
        "outputs": {
            "overall": str(output_dir / "sink_vector_structure_overall.csv"),
            "layer_head": str(output_dir / "sink_vector_structure_layer_head.csv"),
            "tasks": str(output_dir / "tasks.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
