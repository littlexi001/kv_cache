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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
_ACTIVE_COLLECTOR: "SinkVizCollector | None" = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot q/k PCA and sink cosine heatmaps for one layer/head.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="compact_kv,json_kv,needle_sentence,topic_table")
    parser.add_argument("--tasks_per_variant", type=int, default=4)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026070102)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--head", type=int, default=10)
    parser.add_argument("--sink_tokens", type=int, default=16)
    parser.add_argument("--recent_tokens", type=int, default=16)
    parser.add_argument("--other_sample_tokens", type=int, default=32)
    parser.add_argument("--max_query_tokens_per_task", type=int, default=2)
    parser.add_argument("--max_columns_per_group", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


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


class SinkVizCollector:
    def __init__(
        self,
        *,
        layer: int,
        head: int,
        sink_tokens: int,
        recent_tokens: int,
        other_sample_tokens: int,
    ) -> None:
        self.layer = layer
        self.head = head
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.other_sample_tokens = other_sample_tokens
        self.query_tokens: set[int] = set()
        self.spans: dict[str, tuple[int, int]] = {}
        self.queries: list[torch.Tensor] = []
        self.query_labels: list[str] = []
        self.keys: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.pre_k_history: dict[int, list[torch.Tensor]] = defaultdict(list)

    def set_task(self, *, query_tokens: set[int], spans: dict[str, tuple[int, int]], task_label: str) -> None:
        self.query_tokens = query_tokens
        self.spans = spans
        self.task_label = task_label
        self.pre_k_history.clear()

    def store_pre_states(self, layer: int, q_pre: torch.Tensor, k_pre: torch.Tensor) -> None:
        if layer != self.layer:
            return
        if k_pre.shape[1] != q_pre.shape[1]:
            repeat_groups = q_pre.shape[1] // k_pre.shape[1]
            k_pre = k_pre.repeat_interleave(repeat_groups, dim=1)
        self.pre_k_history[layer].append(k_pre.detach().cpu())

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
        if layer != self.layer or query_token not in self.query_tokens:
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
        q = query_states[0, self.head, query_index, :].detach().cpu().float()
        k = key_states[0, self.head, :history_count, :].detach().cpu().float()
        self.queries.append(q)
        self.query_labels.append(f"{self.task_label}:q{query_token}")
        for group, indices in groups.items():
            idx = torch.tensor(indices, dtype=torch.long)
            for vector in k[idx, :]:
                self.keys[group].append(vector.clone())


def _viz_eager_attention_forward(
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
        setattr(modeling_qwen3, "eager_attention_forward", _viz_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _viz_eager_attention_forward
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
def active_collector(collector: SinkVizCollector):
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
    chunk_size: int,
    input_device: torch.device,
) -> None:
    past_key_values = None
    total_tokens = input_ids.shape[-1]
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


def sample_group_vectors(
    keys: dict[str, list[torch.Tensor]],
    max_columns_per_group: int,
    rng: random.Random,
) -> tuple[torch.Tensor, list[str], list[str]]:
    matrices = []
    labels = []
    groups = []
    order = ["sink_first2", "sink_rest", "recent", "evidence_key", "evidence_label", "other_sample"]
    for group in order:
        vectors = keys.get(group, [])
        if not vectors:
            continue
        indices = list(range(len(vectors)))
        if len(indices) > max_columns_per_group:
            indices = sorted(rng.sample(indices, max_columns_per_group))
        for idx in indices:
            matrices.append(vectors[idx])
            labels.append(f"{group}:{idx}")
            groups.append(group)
    return torch.stack(matrices, dim=0), labels, groups


def pca_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    centered = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def plot_pca(q: torch.Tensor, k: torch.Tensor, groups: list[str], output: Path, layer: int, head: int) -> None:
    all_vec = safe_unit(torch.cat([q, k], dim=0))
    xy = pca_2d(all_vec)
    q_xy = xy[: q.shape[0]]
    k_xy = xy[q.shape[0] :]
    colors = {
        "sink_first2": "#d62728",
        "sink_rest": "#ff7f0e",
        "recent": "#1f77b4",
        "evidence_key": "#9467bd",
        "evidence_label": "#2ca02c",
        "other_sample": "#7f7f7f",
    }
    plt.figure(figsize=(8, 6), dpi=160)
    plt.scatter(q_xy[:, 0], q_xy[:, 1], s=46, c="#111111", marker="x", label="query q")
    for group in sorted(set(groups)):
        idx = [i for i, value in enumerate(groups) if value == group]
        pts = k_xy[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.72, c=colors.get(group, "#999999"), label=group)
    plt.axhline(0, color="#cccccc", linewidth=0.6)
    plt.axvline(0, color="#cccccc", linewidth=0.6)
    plt.title(f"Q/K PCA projection, layer {layer}, head {head}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_heatmap(matrix: torch.Tensor, groups: list[str], output: Path, title: str) -> None:
    data = matrix.detach().cpu().numpy()
    plt.figure(figsize=(12, 6), dpi=160)
    image = plt.imshow(data, aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=0.5, interpolation="nearest")
    plt.colorbar(image, fraction=0.025, pad=0.02, label="cosine")
    boundaries = []
    last = None
    centers = []
    labels = []
    start = 0
    for idx, group in enumerate(groups):
        if last is None:
            last = group
            start = idx
        elif group != last:
            boundaries.append(idx - 0.5)
            centers.append((start + idx - 1) / 2)
            labels.append(last)
            start = idx
            last = group
    if last is not None:
        centers.append((start + len(groups) - 1) / 2)
        labels.append(last)
    for boundary in boundaries:
        plt.axvline(boundary, color="black", linewidth=0.8)
    plt.xticks(centers, labels, rotation=25, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    plt.xlabel("sampled K tokens grouped by type")
    plt.ylabel("sampled query vectors")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    rng = random.Random(args.seed)
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

    collector = SinkVizCollector(
        layer=args.layer,
        head=args.head,
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
        other_sample_tokens=args.other_sample_tokens,
    )
    started = time.perf_counter()
    with active_collector(collector):
        for variant in variants:
            tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(args.tasks_per_variant)]
            for task_number, task in enumerate(tasks, start=1):
                if task_number == 1 or task_number == len(tasks) or task_number % args.log_every == 0:
                    print(f"{variant} task {task_number}/{len(tasks)}", flush=True)
                context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
                query_ids = tokenizer(task["query"], return_tensors="pt", add_special_tokens=False)["input_ids"]
                prefill_tokens = int(context_ids.shape[-1])
                eval_tokens = int(query_ids.shape[-1])
                query_tokens = set(choose_query_tokens(prefill_tokens, eval_tokens, args.max_query_tokens_per_task))
                collector.set_task(
                    query_tokens=query_tokens,
                    spans=evidence_spans(tokenizer, task),
                    task_label=f"{variant}{task['task_id']}",
                )
                run_sequence(model, torch.cat([context_ids, query_ids], dim=1), args.chunk_size, input_device)

    q = torch.stack(collector.queries, dim=0).float()
    k, labels, groups = sample_group_vectors(collector.keys, args.max_columns_per_group, rng)
    q_unit = safe_unit(q)
    k_unit = safe_unit(k)
    cosine = q_unit @ k_unit.T
    q_mean = safe_unit(q_unit.mean(dim=0, keepdim=True)).flatten()
    q_res = safe_unit(q_unit - (q_unit @ q_mean).unsqueeze(-1) * q_mean)
    k_res = safe_unit(k_unit - (k_unit @ q_mean).unsqueeze(-1) * q_mean)
    residual_cosine = q_res @ k_res.T

    plot_pca(q, k, groups, output_dir / "qk_pca.png", args.layer, args.head)
    plot_heatmap(cosine, groups, output_dir / "qk_cosine_heatmap.png", f"Raw q-k cosine, layer {args.layer}, head {args.head}")
    plot_heatmap(
        residual_cosine,
        groups,
        output_dir / "qk_residual_cosine_heatmap.png",
        f"Residual q-k cosine after removing q_mean, layer {args.layer}, head {args.head}",
    )

    rows = []
    for group in sorted(set(groups)):
        idx = [i for i, value in enumerate(groups) if value == group]
        raw_values = cosine[:, idx].flatten()
        residual_values = residual_cosine[:, idx].flatten()
        rows.append(
            {
                "group": group,
                "columns": len(idx),
                "raw_cos_mean": float(raw_values.mean().item()),
                "raw_cos_std": float(raw_values.std(unbiased=False).item()),
                "residual_cos_mean": float(residual_values.mean().item()),
                "residual_cos_std": float(residual_values.std(unbiased=False).item()),
            }
        )
    write_csv(
        output_dir / "heatmap_group_summary.csv",
        rows,
        ["group", "columns", "raw_cos_mean", "raw_cos_std", "residual_cos_mean", "residual_cos_std"],
    )
    write_csv(output_dir / "columns.csv", [{"column": i, "label": label, "group": group} for i, (label, group) in enumerate(zip(labels, groups))], ["column", "label", "group"])

    summary = {
        "seconds": time.perf_counter() - started,
        "layer": args.layer,
        "head": args.head,
        "query_count": int(q.shape[0]),
        "key_count": int(k.shape[0]),
        "groups": {group: groups.count(group) for group in sorted(set(groups))},
        "outputs": {
            "pca": str(output_dir / "qk_pca.png"),
            "cosine_heatmap": str(output_dir / "qk_cosine_heatmap.png"),
            "residual_heatmap": str(output_dir / "qk_residual_cosine_heatmap.png"),
            "group_summary": str(output_dir / "heatmap_group_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
