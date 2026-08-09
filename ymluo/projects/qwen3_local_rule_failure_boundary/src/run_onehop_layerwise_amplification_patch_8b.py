from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

import run_incremental_nine_newline_boundary_8b as incremental
import run_incremental_twohop_first_token_8b as first_token
import run_local_rule_failure_boundary as base
import run_overnight_onehop_q_rope_probe_8b as qprobe
import run_twohop_age_distractor_failure_boundary_8b as twohop


SOURCE_TOTAL = 140 * 1024 + 64
TARGET_TOTAL = SOURCE_TOTAL + 64
TRACE_STAGES = (
    "residual_in",
    "attn_norm",
    "attn_out",
    "post_attn_residual",
    "mlp_norm",
    "mlp_out",
    "residual_out",
    "q_pre",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace and causally patch the one-hop Query/QK collapse "
            "between the adjacent 64-token probe points "
            "143,424 -> 143,488."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--critical-heads-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--original-max-position-embeddings",
        type=int,
        default=40960,
    )
    parser.add_argument("--fixed-rope-factor", type=float, default=4.0)
    parser.add_argument(
        "--fixed-max-position-embeddings",
        type=int,
        default=144 * 1024,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def rounded(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def clear_allocator() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(f"cannot extract tensor from {type(output)!r}")


def replace_output_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple) and output:
        return (replacement, *output[1:])
    raise TypeError(f"cannot replace tensor in {type(output)!r}")


class TraceCollector:
    def __init__(self, model: Any) -> None:
        self.enabled = False
        self.values: dict[int, dict[str, torch.Tensor]] = {}
        self.handles: list[Any] = []
        for layer_index, layer in enumerate(model.model.layers):
            self.handles.extend(
                [
                    layer.register_forward_pre_hook(
                        self._pre_hook(layer_index, "residual_in")
                    ),
                    layer.input_layernorm.register_forward_hook(
                        self._output_hook(layer_index, "attn_norm")
                    ),
                    layer.self_attn.register_forward_hook(
                        self._output_hook(layer_index, "attn_out")
                    ),
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        self._pre_hook(
                            layer_index,
                            "post_attn_residual",
                        )
                    ),
                    layer.post_attention_layernorm.register_forward_hook(
                        self._output_hook(layer_index, "mlp_norm")
                    ),
                    layer.mlp.register_forward_hook(
                        self._output_hook(layer_index, "mlp_out")
                    ),
                    layer.register_forward_hook(
                        self._output_hook(layer_index, "residual_out")
                    ),
                    layer.self_attn.q_norm.register_forward_hook(
                        self._q_hook(layer_index)
                    ),
                ]
            )

    def _store(
        self,
        layer_index: int,
        stage: str,
        tensor: torch.Tensor,
        q_tensor: bool = False,
    ) -> None:
        if not self.enabled:
            return
        if q_tensor:
            value = tensor[0, -1].detach().to(
                dtype=torch.float32,
                device="cpu",
            )
        else:
            value = tensor[0].detach().to(
                dtype=torch.bfloat16,
                device="cpu",
            )
        self.values.setdefault(layer_index, {})[stage] = value

    def _pre_hook(self, layer_index: int, stage: str) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                raise RuntimeError(f"{stage} hook has no arguments")
            self._store(layer_index, stage, args[0])

        return hook

    def _output_hook(
        self,
        layer_index: int,
        stage: str,
    ) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            self._store(
                layer_index,
                stage,
                tensor_from_output(output),
            )

        return hook

    def _q_hook(self, layer_index: int) -> Callable[..., None]:
        def hook(
            module: Any,
            args: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            self._store(
                layer_index,
                "q_pre",
                output,
                q_tensor=True,
            )

        return hook

    def start(self) -> None:
        self.values = {}
        self.enabled = True

    def stop(self) -> dict[int, dict[str, torch.Tensor]]:
        self.enabled = False
        expected = len(self.handles) // len(TRACE_STAGES)
        if len(self.values) != expected:
            raise RuntimeError(
                f"captured {len(self.values)} layers, expected {expected}"
            )
        for layer_index, values in self.values.items():
            missing = set(TRACE_STAGES) - set(values)
            if missing:
                raise RuntimeError(
                    f"layer {layer_index} missing stages {sorted(missing)}"
                )
        return self.values

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class QCollector:
    def __init__(
        self,
        model: Any,
        pairs: Sequence[tuple[int, int]],
    ) -> None:
        self.by_layer: dict[int, list[tuple[int, int]]] = {}
        for pair_index, (layer, head) in enumerate(pairs):
            self.by_layer.setdefault(layer, []).append(
                (pair_index, head)
            )
        self.values: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        for layer_index in self.by_layer:
            layer = model.model.layers[layer_index]
            self.handles.append(
                layer.self_attn.q_norm.register_forward_hook(
                    self._hook(layer_index)
                )
            )

    def _hook(self, layer_index: int) -> Callable[..., None]:
        def hook(
            module: Any,
            args: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            last = output[0, -1].detach().float().cpu()
            for pair_index, head in self.by_layer[layer_index]:
                self.values[pair_index] = last[head]

        return hook

    def tensor(self, pair_count: int) -> torch.Tensor:
        if len(self.values) != pair_count:
            raise RuntimeError(
                f"captured {len(self.values)} Q vectors, "
                f"expected {pair_count}"
            )
        return torch.stack(
            [self.values[index] for index in range(pair_count)],
            dim=0,
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def forward_body_tokens(
    model: Any,
    token_ids: Sequence[int],
    cache: Any,
    past_length: int,
    attention_mask_buffer: torch.Tensor,
    chunk_size: int,
) -> tuple[Any, int]:
    for start in range(0, len(token_ids), chunk_size):
        chunk = token_ids[start : start + chunk_size]
        with torch.inference_mode():
            output = incremental.forward_with_shared_mask(
                model,
                torch.tensor([chunk], dtype=torch.long),
                cache,
                past_length,
                attention_mask_buffer,
                with_logits=False,
            )
        cache = output.past_key_values
        past_length += len(chunk)
        del output
    return cache, past_length


def score_qk(
    model: Any,
    query_pre: torch.Tensor,
    gold_keys: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    weights: torch.Tensor,
    query_position: int,
) -> tuple[float, list[float]]:
    scores = qprobe.critical_scores(
        model,
        query_pre,
        gold_keys,
        pairs,
        query_position,
    )
    weighted = sum(
        float(weight) * score
        for weight, score in zip(weights.tolist(), scores)
    )
    return float(weighted), scores


def fixed_margin(
    logits: torch.Tensor,
    gold_token_id: int,
    competitor_token_id: int,
) -> float:
    return float(
        logits[gold_token_id].float().item()
        - logits[competitor_token_id].float().item()
    )


def selected_attention_rows(
    model: Any,
    tokenizer: Any,
    cache_with_query: Any,
    trace: dict[int, dict[str, torch.Tensor]],
    pairs: Sequence[tuple[int, int]],
    weights: torch.Tensor,
    query_position: int,
    gold_position: int,
    token_ids: Sequence[int],
    categories: Sequence[str],
    condition: str,
    total_tokens: int,
    chunk_size: int = 8192,
) -> list[dict[str, Any]]:
    legacy = base.legacy_cache(cache_with_query)
    if len(token_ids) != len(categories):
        raise ValueError("token/category lengths differ")
    category_names = (
        "gold_other",
        "gold_age",
        "distractor_other",
        "distractor_ages",
        "irrelevant_periods",
        "query",
    )
    category_masks = {
        name: torch.tensor(
            [value == name for value in categories],
            dtype=torch.bool,
        )
        for name in category_names
    }
    rows = []
    num_heads = int(model.config.num_attention_heads)
    for index, (layer_index, head) in enumerate(pairs):
        key = legacy[layer_index][0][0]
        kv_heads = int(key.shape[0])
        groups = max(1, num_heads // kv_heads)
        kv_head = min(kv_heads - 1, head // groups)
        query_pre = trace[layer_index]["q_pre"][head].view(1, -1)
        query_post = incremental.apply_rope_at_position(
            model,
            query_pre.to(key.device),
            query_position,
        )[0]
        scale = float(
            getattr(
                model.model.layers[layer_index].self_attn,
                "scaling",
                query_post.shape[-1] ** -0.5,
            )
        )
        chunks = []
        selected_key = key[kv_head]
        for start in range(0, selected_key.shape[0], chunk_size):
            part = selected_key[start : start + chunk_size]
            chunks.append(
                (
                    torch.mv(
                        part.float(),
                        query_post.float(),
                    )
                    * scale
                ).detach().cpu()
            )
        logits = torch.cat(chunks, dim=0)
        probabilities = torch.softmax(logits, dim=0)
        top_position = int(torch.argmax(logits).item())
        row: dict[str, Any] = {
            "condition": condition,
            "total_tokens": total_tokens,
            "layer": layer_index,
            "head": head,
            "weight": rounded(weights[index]),
            "gold_qk": rounded(logits[gold_position]),
            "gold_attention": rounded(probabilities[gold_position]),
            "top_position": top_position,
            "top_token_id": int(token_ids[top_position]),
            "top_token_label": first_token.token_label(
                first_token.token_text(
                    tokenizer,
                    int(token_ids[top_position]),
                )
            ),
            "top_attention": rounded(probabilities[top_position]),
            "attention_entropy": rounded(
                -torch.sum(
                    probabilities
                    * torch.log(probabilities.clamp_min(1e-30))
                )
            ),
        }
        captured_mass = 0.0
        for name, mask in category_masks.items():
            mass = float(probabilities[mask].sum().item())
            row[f"{name}_attention"] = rounded(mass)
            captured_mass += mass
        row["other_attention"] = rounded(1.0 - captured_mass)
        rows.append(row)
        del logits, probabilities, chunks
    return rows


def run_traced_query(
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    cache: Any,
    body_length: int,
    attention_mask_buffer: torch.Tensor,
    collector: TraceCollector,
    answer_variants: dict[str, list[int]],
    pairs: Sequence[tuple[int, int]],
    weights: torch.Tensor,
    gold_position: int,
    token_ids: Sequence[int],
    categories: Sequence[str],
    condition: str,
    total_tokens: int,
) -> tuple[
    Any,
    dict[int, dict[str, torch.Tensor]],
    dict[str, Any],
    torch.Tensor,
    list[dict[str, Any]],
]:
    clear_allocator()
    collector.start()
    with torch.inference_mode():
        output = incremental.forward_with_shared_mask(
            model,
            query_ids,
            cache,
            body_length,
            attention_mask_buffer,
            with_logits=True,
        )
    trace = collector.stop()
    logits = output.logits[0, -1].detach().float().cpu()
    score = first_token.score_first_token(
        tokenizer,
        logits,
        answer_variants,
    )
    attention_rows = selected_attention_rows(
        model,
        tokenizer,
        output.past_key_values,
        trace,
        pairs,
        weights,
        body_length + query_ids.shape[1] - 1,
        gold_position,
        token_ids,
        categories,
        condition,
        total_tokens,
    )
    cache = incremental.crop_cache(
        output.past_key_values,
        body_length,
    )
    del output
    clear_allocator()
    return cache, trace, score, logits, attention_rows


def patch_handle(
    model: Any,
    kind: str,
    layer_index: int,
    source: torch.Tensor,
) -> Any:
    layer = model.model.layers[layer_index]

    if kind == "residual_in":
        def hook(
            module: Any,
            args: tuple[Any, ...],
        ) -> tuple[Any, ...]:
            hidden = args[0]
            replacement = source.to(
                device=hidden.device,
                dtype=hidden.dtype,
            ).unsqueeze(0)
            return (replacement, *args[1:])

        return layer.register_forward_pre_hook(hook)

    module = layer.self_attn if kind == "attn_out" else layer.mlp

    def output_hook(
        hooked_module: Any,
        args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        original = tensor_from_output(output)
        replacement = source.to(
            device=original.device,
            dtype=original.dtype,
        ).unsqueeze(0)
        return replace_output_tensor(output, replacement)

    return module.register_forward_hook(output_hook)


def run_patch_query(
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    cache: Any,
    body_length: int,
    attention_mask_buffer: torch.Tensor,
    answer_variants: dict[str, list[int]],
    pairs: Sequence[tuple[int, int]],
    weights: torch.Tensor,
    gold_keys: torch.Tensor,
    kind: str,
    layer_index: int,
    source: torch.Tensor,
    gold_token_id: int,
    fixed_competitor_id: int,
) -> tuple[Any, dict[str, Any]]:
    clear_allocator()
    q_collector = QCollector(model, pairs)
    patch = patch_handle(model, kind, layer_index, source)
    try:
        with torch.inference_mode():
            output = incremental.forward_with_shared_mask(
                model,
                query_ids,
                cache,
                body_length,
                attention_mask_buffer,
                with_logits=True,
            )
        logits = output.logits[0, -1].detach().float().cpu()
        score = first_token.score_first_token(
            tokenizer,
            logits,
            answer_variants,
        )
        query_pre = q_collector.tensor(len(pairs))
        qk, _ = score_qk(
            model,
            query_pre,
            gold_keys,
            pairs,
            weights,
            body_length + query_ids.shape[1] - 1,
        )
        result = {
            "patch_kind": kind,
            "layer": layer_index,
            "critical_qk": rounded(qk),
            "gold_probability": score["gold_exact_probability"],
            "gold_vs_fixed_competitor_margin": rounded(
                fixed_margin(
                    logits,
                    gold_token_id,
                    fixed_competitor_id,
                )
            ),
            "gold_vs_strongest_margin": score[
                "gold_exact_vs_competitor_margin"
            ],
            "top_token_id": score["top_token_id"],
            "top_token_label": score["top_token_label"],
            "strongest_competitor_token_id": score[
                "strongest_competitor_token_id"
            ],
            "strongest_competitor_token_label": score[
                "strongest_competitor_token_label"
            ],
        }
        cache = incremental.crop_cache(
            output.past_key_values,
            body_length,
        )
        del output, logits, query_pre
        clear_allocator()
        return cache, result
    finally:
        patch.remove()
        q_collector.close()


def vector_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    use_last_token: bool,
) -> dict[str, float]:
    source_f = (
        source[-1] if use_last_token else source
    ).float().reshape(-1)
    target_f = (
        target[-1] if use_last_token else target
    ).float().reshape(-1)
    delta = target_f - source_f
    source_norm = float(source_f.norm().item())
    target_norm = float(target_f.norm().item())
    delta_norm = float(delta.norm().item())
    cosine = float(
        F.cosine_similarity(source_f, target_f, dim=0).item()
    )
    return {
        "source_norm": source_norm,
        "target_norm": target_norm,
        "delta_norm": delta_norm,
        "relative_delta": delta_norm / max(source_norm, 1e-12),
        "cosine": cosine,
    }


def build_trace_rows(
    source: dict[int, dict[str, torch.Tensor]],
    target: dict[int, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_residual_delta = 0.0
    for layer_index in sorted(source):
        layer_metrics: dict[str, dict[str, float]] = {}
        for stage in TRACE_STAGES:
            layer_metrics[stage] = vector_metrics(
                source[layer_index][stage],
                target[layer_index][stage],
                use_last_token=stage != "q_pre",
            )
            row: dict[str, Any] = {
                "layer": layer_index,
                "stage": stage,
            }
            row.update(
                {
                    key: rounded(value)
                    for key, value in layer_metrics[stage].items()
                }
            )
            rows.append(row)

        delta_in = (
            target[layer_index]["residual_in"][-1].float()
            - source[layer_index]["residual_in"][-1].float()
        )
        delta_attn = (
            target[layer_index]["attn_out"][-1].float()
            - source[layer_index]["attn_out"][-1].float()
        )
        delta_mlp = (
            target[layer_index]["mlp_out"][-1].float()
            - source[layer_index]["mlp_out"][-1].float()
        )
        delta_out = (
            target[layer_index]["residual_out"][-1].float()
            - source[layer_index]["residual_out"][-1].float()
        )
        reconstruction = delta_in + delta_attn + delta_mlp
        error = float(
            (delta_out - reconstruction).norm().item()
            / max(float(delta_out.norm().item()), 1e-12)
        )
        residual_delta = float(delta_out.norm().item())
        amplification = (
            residual_delta / previous_residual_delta
            if previous_residual_delta > 0
            else math.nan
        )
        for row in rows[-len(TRACE_STAGES):]:
            row["layer_residual_delta_out"] = rounded(
                residual_delta
            )
            row["layer_residual_amplification"] = (
                rounded(amplification)
                if math.isfinite(amplification)
                else ""
            )
            row["residual_reconstruction_relative_error"] = rounded(
                error
            )
        previous_residual_delta = residual_delta
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    answer_variants = twohop.validate_answer_variants(tokenizer)
    case, query_ids_list, base_body, continuation, categories = (
        qprobe.prepare_case(tokenizer)
    )
    _, weight_map, _ = incremental.load_critical_heads(
        args.critical_heads_csv
    )
    pairs = sorted(weight_map)
    weights = torch.tensor(
        [weight_map[pair] for pair in pairs],
        dtype=torch.float64,
    )
    source_added = SOURCE_TOTAL - qprobe.START_TOTAL
    target_added = TARGET_TOTAL - qprobe.START_TOTAL
    if len(continuation) < target_added:
        raise RuntimeError("continuation is shorter than target")

    if args.dry_run:
        write_json(
            output_dir / "dry_run.json",
            {
                "source_total": SOURCE_TOTAL,
                "target_total": TARGET_TOTAL,
                "base_total": qprobe.START_TOTAL,
                "query_tokens": len(query_ids_list),
                "source_added": source_added,
                "target_added": target_added,
                "critical_head_count": len(pairs),
                "patch_forward_count": 3 * 36,
            },
        )
        return

    model, model_tokenizer = base.load_model_and_tokenizer(
        args,
        args.fixed_max_position_embeddings,
        args.fixed_rope_factor,
    )
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("tokenizer changed while loading model")
    tokenizer = model_tokenizer
    query_ids = torch.tensor(
        query_ids_list,
        dtype=torch.long,
    ).view(1, -1)
    attention_mask_buffer = torch.ones(
        (1, args.fixed_max_position_embeddings),
        dtype=torch.long,
        device=base.input_device(model),
    )
    trace_collector = TraceCollector(model)
    started = time.perf_counter()

    try:
        cache, body_length = forward_body_tokens(
            model,
            base_body,
            None,
            0,
            attention_mask_buffer,
            args.prefill_chunk_size,
        )
        cache, body_length = forward_body_tokens(
            model,
            continuation[:source_added],
            cache,
            body_length,
            attention_mask_buffer,
            1,
        )
        gold_key_map = incremental.extract_gold_keys(
            cache,
            model,
            pairs,
            case["gold_age_span"][0],
        )
        gold_keys = torch.stack(
            [gold_key_map[pair] for pair in pairs],
            dim=0,
        )

        (
            cache,
            source_trace,
            source_score,
            source_logits,
            source_attention_rows,
        ) = (
            run_traced_query(
                model,
                tokenizer,
                query_ids,
                cache,
                body_length,
                attention_mask_buffer,
                trace_collector,
                answer_variants,
                pairs,
                weights,
                case["gold_age_span"][0],
                [
                    *base_body,
                    *continuation[:source_added],
                    *query_ids_list,
                ],
                [
                    *categories[
                        : len(base_body) + source_added
                    ],
                    *(["query"] * len(query_ids_list)),
                ],
                "source",
                SOURCE_TOTAL,
            )
        )
        source_q = torch.stack(
            [
                source_trace[layer]["q_pre"][head]
                for layer, head in pairs
            ],
            dim=0,
        )
        source_qk, source_head_qk = score_qk(
            model,
            source_q,
            gold_keys,
            pairs,
            weights,
            body_length + query_ids.shape[1] - 1,
        )

        cache, body_length = forward_body_tokens(
            model,
            continuation[source_added:target_added],
            cache,
            body_length,
            attention_mask_buffer,
            1,
        )
        (
            cache,
            target_trace,
            target_score,
            target_logits,
            target_attention_rows,
        ) = (
            run_traced_query(
                model,
                tokenizer,
                query_ids,
                cache,
                body_length,
                attention_mask_buffer,
                trace_collector,
                answer_variants,
                pairs,
                weights,
                case["gold_age_span"][0],
                [
                    *base_body,
                    *continuation[:target_added],
                    *query_ids_list,
                ],
                [
                    *categories[
                        : len(base_body) + target_added
                    ],
                    *(["query"] * len(query_ids_list)),
                ],
                "target",
                TARGET_TOTAL,
            )
        )
        target_q = torch.stack(
            [
                target_trace[layer]["q_pre"][head]
                for layer, head in pairs
            ],
            dim=0,
        )
        target_qk, target_head_qk = score_qk(
            model,
            target_q,
            gold_keys,
            pairs,
            weights,
            body_length + query_ids.shape[1] - 1,
        )

        gold_token_id = int(answer_variants[twohop.GOLD_ANSWER][0])
        fixed_competitor_id = int(
            target_score["strongest_competitor_token_id"]
        )
        source_fixed_margin = fixed_margin(
            source_logits,
            gold_token_id,
            fixed_competitor_id,
        )
        target_fixed_margin = fixed_margin(
            target_logits,
            gold_token_id,
            fixed_competitor_id,
        )

        trace_rows = build_trace_rows(source_trace, target_trace)
        write_csv(output_dir / "layer_stage_trace.csv", trace_rows)
        write_csv(
            output_dir / "critical_head_attention.csv",
            [*source_attention_rows, *target_attention_rows],
        )

        head_rows = []
        for index, (layer, head) in enumerate(pairs):
            source_pre = source_q[index].float()
            target_pre = target_q[index].float()
            head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "weight": rounded(weights[index]),
                    "query_cosine": rounded(
                        F.cosine_similarity(
                            source_pre,
                            target_pre,
                            dim=0,
                        ).item()
                    ),
                    "query_relative_l2": rounded(
                        (target_pre - source_pre).norm().item()
                        / max(source_pre.norm().item(), 1e-12)
                    ),
                    "source_qk": rounded(source_head_qk[index]),
                    "target_qk": rounded(target_head_qk[index]),
                    "weighted_qk_change": rounded(
                        weights[index]
                        * (
                            target_head_qk[index]
                            - source_head_qk[index]
                        )
                    ),
                }
            )
        write_csv(output_dir / "critical_head_changes.csv", head_rows)

        patch_rows: list[dict[str, Any]] = []
        for kind in ("residual_in", "attn_out", "mlp_out"):
            for layer_index in range(len(model.model.layers)):
                cache, row = run_patch_query(
                    model,
                    tokenizer,
                    query_ids,
                    cache,
                    body_length,
                    attention_mask_buffer,
                    answer_variants,
                    pairs,
                    weights,
                    gold_keys,
                    kind,
                    layer_index,
                    source_trace[layer_index][kind],
                    gold_token_id,
                    fixed_competitor_id,
                )
                row["qk_recovery_fraction"] = rounded(
                    (row["critical_qk"] - target_qk)
                    / max(source_qk - target_qk, 1e-12)
                )
                row["fixed_margin_recovery_fraction"] = rounded(
                    (
                        row["gold_vs_fixed_competitor_margin"]
                        - target_fixed_margin
                    )
                    / max(
                        source_fixed_margin - target_fixed_margin,
                        1e-12,
                    )
                )
                patch_rows.append(row)
                print(
                    json.dumps(
                        {
                            "kind": kind,
                            "layer": layer_index,
                            "qk": row["critical_qk"],
                            "qk_recovery": row[
                                "qk_recovery_fraction"
                            ],
                            "margin": row[
                                "gold_vs_fixed_competitor_margin"
                            ],
                            "top": row["top_token_label"],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
        write_csv(output_dir / "activation_patch.csv", patch_rows)

        baseline = {
            "source_total": SOURCE_TOTAL,
            "target_total": TARGET_TOTAL,
            "source": {
                "critical_qk": rounded(source_qk),
                "gold_probability": source_score[
                    "gold_exact_probability"
                ],
                "gold_vs_fixed_competitor_margin": rounded(
                    source_fixed_margin
                ),
                "top_token_label": source_score["top_token_label"],
            },
            "target": {
                "critical_qk": rounded(target_qk),
                "gold_probability": target_score[
                    "gold_exact_probability"
                ],
                "gold_vs_fixed_competitor_margin": rounded(
                    target_fixed_margin
                ),
                "top_token_label": target_score["top_token_label"],
            },
            "fixed_competitor_token_id": fixed_competitor_id,
            "fixed_competitor_token_label": target_score[
                "strongest_competitor_token_label"
            ],
        }
        write_json(output_dir / "baseline.json", baseline)
        torch.save(
            {
                "source_total": SOURCE_TOTAL,
                "target_total": TARGET_TOTAL,
                "source_trace": source_trace,
                "target_trace": target_trace,
                "critical_pairs": torch.tensor(pairs),
                "critical_weights": weights,
                "source_query_pre": source_q,
                "target_query_pre": target_q,
            },
            output_dir / "trace_vectors.pt",
        )
        write_json(
            output_dir / "manifest.json",
            {
                "schema_version": 1,
                "experiment": "onehop_layerwise_amplification_patch",
                "source_total": SOURCE_TOTAL,
                "target_total": TARGET_TOTAL,
                "query_tokens": int(query_ids.shape[1]),
                "critical_head_count": len(pairs),
                "layer_count": len(model.model.layers),
                "patch_kinds": [
                    "residual_in",
                    "attn_out",
                    "mlp_out",
                ],
                "rope_scaling": {
                    "type": "yarn",
                    "factor": args.fixed_rope_factor,
                    "original_max_position_embeddings": (
                        args.original_max_position_embeddings
                    ),
                    "max_position_embeddings": (
                        args.fixed_max_position_embeddings
                    ),
                },
                "elapsed_seconds": time.perf_counter() - started,
                "complete": True,
            },
        )
    finally:
        trace_collector.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
