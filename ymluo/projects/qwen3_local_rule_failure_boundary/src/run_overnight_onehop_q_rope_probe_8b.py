from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import run_age_distractor_failure_boundary_8b as boundary
import run_incremental_nine_newline_boundary_8b as incremental
import run_incremental_twohop_first_token_8b as first_token
import run_local_rule_failure_boundary as base
import run_twohop_age_distractor_failure_boundary_8b as twohop


START_TOTAL = 136 * 1024
END_TOTAL = 144 * 1024
MAX_DISTRACTORS = 4607
SAMPLE_STRIDE = 64
ANCHOR_TOTALS = (136 * 1024, 140 * 1024, 142 * 1024, 144 * 1024)
BAND_NAMES = ("high_0_15", "mid_16_31", "low_32_47", "ultralow_48_63")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shared-prefix Q-mean and RoPE decomposition probe for the "
            "one-hop 136K-to-144K boundary."
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
        default=END_TOTAL,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def rounded(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


class QAccumulator:
    def __init__(self, model: Any) -> None:
        self.mode = "idle"
        self.period_mask: torch.Tensor | None = None
        self.body_sums: dict[int, torch.Tensor] = {}
        self.period_sums: dict[int, torch.Tensor] = {}
        self.query: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_index, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.q_norm.register_forward_hook(
                    self._make_hook(layer_index)
                )
            )

    def _make_hook(self, layer_index: int):
        def hook(
            module: Any,
            hook_args: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            if output.ndim != 4:
                raise RuntimeError(
                    f"layer {layer_index}: unexpected Q shape "
                    f"{tuple(output.shape)}"
                )
            values = output[0].detach().float()
            if self.mode == "body":
                if layer_index not in self.body_sums:
                    self.body_sums[layer_index] = torch.zeros(
                        values.shape[1:],
                        dtype=torch.float32,
                        device=values.device,
                    )
                    self.period_sums[layer_index] = torch.zeros_like(
                        self.body_sums[layer_index]
                    )
                self.body_sums[layer_index].add_(
                    values.sum(dim=0)
                )
                if self.period_mask is not None:
                    mask = self.period_mask.to(values.device)
                    self.period_sums[layer_index].add_(
                        values[mask].sum(dim=0)
                    )
            elif self.mode == "query":
                self.query[layer_index] = (
                    values[-1].detach().cpu()
                )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def prepare_case(
    tokenizer: Any,
) -> tuple[dict[str, Any], list[int], list[int], list[int], list[str]]:
    distractor_pool = boundary.build_distractor_pool(tokenizer)
    case = boundary.build_case(
        tokenizer,
        distractor_pool,
        END_TOTAL,
        MAX_DISTRACTORS,
        filler_text=".",
    )
    query_start, query_end = case["query_span"]
    query_ids = case["prompt_ids"][query_start:query_end]
    query_length = len(query_ids)
    base_body_length = START_TOTAL - query_length
    max_body = case["prompt_ids"][:query_start]
    base_body = max_body[:base_body_length]
    continuation = max_body[base_body_length:]
    lookup = incremental.build_category_lookup(
        END_TOTAL,
        case["category_positions"],
    )
    categories = lookup[: len(max_body)]
    return case, query_ids, base_body, continuation, categories


def forward_body(
    model: Any,
    input_ids: torch.Tensor,
    cache: Any,
    past_length: int,
    attention_mask_buffer: torch.Tensor,
    accumulator: QAccumulator,
    categories: Sequence[str],
) -> Any:
    accumulator.mode = "body"
    accumulator.period_mask = torch.tensor(
        [
            category == "irrelevant_periods"
            for category in categories
        ],
        dtype=torch.bool,
    )
    try:
        with torch.inference_mode():
            return incremental.forward_with_shared_mask(
                model,
                input_ids,
                cache,
                past_length,
                attention_mask_buffer,
                with_logits=False,
            )
    finally:
        accumulator.mode = "idle"
        accumulator.period_mask = None


def prefill_with_q_sums(
    model: Any,
    body_ids: Sequence[int],
    categories: Sequence[str],
    chunk_size: int,
    attention_mask_buffer: torch.Tensor,
    accumulator: QAccumulator,
) -> tuple[Any, float, int, int]:
    cache = None
    past_length = 0
    period_count = 0
    base.synchronize()
    started = time.perf_counter()
    for start in range(0, len(body_ids), chunk_size):
        chunk_ids = body_ids[start : start + chunk_size]
        chunk_categories = categories[start : start + len(chunk_ids)]
        output = forward_body(
            model,
            torch.tensor(
                [chunk_ids],
                dtype=torch.long,
            ),
            cache,
            past_length,
            attention_mask_buffer,
            accumulator,
            chunk_categories,
        )
        cache = output.past_key_values
        past_length += len(chunk_ids)
        period_count += sum(
            category == "irrelevant_periods"
            for category in chunk_categories
        )
        del output
    base.synchronize()
    return (
        cache,
        time.perf_counter() - started,
        past_length,
        period_count,
    )


def capture_query(
    model: Any,
    query_ids: torch.Tensor,
    cache: Any,
    body_length: int,
    attention_mask_buffer: torch.Tensor,
    accumulator: QAccumulator,
) -> tuple[Any, dict[int, torch.Tensor]]:
    accumulator.query = {}
    accumulator.mode = "query"
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
    finally:
        accumulator.mode = "idle"
    if len(accumulator.query) != len(model.model.layers):
        raise RuntimeError(
            f"captured {len(accumulator.query)} query layers"
        )
    return output, dict(accumulator.query)


def critical_query_tensor(
    query: dict[int, torch.Tensor],
    pairs: Sequence[tuple[int, int]],
) -> torch.Tensor:
    return torch.stack(
        [
            query[layer][head]
            for layer, head in pairs
        ],
        dim=0,
    )


def critical_scores(
    model: Any,
    queries: torch.Tensor,
    keys: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    position: int,
) -> list[float]:
    scores = []
    for index, (layer, head) in enumerate(pairs):
        device = model.model.layers[layer].self_attn.q_proj.weight.device
        query = queries[index].to(device).view(1, -1)
        post = incremental.apply_rope_at_position(
            model,
            query,
            position,
        )[0]
        key = keys[index].to(device)
        scale = float(
            getattr(
                model.model.layers[layer].self_attn,
                "scaling",
                post.shape[-1] ** -0.5,
            )
        )
        scores.append(
            float(torch.dot(post.float(), key.float()).item())
            * scale
        )
    return scores


def weighted_score(
    scores: Sequence[float],
    weights: torch.Tensor,
) -> float:
    return sum(
        float(weight) * float(score)
        for weight, score in zip(weights, scores)
    )


def q_mean_rows(
    query: dict[int, torch.Tensor],
    accumulator: QAccumulator,
    body_count: int,
    period_count: int,
    total_tokens: int,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = []
    query_layers = []
    global_layers = []
    period_layers = []
    for layer_index in sorted(query):
        q = query[layer_index].float()
        global_mean = (
            accumulator.body_sums[layer_index].detach().cpu()
            / float(body_count)
        )
        period_mean = (
            accumulator.period_sums[layer_index].detach().cpu()
            / float(period_count)
        )
        query_layers.append(q)
        global_layers.append(global_mean)
        period_layers.append(period_mean)
        for head in range(q.shape[0]):
            q_head = q[head]
            global_head = global_mean[head]
            period_head = period_mean[head]
            q_norm = float(torch.linalg.vector_norm(q_head).item())
            rows.append(
                {
                    "total_tokens": total_tokens,
                    "layer": layer_index,
                    "head": head,
                    "q_norm": rounded(q_norm),
                    "global_mean_norm": rounded(
                        torch.linalg.vector_norm(
                            global_head
                        ).item()
                    ),
                    "period_mean_norm": rounded(
                        torch.linalg.vector_norm(
                            period_head
                        ).item()
                    ),
                    "q_global_cosine": rounded(
                        F.cosine_similarity(
                            q_head,
                            global_head,
                            dim=0,
                        ).item()
                    ),
                    "q_period_cosine": rounded(
                        F.cosine_similarity(
                            q_head,
                            period_head,
                            dim=0,
                        ).item()
                    ),
                    "q_global_relative_l2": rounded(
                        torch.linalg.vector_norm(
                            q_head - global_head
                        ).item()
                        / max(q_norm, 1e-12)
                    ),
                    "q_period_relative_l2": rounded(
                        torch.linalg.vector_norm(
                            q_head - period_head
                        ).item()
                        / max(q_norm, 1e-12)
                    ),
                }
            )
    return (
        rows,
        torch.stack(query_layers),
        torch.stack(global_layers),
        torch.stack(period_layers),
    )


def fixed_rope_curve(
    model: Any,
    baseline_queries: torch.Tensor,
    keys: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    weights: torch.Tensor,
    start_position: int,
    end_position: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    count = end_position - start_position + 1
    total = torch.zeros(count, dtype=torch.float64)
    bands = {
        name: torch.zeros(count, dtype=torch.float64)
        for name in BAND_NAMES
    }
    band_slices = (
        slice(0, 16),
        slice(16, 32),
        slice(32, 48),
        slice(48, 64),
    )
    for index, (layer, head) in enumerate(pairs):
        device = model.model.layers[layer].self_attn.q_proj.weight.device
        q = baseline_queries[index].to(device)
        query = q.view(1, 1, 1, -1).expand(
            1,
            1,
            count,
            q.shape[-1],
        )
        positions = torch.arange(
            start_position,
            end_position + 1,
            dtype=torch.long,
            device=device,
        ).view(1, -1)
        cos, sin = model.model.rotary_emb(query, positions)
        rotated = base.apply_rope_to_q(
            query,
            (cos, sin),
        )[0, 0].float()
        key = keys[index].to(device).float()
        scale = float(
            getattr(
                model.model.layers[layer].self_attn,
                "scaling",
                q.shape[-1] ** -0.5,
            )
        )
        weighted_scale = float(weights[index]) * scale
        pair_contributions = (
            rotated[:, :64] * key[:64]
            + rotated[:, 64:] * key[64:]
        ) * weighted_scale
        total.add_(
            pair_contributions.sum(dim=-1).double().cpu()
        )
        for name, band_slice in zip(BAND_NAMES, band_slices):
            bands[name].add_(
                pair_contributions[
                    :,
                    band_slice,
                ].sum(dim=-1).double().cpu()
            )
        del query, positions, cos, sin, rotated, pair_contributions
    return total, bands


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
        prepare_case(tokenizer)
    )
    selected_heads, weight_map, ranked_pairs = (
        incremental.load_critical_heads(args.critical_heads_csv)
    )
    pairs = sorted(weight_map)
    weights = torch.tensor(
        [weight_map[pair] for pair in pairs],
        dtype=torch.float64,
    )
    del selected_heads
    if args.dry_run:
        write_json(
            output_dir / "dry_run.json",
            {
                "schema_version": 1,
                "total_tokens": len(case["prompt_ids"]),
                "base_body_tokens": len(base_body),
                "continuation_tokens": len(continuation),
                "query_tokens": len(query_ids_list),
                "critical_head_count": len(pairs),
                "ranked_head_count": len(ranked_pairs),
                "filler_text": case["filler_text"],
                "filler_token_id": case["filler_token_id"],
            },
        )
        print(
            (output_dir / "dry_run.json").read_text(
                encoding="utf-8"
            )
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
    attention_mask_buffer = torch.ones(
        (1, args.fixed_max_position_embeddings),
        dtype=torch.long,
        device=base.input_device(model),
    )
    query_ids = torch.tensor(
        query_ids_list,
        dtype=torch.long,
    ).view(1, -1)
    accumulator = QAccumulator(model)
    started = time.perf_counter()

    try:
        cache, prefill_seconds, body_count, period_count = (
            prefill_with_q_sums(
                model,
                base_body,
                categories[: len(base_body)],
                args.prefill_chunk_size,
                attention_mask_buffer,
                accumulator,
            )
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

        sample_totals = []
        sample_queries = []
        probe_rows = []
        probe_head_rows = []
        qmean_rows_all = []
        anchor_totals = []
        anchor_query = []
        anchor_global = []
        anchor_period = []
        baseline_queries: torch.Tensor | None = None
        baseline_position: int | None = None
        body_length = len(base_body)

        for added in range(len(continuation) + 1):
            if added > 0:
                next_id = int(continuation[added - 1])
                category = categories[body_length]
                body_output = forward_body(
                    model,
                    torch.tensor([[next_id]], dtype=torch.long),
                    cache,
                    body_length,
                    attention_mask_buffer,
                    accumulator,
                    [category],
                )
                cache = body_output.past_key_values
                body_length += 1
                body_count += 1
                period_count += int(
                    category == "irrelevant_periods"
                )
                del body_output

            total_tokens = START_TOTAL + added
            if (
                added % SAMPLE_STRIDE != 0
                and total_tokens not in ANCHOR_TOTALS
                and added != len(continuation)
            ):
                continue

            query_output, query = capture_query(
                model,
                query_ids,
                cache,
                body_length,
                attention_mask_buffer,
                accumulator,
            )
            query_position = body_length + len(query_ids_list) - 1
            critical_query = critical_query_tensor(query, pairs)
            if baseline_queries is None:
                baseline_queries = critical_query.clone()
                baseline_position = query_position
            assert baseline_position is not None
            actual_scores = critical_scores(
                model,
                critical_query,
                gold_keys,
                pairs,
                query_position,
            )
            position_scores = critical_scores(
                model,
                baseline_queries,
                gold_keys,
                pairs,
                query_position,
            )
            content_scores = critical_scores(
                model,
                critical_query,
                gold_keys,
                pairs,
                baseline_position,
            )
            baseline_scores = critical_scores(
                model,
                baseline_queries,
                gold_keys,
                pairs,
                baseline_position,
            )
            actual = weighted_score(actual_scores, weights)
            position_only = weighted_score(
                position_scores,
                weights,
            )
            content_only = weighted_score(
                content_scores,
                weights,
            )
            baseline_score = weighted_score(
                baseline_scores,
                weights,
            )
            interaction = (
                actual
                - position_only
                - content_only
                + baseline_score
            )
            score = first_token.score_first_token(
                tokenizer,
                query_output.logits[0, -1],
                answer_variants,
            )
            sample_totals.append(total_tokens)
            sample_queries.append(critical_query)
            probe_rows.append(
                {
                    "total_tokens": total_tokens,
                    "query_position": query_position,
                    "critical_qk_actual": rounded(actual),
                    "critical_qk_position_only": rounded(
                        position_only
                    ),
                    "critical_qk_content_only": rounded(
                        content_only
                    ),
                    "critical_qk_baseline": rounded(
                        baseline_score
                    ),
                    "critical_qk_interaction": rounded(
                        interaction
                    ),
                    "gold_probability": score[
                        "gold_exact_probability"
                    ],
                    "gold_vs_competitor_margin": score[
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
            )
            for index, (layer, head) in enumerate(pairs):
                head_interaction = (
                    actual_scores[index]
                    - position_scores[index]
                    - content_scores[index]
                    + baseline_scores[index]
                )
                probe_head_rows.append(
                    {
                        "total_tokens": total_tokens,
                        "layer": layer,
                        "head": head,
                        "weight": rounded(weights[index]),
                        "qk_actual": rounded(
                            actual_scores[index]
                        ),
                        "qk_position_only": rounded(
                            position_scores[index]
                        ),
                        "qk_content_only": rounded(
                            content_scores[index]
                        ),
                        "qk_baseline": rounded(
                            baseline_scores[index]
                        ),
                        "qk_interaction": rounded(
                            head_interaction
                        ),
                        "weighted_actual": rounded(
                            float(weights[index])
                            * actual_scores[index]
                        ),
                        "weighted_position_delta": rounded(
                            float(weights[index])
                            * (
                                position_scores[index]
                                - baseline_scores[index]
                            )
                        ),
                        "weighted_content_delta": rounded(
                            float(weights[index])
                            * (
                                content_scores[index]
                                - baseline_scores[index]
                            )
                        ),
                        "weighted_interaction": rounded(
                            float(weights[index])
                            * head_interaction
                        ),
                    }
                )

            (
                mean_rows,
                query_tensor,
                global_tensor,
                period_tensor,
            ) = q_mean_rows(
                query,
                accumulator,
                body_count,
                period_count,
                total_tokens,
            )
            qmean_rows_all.extend(mean_rows)
            if total_tokens in ANCHOR_TOTALS:
                anchor_totals.append(total_tokens)
                anchor_query.append(query_tensor)
                anchor_global.append(global_tensor)
                anchor_period.append(period_tensor)
            del query_tensor, global_tensor, period_tensor

            cache = incremental.crop_cache(
                query_output.past_key_values,
                body_length,
            )
            del query_output, query
            if added % 512 == 0:
                print(
                    json.dumps(
                        {
                            "added": added,
                            "total": total_tokens,
                            "qk_actual": rounded(actual),
                            "position_only": rounded(position_only),
                            "content_only": rounded(content_only),
                            "top": score["top_token_label"],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

        assert baseline_queries is not None
        assert baseline_position is not None
        fixed_total, fixed_bands = fixed_rope_curve(
            model,
            baseline_queries,
            gold_keys,
            pairs,
            weights,
            baseline_position,
            baseline_position + END_TOTAL - START_TOTAL,
        )

        with (output_dir / "probe_points.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(probe_rows[0]),
            )
            writer.writeheader()
            writer.writerows(probe_rows)
        with (output_dir / "q_mean_heads.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(qmean_rows_all[0]),
            )
            writer.writeheader()
            writer.writerows(qmean_rows_all)
        with (output_dir / "probe_head_points.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(probe_head_rows[0]),
            )
            writer.writeheader()
            writer.writerows(probe_head_rows)
        with (output_dir / "fixed_rope_curve.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            fields = [
                "total_tokens",
                "query_position",
                "fixed_q_rope_qk",
                *BAND_NAMES,
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for offset in range(len(fixed_total)):
                row = {
                    "total_tokens": START_TOTAL + offset,
                    "query_position": baseline_position + offset,
                    "fixed_q_rope_qk": rounded(
                        fixed_total[offset].item()
                    ),
                }
                for name in BAND_NAMES:
                    row[name] = rounded(
                        fixed_bands[name][offset].item()
                    )
                writer.writerow(row)

        torch.save(
            {
                "sample_totals": torch.tensor(sample_totals),
                "critical_pairs": torch.tensor(pairs),
                "critical_weights": weights.float(),
                "critical_query_pre": torch.stack(sample_queries),
                "critical_gold_key_post": gold_keys,
                "anchor_totals": torch.tensor(anchor_totals),
                "anchor_query_pre": torch.stack(anchor_query),
                "anchor_global_q_mean": torch.stack(anchor_global),
                "anchor_period_q_mean": torch.stack(anchor_period),
            },
            output_dir / "probe_vectors.pt",
        )
        write_json(
            output_dir / "manifest.json",
            {
                "schema_version": 1,
                "experiment": "onehop_q_mean_rope_probe",
                "start_total_tokens": START_TOTAL,
                "end_total_tokens": END_TOTAL,
                "sample_stride": SAMPLE_STRIDE,
                "sample_count": len(sample_totals),
                "q_mean_sample_stride": SAMPLE_STRIDE,
                "q_mean_row_count": len(qmean_rows_all),
                "anchor_totals": anchor_totals,
                "critical_head_count": len(pairs),
                "probe_head_row_count": len(probe_head_rows),
                "q_stage": "post-q_norm, pre-RoPE",
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
                "global_mean_definition": (
                    "mean pre-RoPE Q over every body token before the query"
                ),
                "period_mean_definition": (
                    "mean pre-RoPE Q over irrelevant period filler tokens"
                ),
                "prefill_count": 1,
                "prefill_seconds": rounded(prefill_seconds),
                "body_count_end": body_count,
                "period_count_end": period_count,
                "elapsed_seconds": rounded(
                    time.perf_counter() - started
                ),
                "complete": True,
            },
        )
    finally:
        accumulator.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
