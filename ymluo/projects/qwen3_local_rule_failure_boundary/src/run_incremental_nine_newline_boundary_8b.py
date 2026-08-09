from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import run_age_distractor_failure_boundary_8b as boundary
import run_fixed300_age_distractor_qk_8b as age
import run_local_rule_failure_boundary as base


START_TOTAL = 139264
END_TOTAL = 147456


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Token-by-token 136K->144K nine-vs-newline boundary scan with one shared-prefix prefill."
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
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument(
        "--max-added-tokens",
        type=int,
        default=END_TOTAL - START_TOTAL,
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    return parser.parse_args()


def rounded(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def token_ids(tokenizer: Any, text: str, label: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise RuntimeError(f"{label} tokenized to no tokens")
    return [int(token_id) for token_id in ids]


def one_token_id(tokenizer: Any, text: str, label: str) -> int:
    ids = token_ids(tokenizer, text, label)
    if len(ids) != 1:
        raise RuntimeError(f"{label} must be one token, got {ids}: {text!r}")
    return ids[0]


def load_critical_heads(
    path: str,
) -> tuple[dict[int, set[int]], dict[tuple[int, int], float], list[tuple[int, int]]]:
    selected: dict[int, set[int]] = {}
    raw_weights: dict[tuple[int, int], float] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("critical") != "True":
                continue
            layer = int(row["layer"])
            head = int(row["head"])
            selected.setdefault(layer, set()).add(head)
            raw_weights[(layer, head)] = float(row["correct_context_gold_mass"])
    if not raw_weights:
        raise RuntimeError("critical-head CSV selected no heads")
    total = sum(raw_weights.values())
    weights = {pair: value / total for pair, value in raw_weights.items()}
    ranked = sorted(raw_weights, key=raw_weights.get, reverse=True)
    return selected, weights, ranked


def build_category_lookup(
    total_tokens: int,
    category_positions: dict[str, Sequence[int]],
) -> list[str]:
    lookup = ["unknown"] * total_tokens
    for category, positions in category_positions.items():
        for position in positions:
            lookup[int(position)] = category
    if any(value == "unknown" for value in lookup):
        raise RuntimeError("category partition does not cover the maximum prompt")
    return lookup


def forward_with_shared_mask(
    model: Any,
    input_ids: torch.Tensor,
    cache: Any,
    past_length: int,
    attention_mask_buffer: torch.Tensor,
    *,
    with_logits: bool,
) -> Any:
    query_length = int(input_ids.shape[1])
    device = base.input_device(model)
    ids = input_ids.to(device)
    positions = torch.arange(
        past_length,
        past_length + query_length,
        dtype=torch.long,
        device=device,
    )
    kwargs = {
        "input_ids": ids,
        "past_key_values": cache,
        "attention_mask": attention_mask_buffer[:, : past_length + query_length],
        "position_ids": positions.view(1, -1),
        "cache_position": positions,
        "use_cache": True,
        "return_dict": True,
    }
    if with_logits:
        kwargs["logits_to_keep"] = 1
        return model(**kwargs)
    return model.model(**kwargs)


def capture_selected_pre_query(
    model: Any,
    selected_heads: dict[int, set[int]],
    query_ids: torch.Tensor,
    cache: Any,
    body_length: int,
    attention_mask_buffer: torch.Tensor,
) -> tuple[Any, dict[tuple[int, int], torch.Tensor], float]:
    captured: dict[tuple[int, int], torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int, heads: list[int]):
        def hook(module: Any, hook_args: tuple[Any, ...], output: torch.Tensor) -> None:
            if output.ndim != 4:
                raise RuntimeError(
                    f"layer {layer_index}: unexpected Q-Norm shape {tuple(output.shape)}"
                )
            for head in heads:
                captured[(layer_index, head)] = (
                    output[0, -1, head].detach().float().cpu()
                )

        return hook

    for layer_index, heads in selected_heads.items():
        handles.append(
            model.model.layers[layer_index].self_attn.q_norm.register_forward_hook(
                make_hook(layer_index, sorted(heads))
            )
        )
    base.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = forward_with_shared_mask(
                model,
                query_ids,
                cache,
                body_length,
                attention_mask_buffer,
                with_logits=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    base.synchronize()
    expected = sum(len(heads) for heads in selected_heads.values())
    if len(captured) != expected:
        raise RuntimeError(f"captured {len(captured)} critical Q vectors, expected {expected}")
    return output, captured, time.perf_counter() - started


def extract_gold_keys(
    cache: Any,
    model: Any,
    selected_pairs: Sequence[tuple[int, int]],
    gold_position: int,
) -> dict[tuple[int, int], torch.Tensor]:
    legacy = base.legacy_cache(cache)
    output: dict[tuple[int, int], torch.Tensor] = {}
    num_heads = int(model.config.num_attention_heads)
    for layer_index, head in selected_pairs:
        key = legacy[layer_index][0][0]
        kv_heads = int(key.shape[0])
        groups = max(1, num_heads // kv_heads)
        kv_head = min(kv_heads - 1, head // groups)
        output[(layer_index, head)] = (
            key[kv_head, gold_position].detach().float().cpu()
        )
    return output


def selected_query_metrics(
    model: Any,
    query_pre: dict[tuple[int, int], torch.Tensor],
    gold_keys: dict[tuple[int, int], torch.Tensor],
    baseline_pre: dict[tuple[int, int], torch.Tensor],
    baseline_post: dict[tuple[int, int], torch.Tensor],
    weights: dict[tuple[int, int], float],
    query_position: int,
    top_pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    qk_values = []
    pre_cosines = []
    post_cosines = []
    weighted_qk = 0.0
    weighted_pre_cosine = 0.0
    weighted_post_cosine = 0.0
    top: dict[str, Any] = {}
    for pair, pre_cpu in query_pre.items():
        layer_index, head = pair
        layer = model.model.layers[layer_index]
        device = layer.self_attn.q_proj.weight.device
        pre = pre_cpu.to(device)
        post = apply_rope_at_position(model, pre.view(1, -1), query_position)[0]
        key = gold_keys[pair].to(device)
        scale = float(getattr(layer.self_attn, "scaling", pre.shape[-1] ** -0.5))
        qk = float(torch.dot(post.float(), key.float()).item()) * scale
        pre_cosine = float(
            F.cosine_similarity(
                pre_cpu.float(),
                baseline_pre[pair].float(),
                dim=0,
            ).item()
        )
        post_cpu = post.detach().float().cpu()
        post_cosine = float(
            F.cosine_similarity(
                post_cpu,
                baseline_post[pair].float(),
                dim=0,
            ).item()
        )
        qk_values.append(qk)
        pre_cosines.append(pre_cosine)
        post_cosines.append(post_cosine)
        weight = weights[pair]
        weighted_qk += weight * qk
        weighted_pre_cosine += weight * pre_cosine
        weighted_post_cosine += weight * post_cosine
        if pair in top_pairs:
            top[f"L{layer_index}H{head}"] = {
                "qk": rounded(qk),
                "pre_cosine": rounded(pre_cosine),
                "post_cosine": rounded(post_cosine),
            }
    qk_sorted = sorted(qk_values)
    middle = len(qk_sorted) // 2
    median_qk = (
        qk_sorted[middle]
        if len(qk_sorted) % 2
        else 0.5 * (qk_sorted[middle - 1] + qk_sorted[middle])
    )
    return {
        "critical_qk_mean": rounded(sum(qk_values) / len(qk_values)),
        "critical_qk_median": rounded(median_qk),
        "critical_qk_weighted": rounded(weighted_qk),
        "critical_pre_cosine_mean": rounded(sum(pre_cosines) / len(pre_cosines)),
        "critical_pre_cosine_weighted": rounded(weighted_pre_cosine),
        "critical_post_cosine_mean": rounded(sum(post_cosines) / len(post_cosines)),
        "critical_post_cosine_weighted": rounded(weighted_post_cosine),
        "top_heads": top,
    }


def apply_rope_at_position(
    model: Any,
    query_pre: torch.Tensor,
    position: int,
) -> torch.Tensor:
    query = query_pre.view(1, query_pre.shape[0], 1, query_pre.shape[1])
    position_ids = torch.tensor(
        [[position]],
        dtype=torch.long,
        device=query.device,
    )
    cos, sin = model.model.rotary_emb(query, position_ids)
    return base.apply_rope_to_q(query, (cos, sin))[0, :, 0, :]


def crop_cache(cache: Any, length: int) -> Any:
    if not hasattr(cache, "crop"):
        raise RuntimeError("Transformers cache does not support crop()")
    cache.crop(length)
    return cache


def flatten_row(
    point: dict[str, Any],
    top_pair_names: Sequence[str],
) -> dict[str, Any]:
    row = {
        key: value
        for key, value in point.items()
        if key not in {"top_heads"}
    }
    for name in top_pair_names:
        values = point["top_heads"].get(name, {})
        row[f"{name}_qk"] = values.get("qk")
        row[f"{name}_pre_cosine"] = values.get("pre_cosine")
        row[f"{name}_post_cosine"] = values.get("post_cosine")
    return row


def checkpoint_summary(
    rows: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    prefill_seconds: float,
    continuation_counts: dict[str, int],
    started: float,
) -> None:
    crossings: dict[str, Any] = {}
    definitions = {
        "nine_vs_newline": "nine_newline_margin",
        "full_vocab": "full_vocab_margin",
        "age_candidate": "candidate_margin",
    }
    for name, field in definitions.items():
        failed = next((row for row in rows if float(row[field]) <= 0.0), None)
        crossings[name] = failed
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "experiment": "incremental_nine_vs_newline_boundary",
            "start_total_tokens": START_TOTAL,
            "end_total_tokens": START_TOTAL + args.max_added_tokens,
            "requested_added_tokens": args.max_added_tokens,
            "completed_points": len(rows),
            "last_point": rows[-1] if rows else None,
            "first_crossings": crossings,
            "continuation_category_counts": continuation_counts,
            "prefill_seconds": rounded(prefill_seconds),
            "elapsed_seconds": rounded(time.perf_counter() - started),
            "single_shared_prefix_prefill": True,
        },
    )


def main() -> None:
    args = parse_args()
    max_added = min(args.max_added_tokens, END_TOTAL - START_TOTAL)
    if max_added < 0:
        raise ValueError("max-added-tokens must be non-negative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    answer_token_ids = age.validate_answer_vocabulary(tokenizer)
    nine_id = int(answer_token_ids["nine"])
    newline_id = one_token_id(tokenizer, " \n\n", "newline competitor")
    distractor_pool = boundary.build_distractor_pool(tokenizer)
    max_case = boundary.build_case(
        tokenizer,
        distractor_pool,
        END_TOTAL,
        4607,
    )
    query_start, query_end = max_case["query_span"]
    query_ids_list = max_case["prompt_ids"][query_start:query_end]
    query_length = len(query_ids_list)
    base_body_length = START_TOTAL - query_length
    max_body_ids = max_case["prompt_ids"][:query_start]
    if len(max_body_ids) != END_TOTAL - query_length:
        raise AssertionError("unexpected maximum body length")
    base_body_ids = max_body_ids[:base_body_length]
    continuation_ids = max_body_ids[base_body_length:]
    if len(continuation_ids) != END_TOTAL - START_TOTAL:
        raise AssertionError("continuation must contain exactly 8192 tokens")
    continuation_ids = continuation_ids[:max_added]
    category_lookup = build_category_lookup(
        END_TOTAL,
        max_case["category_positions"],
    )
    continuation_categories = category_lookup[
        base_body_length : base_body_length + max_added
    ]
    continuation_counts = {
        category: continuation_categories.count(category)
        for category in boundary.PARTITION_ORDER
    }

    selected_heads, weights, ranked_pairs = load_critical_heads(args.critical_heads_csv)
    selected_pairs = sorted(weights)
    top_pairs = ranked_pairs[:10]
    top_pair_names = [f"L{layer}H{head}" for layer, head in top_pairs]

    max_position = START_TOTAL + max_added
    rope_factor = base.rope_factor_for_length(
        max_position,
        args.original_max_position_embeddings,
    )
    model, tokenizer = base.load_model_and_tokenizer(
        args,
        max_position,
        rope_factor,
    )
    input_device = base.input_device(model)
    attention_mask_buffer = torch.ones(
        (1, max_position),
        dtype=torch.long,
        device=input_device,
    )
    query_ids = torch.tensor(query_ids_list, dtype=torch.long).view(1, -1)
    body_prompt = torch.tensor(base_body_ids, dtype=torch.long).view(1, -1)

    design = {
        "schema_version": 1,
        "experiment": "incremental_nine_vs_newline_boundary",
        "model_name_or_path": args.model_name_or_path,
        "single_shared_prefix_prefill": True,
        "start_total_tokens": START_TOTAL,
        "end_total_tokens": START_TOTAL + max_added,
        "base_body_tokens": base_body_length,
        "query_tokens": query_length,
        "continuation_tokens": max_added,
        "gold_age_position": max_case["gold_age_span"][0],
        "nine_token_id": nine_id,
        "newline_token_id": newline_id,
        "newline_token_text": tokenizer.decode([newline_id]),
        "critical_head_count": len(selected_pairs),
        "critical_heads": [
            {
                "layer": layer,
                "head": head,
                "weight": weights[(layer, head)],
            }
            for layer, head in ranked_pairs
        ],
        "top_head_names": top_pair_names,
        "continuation_category_counts": continuation_counts,
        "base_body_suffix": tokenizer.decode(base_body_ids[-100:]),
        "continuation_prefix": tokenizer.decode(continuation_ids[:100]),
        "continuation_suffix": tokenizer.decode(continuation_ids[-100:]),
        "query_text": tokenizer.decode(query_ids_list),
    }
    write_json(output_dir / "design.json", design)

    started = time.perf_counter()
    legacy_cache, prefill_seconds = base.prefill_sequence(
        model,
        body_prompt,
        args.prefill_chunk_size,
    )
    cache = base.cache_from_legacy(legacy_cache)
    del legacy_cache, body_prompt
    gold_keys = extract_gold_keys(
        cache,
        model,
        selected_pairs,
        max_case["gold_age_span"][0],
    )

    csv_path = output_dir / "points.csv"
    rows: list[dict[str, Any]] = []
    baseline_pre: dict[tuple[int, int], torch.Tensor] | None = None
    baseline_post: dict[tuple[int, int], torch.Tensor] | None = None
    body_length = base_body_length
    csv_handle = csv_path.open("w", encoding="utf-8", newline="")
    writer: csv.DictWriter | None = None
    point_times: list[float] = []
    try:
        for added in range(max_added + 1):
            point_started = time.perf_counter()
            if added > 0:
                next_id = continuation_ids[added - 1]
                with torch.inference_mode():
                    body_output = forward_with_shared_mask(
                        model,
                        torch.tensor([[next_id]], dtype=torch.long),
                        cache,
                        body_length,
                        attention_mask_buffer,
                        with_logits=False,
                    )
                cache = body_output.past_key_values
                body_length += 1
                del body_output

            query_output, query_pre, query_seconds = capture_selected_pre_query(
                model,
                selected_heads,
                query_ids,
                cache,
                body_length,
                attention_mask_buffer,
            )
            logits = query_output.logits[0, -1].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            top_id = int(torch.argmax(logits).item())
            answer = age.score_answer(tokenizer, query_output, answer_token_ids)
            query_position = body_length + query_length - 1

            if baseline_pre is None:
                baseline_pre = {
                    pair: value.clone()
                    for pair, value in query_pre.items()
                }
                baseline_post = {
                    pair: apply_rope_at_position(
                        model,
                        value.to(
                            model.model.layers[pair[0]].self_attn.q_proj.weight.device
                        ).view(1, -1),
                        query_position,
                    )[0].detach().float().cpu()
                    for pair, value in query_pre.items()
                }
            assert baseline_post is not None
            query_metrics = selected_query_metrics(
                model,
                query_pre,
                gold_keys,
                baseline_pre,
                baseline_post,
                weights,
                query_position,
                top_pairs,
            )
            point = {
                "added_tokens": added,
                "total_tokens": START_TOTAL + added,
                "body_tokens": body_length,
                "query_position": query_position,
                "added_token_id": (
                    None if added == 0 else continuation_ids[added - 1]
                ),
                "added_token_text": (
                    ""
                    if added == 0
                    else tokenizer.decode([continuation_ids[added - 1]])
                ),
                "added_token_category": (
                    "baseline" if added == 0 else continuation_categories[added - 1]
                ),
                "nine_probability": rounded(probabilities[nine_id].item()),
                "newline_probability": rounded(probabilities[newline_id].item()),
                "nine_log_probability": rounded(log_probs[nine_id].item()),
                "newline_log_probability": rounded(log_probs[newline_id].item()),
                "nine_newline_margin": rounded(
                    (logits[nine_id] - logits[newline_id]).item()
                ),
                "gold_ppl": answer["gold_ppl"],
                "full_vocab_margin": answer["full_vocab_margin"],
                "full_vocab_correct": answer["full_vocab_correct"],
                "top_token_id": top_id,
                "top_token": tokenizer.decode([top_id]),
                "candidate_margin": answer["candidate_margin"],
                "candidate_correct": answer["candidate_correct"],
                "candidate_prediction": answer["candidate_prediction"],
                "critical_qk_mean": query_metrics["critical_qk_mean"],
                "critical_qk_median": query_metrics["critical_qk_median"],
                "critical_qk_weighted": query_metrics["critical_qk_weighted"],
                "critical_pre_cosine_mean": query_metrics[
                    "critical_pre_cosine_mean"
                ],
                "critical_pre_cosine_weighted": query_metrics[
                    "critical_pre_cosine_weighted"
                ],
                "critical_post_cosine_mean": query_metrics[
                    "critical_post_cosine_mean"
                ],
                "critical_post_cosine_weighted": query_metrics[
                    "critical_post_cosine_weighted"
                ],
                "query_seconds": rounded(query_seconds),
                "point_seconds": rounded(time.perf_counter() - point_started),
                "top_heads": query_metrics["top_heads"],
            }
            flat = flatten_row(point, top_pair_names)
            if writer is None:
                writer = csv.DictWriter(csv_handle, fieldnames=list(flat))
                writer.writeheader()
            writer.writerow(flat)
            csv_handle.flush()
            rows.append(point)
            point_times.append(float(point["point_seconds"]))

            cache = crop_cache(query_output.past_key_values, body_length)
            del query_output

            if (
                added % args.checkpoint_every == 0
                or added == max_added
            ):
                checkpoint_summary(
                    rows,
                    output_dir,
                    args,
                    prefill_seconds,
                    continuation_counts,
                    started,
                )
                recent = point_times[-min(len(point_times), args.checkpoint_every) :]
                seconds_per_point = sum(recent) / len(recent)
                remaining = max_added - added
                print(
                    json.dumps(
                        {
                            "added": added,
                            "total": START_TOTAL + added,
                            "nine_newline_margin": point["nine_newline_margin"],
                            "full_vocab_margin": point["full_vocab_margin"],
                            "candidate_margin": point["candidate_margin"],
                            "top_token": point["top_token"],
                            "critical_qk_weighted": point["critical_qk_weighted"],
                            "seconds_per_point_recent": rounded(seconds_per_point),
                            "eta_seconds": rounded(remaining * seconds_per_point),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    finally:
        csv_handle.close()

    checkpoint_summary(
        rows,
        output_dir,
        args,
        prefill_seconds,
        continuation_counts,
        started,
    )


if __name__ == "__main__":
    main()
