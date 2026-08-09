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
import run_fixed300_age_distractor_qk_8b as age
import run_local_rule_failure_boundary as base


CORRECT_TOTAL = 139264
CORRECT_DISTRACTORS = 4351
FAILED_TOTAL = 147456
FAILED_DISTRACTORS = 4607


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-RoPE Query x position counterfactual ablation for the 136K/144K age boundary."
    )
    parser.add_argument("--model-name-or-path", required=True)
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
        "--critical-mass-threshold",
        type=float,
        default=0.01,
        help="136K gold-age attention threshold defining critical layer-heads",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def rounded(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def capture_query_pre_rope(
    model: Any,
    prefix_cache: Any,
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
) -> tuple[Any, dict[int, torch.Tensor], dict[int, torch.Tensor], float]:
    layers = list(model.model.layers)
    captured_pre: dict[int, torch.Tensor] = {}
    captured_post: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(module: Any, hook_args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and hook_args:
                hidden_states = hook_args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(hook_args) >= 2:
                position_embeddings = hook_args[1]
            if hidden_states is None:
                raise RuntimeError(f"layer {layer_index}: missing hidden states")
            projected = module.q_proj(hidden_states)
            batch, query_length, _ = projected.shape
            head_dim = int(module.head_dim)
            num_heads = int(projected.shape[-1] // head_dim)
            query = projected.view(batch, query_length, num_heads, head_dim)
            if getattr(module, "q_norm", None) is not None:
                query = module.q_norm(query)
            query = query.transpose(1, 2)
            pre = query[:, :, -1, :].detach()
            post = base.apply_rope_to_q(query, position_embeddings)[:, :, -1, :].detach()
            captured_pre[layer_index] = pre.float().cpu()
            captured_post[layer_index] = post.float().cpu()

        return hook

    for layer_index, layer in enumerate(layers):
        handles.append(
            layer.self_attn.register_forward_pre_hook(
                make_hook(layer_index),
                with_kwargs=True,
            )
        )
    base.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = base.forward_with_cache(
                model,
                last_prompt_id.to(base.input_device(model)),
                prefix_cache,
                prompt_len_minus_one,
            )
    finally:
        for handle in handles:
            handle.remove()
    base.synchronize()
    if len(captured_pre) != len(layers):
        raise RuntimeError(
            f"captured {len(captured_pre)} query tensors for {len(layers)} layers"
        )
    return output, captured_pre, captured_post, time.perf_counter() - started


def rotary_components(
    model: Any,
    reference: torch.Tensor,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = torch.tensor(
        [[position]],
        dtype=torch.long,
        device=reference.device,
    )
    cos, sin = model.model.rotary_emb(reference, position_ids)
    return (
        cos.to(device=reference.device, dtype=reference.dtype),
        sin.to(device=reference.device, dtype=reference.dtype),
    )


def apply_rope_at_position(
    model: Any,
    query_pre: torch.Tensor,
    position: int,
) -> torch.Tensor:
    query = query_pre.view(1, query_pre.shape[0], 1, query_pre.shape[1])
    position_embeddings = rotary_components(model, query, position)
    return base.apply_rope_to_q(query, position_embeddings)[0, :, 0, :]


def inverse_rope(
    query_post: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos_vector = cos[0, 0].to(query_post)
    sin_vector = sin[0, 0].to(query_post)
    denominator = (cos_vector.square() + sin_vector.square()).clamp_min(1e-20)
    return (
        query_post * cos_vector
        - base.rotate_half(query_post) * sin_vector
    ) / denominator


def fake_pre_for_target_position(
    model: Any,
    target_pre: torch.Tensor,
    target_position: int,
    actual_position: int,
) -> torch.Tensor:
    target_post = apply_rope_at_position(model, target_pre, target_position)
    reference = target_post.view(1, target_post.shape[0], 1, target_post.shape[1])
    actual_cos, actual_sin = rotary_components(model, reference, actual_position)
    fake_pre = inverse_rope(target_post, actual_cos, actual_sin)
    reconstructed = apply_rope_at_position(model, fake_pre, actual_position)
    error = float((reconstructed.float() - target_post.float()).abs().max().item())
    # Qwen3-8B runs these tensors in BF16. At large YaRN positions the final
    # quantization can produce one BF16 bin (commonly 0.0625) of error even
    # though the FP32 inverse is exact to ~1e-6.
    if error > 0.15:
        raise RuntimeError(f"RoPE inverse reconstruction error too large: {error}")
    return fake_pre


def cache_crop(cache: Any, length: int) -> Any:
    if not hasattr(cache, "crop"):
        raise RuntimeError("the installed Transformers cache does not support crop()")
    cache.crop(length)
    return cache


def extract_gold_keys(
    query_output: Any,
    gold_position: int,
) -> dict[int, torch.Tensor]:
    cache = base.legacy_cache(query_output.past_key_values)
    return {
        layer_index: layer_cache[0][0, :, gold_position, :].detach().float().cpu()
        for layer_index, layer_cache in enumerate(cache)
    }


def key_consistency(
    correct_keys: dict[int, torch.Tensor],
    failed_keys: dict[int, torch.Tensor],
) -> dict[str, Any]:
    cosine_rows: list[float] = []
    max_abs_rows: list[float] = []
    norm_ratio_rows: list[float] = []
    per_layer = []
    for layer in sorted(correct_keys):
        correct = correct_keys[layer]
        failed = failed_keys[layer]
        cosines = F.cosine_similarity(correct, failed, dim=-1)
        max_abs = (correct - failed).abs().amax(dim=-1)
        ratios = failed.norm(dim=-1) / correct.norm(dim=-1).clamp_min(1e-20)
        cosine_rows.extend(float(value) for value in cosines.tolist())
        max_abs_rows.extend(float(value) for value in max_abs.tolist())
        norm_ratio_rows.extend(float(value) for value in ratios.tolist())
        per_layer.append(
            {
                "layer": layer,
                "mean_cosine": rounded(cosines.mean().item()),
                "min_cosine": rounded(cosines.min().item()),
                "max_abs_difference": rounded(max_abs.max().item()),
                "mean_norm_ratio": rounded(ratios.mean().item()),
            }
        )
    return {
        "mean_cosine": rounded(sum(cosine_rows) / len(cosine_rows)),
        "min_cosine": rounded(min(cosine_rows)),
        "max_abs_difference": rounded(max(max_abs_rows)),
        "mean_norm_ratio": rounded(sum(norm_ratio_rows) / len(norm_ratio_rows)),
        "per_layer": per_layer,
    }


def query_pre_similarity(
    correct_query: dict[int, torch.Tensor],
    failed_query: dict[int, torch.Tensor],
) -> dict[str, Any]:
    per_layer = []
    all_cosines: list[float] = []
    all_norm_ratios: list[float] = []
    for layer in sorted(correct_query):
        correct = correct_query[layer][0]
        failed = failed_query[layer][0]
        cosines = F.cosine_similarity(correct, failed, dim=-1)
        ratios = failed.norm(dim=-1) / correct.norm(dim=-1).clamp_min(1e-20)
        all_cosines.extend(float(value) for value in cosines.tolist())
        all_norm_ratios.extend(float(value) for value in ratios.tolist())
        per_layer.append(
            {
                "layer": layer,
                "mean_cosine": rounded(cosines.mean().item()),
                "min_cosine": rounded(cosines.min().item()),
                "mean_norm_ratio": rounded(ratios.mean().item()),
            }
        )
    return {
        "mean_cosine": rounded(sum(all_cosines) / len(all_cosines)),
        "min_cosine": rounded(min(all_cosines)),
        "mean_norm_ratio": rounded(sum(all_norm_ratios) / len(all_norm_ratios)),
        "per_layer": per_layer,
    }


def evaluate_queries_on_cache(
    model: Any,
    query_output: Any,
    query_conditions: dict[str, tuple[dict[int, torch.Tensor], int]],
    category_positions: dict[str, Sequence[int]],
) -> dict[str, Any]:
    layers = list(model.model.layers)
    cache = base.legacy_cache(query_output.past_key_values)
    key_length = int(cache[0][0].shape[2])
    categories = list(boundary.PARTITION_ORDER)
    output: dict[str, Any] = {}
    for condition, (pre_by_layer, query_position) in query_conditions.items():
        head_rows = []
        for layer_index, layer in enumerate(layers):
            key = cache[layer_index][0][0]
            pre = pre_by_layer[layer_index][0].to(key.device)
            post = apply_rope_at_position(model, pre, query_position)
            num_heads = int(post.shape[0])
            kv_heads = int(key.shape[0])
            groups = max(1, num_heads // kv_heads)
            scale = float(getattr(layer.self_attn, "scaling", post.shape[-1] ** -0.5))
            category_indices = {
                category: torch.tensor(
                    list(category_positions[category]),
                    dtype=torch.long,
                    device=key.device,
                )
                for category in categories
            }
            for head in range(num_heads):
                kv_head = min(kv_heads - 1, head // groups)
                logits = torch.matmul(key[kv_head].float(), post[head].float()) * scale
                probabilities = torch.softmax(logits, dim=-1)
                gold_index = category_indices["gold_age"]
                if int(gold_index.numel()) != 1:
                    raise RuntimeError("gold_age must contain exactly one token")
                gold_position = int(gold_index.item())
                gold_logit = float(logits[gold_position].item())
                gold_mass = float(probabilities[gold_position].item())
                category_mass = {
                    category: rounded(
                        probabilities.index_select(0, indices).sum().item()
                    )
                    for category, indices in category_indices.items()
                }
                head_rows.append(
                    {
                        "layer": layer_index,
                        "head": head,
                        "gold_logit": rounded(gold_logit),
                        "gold_mass": rounded(gold_mass),
                        "gold_rank": int((logits > gold_logit).sum().item()) + 1,
                        "category_mass": category_mass,
                    }
                )
        output[condition] = {
            "query_position": query_position,
            "key_length": key_length,
            "head_rows": head_rows,
        }
    return output


def condition_lookup(condition: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["layer"]), int(row["head"])): row
        for row in condition["head_rows"]
    }


def summarize_counterfactuals(
    correct_baseline: dict[str, Any],
    failed_conditions: dict[str, Any],
    critical_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, set[int]]]:
    correct_lookup = condition_lookup(correct_baseline)
    lookups = {
        name: condition_lookup(body)
        for name, body in failed_conditions.items()
    }
    critical_pairs = {
        pair
        for pair, row in correct_lookup.items()
        if float(row["gold_mass"]) >= critical_threshold
    }
    critical_by_layer: dict[int, set[int]] = {}
    for layer, head in critical_pairs:
        critical_by_layer.setdefault(layer, set()).add(head)

    rows: list[dict[str, Any]] = []
    condition_names = list(failed_conditions)
    for pair in sorted(correct_lookup):
        layer, head = pair
        row: dict[str, Any] = {
            "layer": layer,
            "head": head,
            "critical": pair in critical_pairs,
            "correct_context_gold_logit": correct_lookup[pair]["gold_logit"],
            "correct_context_gold_mass": correct_lookup[pair]["gold_mass"],
            "correct_context_gold_rank": correct_lookup[pair]["gold_rank"],
        }
        for name in condition_names:
            row[f"{name}_gold_logit"] = lookups[name][pair]["gold_logit"]
            row[f"{name}_gold_mass"] = lookups[name][pair]["gold_mass"]
            row[f"{name}_gold_rank"] = lookups[name][pair]["gold_rank"]

        s00 = float(row["both_repaired_gold_logit"])
        s01 = float(row["query_repaired_gold_logit"])
        s10 = float(row["rope_repaired_gold_logit"])
        s11 = float(row["failed_actual_gold_logit"])
        row["query_state_shapley_logit"] = 0.5 * ((s10 - s00) + (s11 - s01))
        row["rope_position_shapley_logit"] = 0.5 * ((s01 - s00) + (s11 - s10))
        row["total_failed_minus_repaired_logit"] = s11 - s00

        m00 = math.log(max(float(row["both_repaired_gold_mass"]), 1e-30))
        m01 = math.log(max(float(row["query_repaired_gold_mass"]), 1e-30))
        m10 = math.log(max(float(row["rope_repaired_gold_mass"]), 1e-30))
        m11 = math.log(max(float(row["failed_actual_gold_mass"]), 1e-30))
        row["query_state_shapley_log_mass"] = 0.5 * ((m10 - m00) + (m11 - m01))
        row["rope_position_shapley_log_mass"] = 0.5 * ((m01 - m00) + (m11 - m10))
        row["total_failed_minus_repaired_log_mass"] = m11 - m00
        rows.append(row)

    def subset(rows_subset: list[dict[str, Any]]) -> dict[str, Any]:
        def average(field: str) -> float:
            return sum(float(row[field]) for row in rows_subset) / len(rows_subset)

        return {
            "count": len(rows_subset),
            "query_state_shapley_logit": rounded(average("query_state_shapley_logit")),
            "rope_position_shapley_logit": rounded(average("rope_position_shapley_logit")),
            "total_failed_minus_repaired_logit": rounded(
                average("total_failed_minus_repaired_logit")
            ),
            "query_state_shapley_log_mass": rounded(
                average("query_state_shapley_log_mass")
            ),
            "rope_position_shapley_log_mass": rounded(
                average("rope_position_shapley_log_mass")
            ),
            "total_failed_minus_repaired_log_mass": rounded(
                average("total_failed_minus_repaired_log_mass")
            ),
            "mean_gold_mass": {
                name: rounded(
                    sum(float(row[f"{name}_gold_mass"]) for row in rows_subset)
                    / len(rows_subset)
                )
                for name in condition_names
            },
            "mean_gold_rank": {
                name: rounded(
                    sum(float(row[f"{name}_gold_rank"]) for row in rows_subset)
                    / len(rows_subset)
                )
                for name in condition_names
            },
        }

    critical_rows = [row for row in rows if row["critical"]]
    summary = {
        "critical_threshold": critical_threshold,
        "all_heads": subset(rows),
        "critical_heads": subset(critical_rows),
    }
    return summary, rows, critical_by_layer


def install_query_intervention_hooks(
    model: Any,
    target_pre: dict[int, torch.Tensor],
    target_position: int,
    actual_position: int,
    selected_heads: dict[int, set[int]] | None,
) -> list[Any]:
    handles = []

    def make_hook(layer_index: int):
        def hook(module: Any, hook_args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            if output.shape[0] != 1 or output.shape[1] != 1:
                raise RuntimeError(
                    f"expected one query token at layer {layer_index}, got {tuple(output.shape)}"
                )
            target = target_pre[layer_index][0].to(
                device=output.device,
                dtype=torch.float32,
            )
            fake = fake_pre_for_target_position(
                model,
                target,
                target_position,
                actual_position,
            ).to(output)
            replacement = output.clone()
            if selected_heads is None:
                replacement[0, 0] = fake
            else:
                heads = sorted(selected_heads.get(layer_index, set()))
                if heads:
                    index = torch.tensor(heads, dtype=torch.long, device=output.device)
                    replacement[0, 0].index_copy_(0, index, fake.index_select(0, index))
            return replacement

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_norm.register_forward_hook(make_hook(layer_index)))
    return handles


def run_forward_intervention(
    model: Any,
    tokenizer: Any,
    answer_token_ids: dict[str, int],
    prefix_cache: Any,
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
    target_pre: dict[int, torch.Tensor],
    target_position: int,
    actual_position: int,
    selected_heads: dict[int, set[int]] | None,
) -> tuple[dict[str, Any], float]:
    handles = install_query_intervention_hooks(
        model,
        target_pre,
        target_position,
        actual_position,
        selected_heads,
    )
    base.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = base.forward_with_cache(
                model,
                last_prompt_id.to(base.input_device(model)),
                prefix_cache,
                prompt_len_minus_one,
            )
        answer = age.score_answer(tokenizer, output, answer_token_ids)
        prefix_cache = cache_crop(output.past_key_values, prompt_len_minus_one)
        del output
    finally:
        for handle in handles:
            handle.remove()
    base.synchronize()
    return answer, time.perf_counter() - started


def save_head_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key not in {"prompt_ids", "category_positions"}
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    answer_token_ids = age.validate_answer_vocabulary(tokenizer)
    distractor_pool = boundary.build_distractor_pool(tokenizer)
    correct_case = boundary.build_case(
        tokenizer,
        distractor_pool,
        CORRECT_TOTAL,
        CORRECT_DISTRACTORS,
    )
    failed_case = boundary.build_case(
        tokenizer,
        distractor_pool,
        FAILED_TOTAL,
        FAILED_DISTRACTORS,
    )
    max_position = FAILED_TOTAL
    rope_factor = base.rope_factor_for_length(
        max_position,
        args.original_max_position_embeddings,
    )
    model, tokenizer = base.load_model_and_tokenizer(
        args,
        max_position,
        rope_factor,
    )

    correct_prompt = torch.tensor(correct_case["prompt_ids"], dtype=torch.long).view(1, -1)
    correct_legacy, correct_prefill_seconds = base.prefill_sequence(
        model,
        correct_prompt[:, :-1],
        args.prefill_chunk_size,
    )
    correct_prefix_cache = base.cache_from_legacy(correct_legacy)
    del correct_legacy
    correct_output, correct_pre, correct_post, correct_query_seconds = capture_query_pre_rope(
        model,
        correct_prefix_cache,
        correct_prompt[:, -1:],
        CORRECT_TOTAL - 1,
    )
    correct_answer = age.score_answer(tokenizer, correct_output, answer_token_ids)
    correct_direct = evaluate_queries_on_cache(
        model,
        correct_output,
        {"correct_actual": (correct_pre, CORRECT_TOTAL - 1)},
        correct_case["category_positions"],
    )["correct_actual"]
    correct_keys = extract_gold_keys(
        correct_output,
        correct_case["category_positions"]["gold_age"][0],
    )
    cache_crop(correct_output.past_key_values, CORRECT_TOTAL - 1)
    del correct_output, correct_prefix_cache, correct_prompt
    gc.collect()
    torch.cuda.empty_cache()

    failed_prompt = torch.tensor(failed_case["prompt_ids"], dtype=torch.long).view(1, -1)
    failed_legacy, failed_prefill_seconds = base.prefill_sequence(
        model,
        failed_prompt[:, :-1],
        args.prefill_chunk_size,
    )
    failed_prefix_cache = base.cache_from_legacy(failed_legacy)
    del failed_legacy
    failed_output, failed_pre, failed_post, failed_query_seconds = capture_query_pre_rope(
        model,
        failed_prefix_cache,
        failed_prompt[:, -1:],
        FAILED_TOTAL - 1,
    )
    failed_answer = age.score_answer(tokenizer, failed_output, answer_token_ids)
    failed_keys = extract_gold_keys(
        failed_output,
        failed_case["category_positions"]["gold_age"][0],
    )
    failed_conditions = evaluate_queries_on_cache(
        model,
        failed_output,
        {
            "both_repaired": (correct_pre, CORRECT_TOTAL - 1),
            "query_repaired": (correct_pre, FAILED_TOTAL - 1),
            "rope_repaired": (failed_pre, CORRECT_TOTAL - 1),
            "failed_actual": (failed_pre, FAILED_TOTAL - 1),
        },
        failed_case["category_positions"],
    )
    counterfactual_summary, head_rows, critical_by_layer = summarize_counterfactuals(
        correct_direct,
        failed_conditions,
        args.critical_mass_threshold,
    )
    failed_prefix_cache = cache_crop(failed_output.past_key_values, FAILED_TOTAL - 1)
    del failed_output
    gc.collect()
    torch.cuda.empty_cache()

    vector_artifact = {
        "correct_pre_rope_query": torch.stack(
            [correct_pre[layer][0] for layer in sorted(correct_pre)]
        ),
        "failed_pre_rope_query": torch.stack(
            [failed_pre[layer][0] for layer in sorted(failed_pre)]
        ),
        "correct_post_rope_query": torch.stack(
            [correct_post[layer][0] for layer in sorted(correct_post)]
        ),
        "failed_post_rope_query": torch.stack(
            [failed_post[layer][0] for layer in sorted(failed_post)]
        ),
        "correct_gold_post_rope_key": torch.stack(
            [correct_keys[layer] for layer in sorted(correct_keys)]
        ),
        "failed_gold_post_rope_key": torch.stack(
            [failed_keys[layer] for layer in sorted(failed_keys)]
        ),
    }
    torch.save(vector_artifact, output_dir / "prerope_vectors.pt")
    key_check = key_consistency(correct_keys, failed_keys)
    query_check = query_pre_similarity(correct_pre, failed_pre)
    representation_checkpoint = {
        "schema_version": 1,
        "experiment": "age_distractor_prerope_query_position_ablation",
        "status": "representation_complete",
        "baseline_answers": {
            "correct": correct_answer,
            "failed": failed_answer,
        },
        "gold_key_consistency": key_check,
        "pre_rope_query_similarity": query_check,
        "counterfactual_summary": counterfactual_summary,
    }
    write_json(output_dir / "representation_result.json", representation_checkpoint)
    save_head_csv(output_dir / "counterfactual_heads.csv", head_rows)

    forward_results: dict[str, Any] = {
        "failed_actual": failed_answer,
    }
    interventions = [
        ("rope_repaired_all", failed_pre, CORRECT_TOTAL - 1, None),
        ("query_repaired_all", correct_pre, FAILED_TOTAL - 1, None),
        ("both_repaired_all", correct_pre, CORRECT_TOTAL - 1, None),
        ("rope_repaired_critical", failed_pre, CORRECT_TOTAL - 1, critical_by_layer),
        ("query_repaired_critical", correct_pre, FAILED_TOTAL - 1, critical_by_layer),
        ("both_repaired_critical", correct_pre, CORRECT_TOTAL - 1, critical_by_layer),
    ]
    intervention_timing: dict[str, float] = {}
    for name, target_pre, target_position, selected in interventions:
        answer, seconds = run_forward_intervention(
            model,
            tokenizer,
            answer_token_ids,
            failed_prefix_cache,
            failed_prompt[:, -1:],
            FAILED_TOTAL - 1,
            target_pre,
            target_position,
            FAILED_TOTAL - 1,
            selected,
        )
        forward_results[name] = answer
        intervention_timing[name] = rounded(seconds)

    result = {
        "schema_version": 1,
        "experiment": "age_distractor_prerope_query_position_ablation",
        "model_name_or_path": args.model_name_or_path,
        "model_config": {
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "num_attention_heads": int(model.config.num_attention_heads),
            "num_key_value_heads": int(model.config.num_key_value_heads),
            "head_dim": int(model.config.head_dim),
            "rope_theta": float(model.config.rope_theta),
            "max_position_embeddings": int(model.config.max_position_embeddings),
            "rope_scaling": model.config.rope_scaling,
        },
        "positions": {
            "gold_age": int(correct_case["category_positions"]["gold_age"][0]),
            "correct_query": CORRECT_TOTAL - 1,
            "failed_query": FAILED_TOTAL - 1,
            "position_delta": FAILED_TOTAL - CORRECT_TOTAL,
        },
        "cases": {
            "correct": public_case(correct_case),
            "failed": public_case(failed_case),
        },
        "baseline_answers": {
            "correct": correct_answer,
            "failed": failed_answer,
        },
        "gold_key_consistency": key_check,
        "pre_rope_query_similarity": query_check,
        "counterfactual_summary": counterfactual_summary,
        "forward_interventions": forward_results,
        "timing": {
            "correct_prefill_seconds": rounded(correct_prefill_seconds),
            "correct_query_seconds": rounded(correct_query_seconds),
            "failed_prefill_seconds": rounded(failed_prefill_seconds),
            "failed_query_seconds": rounded(failed_query_seconds),
            "interventions": intervention_timing,
        },
        "artifacts": {
            "vectors": "prerope_vectors.pt",
            "head_csv": "counterfactual_heads.csv",
        },
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps(
        {
            "baseline_answers": result["baseline_answers"],
            "gold_key_consistency": result["gold_key_consistency"],
            "pre_rope_query_similarity": result["pre_rope_query_similarity"],
            "counterfactual_summary": result["counterfactual_summary"],
            "forward_interventions": result["forward_interventions"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
