from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

import run_attention_confidence_sweep_8b as attention_runner
import run_local_rule_failure_boundary as base
import run_semantic_common_tail_pilot_8b as pilot


TARGET_ID = "scraper.n.01"
COMPETITOR_ID = "toolbox.n.01"


def parse_csv_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def component_vector(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    return value[:, -1, :].detach()


def install_residual_hooks(model: Any) -> tuple[dict[str, Any], list[Any]]:
    layers = list(model.model.layers)
    captured: dict[str, Any] = {
        "initial": None,
        "attention": {},
        "mlp": {},
        "layer_output": {},
    }
    handles: list[Any] = []

    def first_layer_pre_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if hidden is not None:
            captured["initial"] = hidden[:, -1, :].detach()

    handles.append(layers[0].register_forward_pre_hook(first_layer_pre_hook, with_kwargs=True))

    for layer_index, layer in enumerate(layers):
        def attention_hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
            index: int = layer_index,
        ) -> None:
            captured["attention"][index] = component_vector(output)

        def mlp_hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
            index: int = layer_index,
        ) -> None:
            captured["mlp"][index] = component_vector(output)

        def layer_hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
            index: int = layer_index,
        ) -> None:
            captured["layer_output"][index] = component_vector(output)

        handles.append(layer.self_attn.register_forward_hook(attention_hook, with_kwargs=True))
        handles.append(layer.mlp.register_forward_hook(mlp_hook, with_kwargs=True))
        handles.append(layer.register_forward_hook(layer_hook, with_kwargs=True))
    return captured, handles


def remove_hooks(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()


def label_score_details(
    tokenizer: Any,
    logits: torch.Tensor,
    label_ids: Sequence[int],
    gold_id: int,
) -> dict[str, Any]:
    values = logits[0, -1].float()
    log_probs = torch.log_softmax(values, dim=-1)
    rows = []
    for label, token_id in zip(pilot.LABELS, label_ids):
        rows.append(
            {
                "label": label,
                "token_id": int(token_id),
                "logit": float(values[token_id].item()),
                "logprob": float(log_probs[token_id].item()),
                "probability": float(torch.exp(log_probs[token_id]).item()),
            }
        )
    rows.sort(key=lambda row: row["logit"], reverse=True)
    score = pilot.score_logits(tokenizer, logits, gold_id, label_ids)
    score["candidate_rows"] = rows
    return score


@torch.inference_mode()
def summarize_all_entries_attention(
    model: Any,
    output: Any,
    captured_queries: dict[int, torch.Tensor],
    spans: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cache = base.legacy_cache(output.past_key_values)
    key_length = int(cache[0][0].shape[2])
    top20_count = min(20, key_length)
    concept_ids = [concept["concept_id"] for concept in pilot.CONCEPTS]
    layer_rows: list[dict[str, Any]] = []
    contrast_heads: list[dict[str, Any]] = []

    for layer_index, layer_cache in enumerate(cache):
        keys = layer_cache[0][0]
        queries = captured_queries[layer_index][0]
        q_heads = int(queries.shape[0])
        kv_heads = int(keys.shape[0])
        group_size = q_heads // kv_heads
        scale = float(model.model.layers[layer_index].self_attn.scaling)
        per_concept: dict[str, dict[str, list[float]]] = {
            concept_id: {
                "entry_mass": [],
                "definition_mass": [],
                "label_mass": [],
                "entry_logsumexp": [],
                "definition_logit_mean": [],
                "definition_logit_max": [],
                "label_logit": [],
                "entry_hit_top20": [],
            }
            for concept_id in concept_ids
        }
        for kv_index in range(kv_heads):
            first_head = kv_index * group_size
            q = queries[first_head : first_head + group_size].float()
            k = keys[kv_index].float()
            logits = torch.matmul(q, k.transpose(0, 1)) * scale
            probabilities = torch.softmax(logits, dim=1)
            top20_indices = torch.topk(logits, k=top20_count, dim=1).indices
            group_metrics: dict[str, dict[str, torch.Tensor]] = {}
            for concept_id in concept_ids:
                entry = tuple(spans[concept_id]["entry"])
                definition = tuple(spans[concept_id]["definition"])
                label_position = int(spans[concept_id]["label_span"][0])
                metrics = {
                    "entry_mass": probabilities[:, entry[0] : entry[1]].sum(dim=1),
                    "definition_mass": probabilities[:, definition[0] : definition[1]].sum(dim=1),
                    "label_mass": probabilities[:, label_position],
                    "entry_logsumexp": torch.logsumexp(logits[:, entry[0] : entry[1]], dim=1),
                    "definition_logit_mean": logits[:, definition[0] : definition[1]].mean(dim=1),
                    "definition_logit_max": logits[:, definition[0] : definition[1]].max(dim=1).values,
                    "label_logit": logits[:, label_position],
                    "entry_hit_top20": (
                        (top20_indices >= entry[0]) & (top20_indices < entry[1])
                    ).any(dim=1).float(),
                }
                group_metrics[concept_id] = metrics
                for name, tensor in metrics.items():
                    per_concept[concept_id][name].extend(float(value) for value in tensor.tolist())
            for local_head in range(group_size):
                target = group_metrics[TARGET_ID]
                competitor = group_metrics[COMPETITOR_ID]
                contrast_heads.append(
                    {
                        "layer": layer_index,
                        "head": first_head + local_head,
                        **{
                            f"target_{name}": float(tensor[local_head].item())
                            for name, tensor in target.items()
                        },
                        **{
                            f"competitor_{name}": float(tensor[local_head].item())
                            for name, tensor in competitor.items()
                        },
                    }
                )
        layer_rows.append(
            {
                "layer": layer_index,
                "concepts": {
                    concept_id: {
                        name: mean(values)
                        for name, values in metrics.items()
                    }
                    for concept_id, metrics in per_concept.items()
                },
            }
        )

    model_mean = {
        concept_id: {
            name: mean(layer["concepts"][concept_id][name] for layer in layer_rows)
            for name in layer_rows[0]["concepts"][concept_id]
        }
        for concept_id in concept_ids
    }
    return {
        "key_length": key_length,
        "layer_mean": layer_rows,
        "model_mean": model_mean,
        "target_competitor_head_rows": contrast_heads,
    }


@torch.inference_mode()
def residual_analysis(
    model: Any,
    captured: dict[str, Any],
    gold_id: int,
    wrong_id: int,
    exact_final_gap: float,
) -> dict[str, Any]:
    layers = list(model.model.layers)
    if captured["initial"] is None:
        raise RuntimeError("initial residual was not captured")
    if len(captured["layer_output"]) != len(layers):
        raise RuntimeError("not all layer outputs were captured")
    norm = model.model.norm
    output_weights = model.lm_head.weight
    weight_difference = output_weights[gold_id].float() - output_weights[wrong_id].float()

    def logit_lens_gap(hidden: torch.Tensor) -> float:
        normalized = norm(hidden).float()
        return float(torch.matmul(normalized, weight_difference).item())

    lens = [{"stage": "embedding", "layer": -1, "gold_minus_wrong": logit_lens_gap(captured["initial"])}]
    for layer_index in range(len(layers)):
        lens.append(
            {
                "stage": "after_layer",
                "layer": layer_index,
                "gold_minus_wrong": logit_lens_gap(captured["layer_output"][layer_index]),
            }
        )

    final_hidden = captured["layer_output"][len(layers) - 1].float()
    variance_epsilon = float(norm.variance_epsilon)
    rms_scale = torch.rsqrt(final_hidden.pow(2).mean(dim=-1, keepdim=True) + variance_epsilon)
    effective_direction = weight_difference * norm.weight.float() * rms_scale.squeeze(0)

    def direct(vector: torch.Tensor) -> float:
        return float(torch.sum(vector.float() * effective_direction).item())

    contributions = []
    for layer_index in range(len(layers)):
        contributions.append(
            {
                "layer": layer_index,
                "attention_gold_minus_wrong": direct(captured["attention"][layer_index]),
                "mlp_gold_minus_wrong": direct(captured["mlp"][layer_index]),
            }
        )
    embedding_contribution = direct(captured["initial"])
    reconstructed = embedding_contribution + sum(
        row["attention_gold_minus_wrong"] + row["mlp_gold_minus_wrong"] for row in contributions
    )
    return {
        "gold_token_id": gold_id,
        "wrong_token_id": wrong_id,
        "exact_final_gold_minus_wrong": exact_final_gap,
        "logit_lens": lens,
        "direct_logit_attribution": {
            "embedding": embedding_contribution,
            "layers": contributions,
            "reconstructed_gold_minus_wrong": reconstructed,
            "reconstruction_error_vs_exact": reconstructed - exact_final_gap,
            "attention_total": sum(row["attention_gold_minus_wrong"] for row in contributions),
            "mlp_total": sum(row["mlp_gold_minus_wrong"] for row in contributions),
        },
    }


def run_final_probe(
    model: Any,
    tokenizer: Any,
    cache: Any,
    base_length: int,
    suffix_ids: Sequence[int],
    spans: dict[str, dict[str, Any]],
    label_ids: Sequence[int],
    gold_id: int,
    wrong_id: int,
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    extend_seconds = pilot.extend_cache(model, cache, suffix_ids[:-1], base_length, chunk_size)
    prompt_len_minus_one = base_length + len(suffix_ids) - 1
    last = torch.tensor([suffix_ids[-1]], dtype=torch.long).view(1, 1)
    residuals, handles = install_residual_hooks(model)
    try:
        output, queries, final_seconds = attention_runner.capture_query_states(
            model, cache, last, prompt_len_minus_one
        )
    finally:
        remove_hooks(handles)
    scores = label_score_details(tokenizer, output.logits, label_ids, gold_id)
    exact_gap = float(
        output.logits[0, -1, gold_id].float().item()
        - output.logits[0, -1, wrong_id].float().item()
    )
    attention = summarize_all_entries_attention(model, output, queries, spans)
    residual = residual_analysis(model, residuals, gold_id, wrong_id, exact_gap)
    output.past_key_values.crop(base_length)
    del output, queries, last, residuals
    pilot.release_cuda()
    return {
        "scores": scores,
        "attention": attention,
        "residual": residual,
    }, {"extend_seconds": extend_seconds, "final_seconds": final_seconds}


def write_report(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Scraper failure structural probe",
        "",
        "All cases use a 64K body and a 4K evidence-query gap.",
        "",
        "| Filler | Query | Correct | PPL | D-G logit | Scraper entry mass | Toolbox entry mass | Scraper label mass | Toolbox label mass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        target = row["attention"]["model_mean"][TARGET_ID]
        competitor = row["attention"]["model_mean"][COMPETITOR_ID]
        exact_gap = row["residual"]["exact_final_gold_minus_wrong"]
        lines.append(
            f"| {row['filler_type']} | {row['query_mode']} | {row['scores']['candidate_correct']} | "
            f"{row['scores']['gold_ppl']:.4f} | {exact_gap:.4f} | {target['entry_mass']:.6f} | "
            f"{competitor['entry_mass']:.6f} | {target['label_mass']:.6f} | {competitor['label_mass']:.6f} |"
        )
    lines.extend(["", "## Direct-logit attribution by layer band", ""])
    lines.append("| Filler | Query | Layers | Attention D-G | MLP D-G |")
    lines.append("|---|---|---|---:|---:|")
    for row in rows:
        contributions = row["residual"]["direct_logit_attribution"]["layers"]
        for start, end in ((0, 12), (12, 24), (24, 36)):
            band = contributions[start:end]
            lines.append(
                f"| {row['filler_type']} | {row['query_mode']} | {start}-{end - 1} | "
                f"{sum(x['attention_gold_minus_wrong'] for x in band):.4f} | "
                f"{sum(x['mlp_gold_minus_wrong'] for x in band):.4f} |"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted structural probe for scraper D-vs-G failures.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--body_length", type=int, default=65536)
    parser.add_argument("--gap", type=int, default=4096)
    parser.add_argument("--filler_types", default="plain,semantic")
    parser.add_argument("--query_modes", default="lemma,paraphrase")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--prefill_chunk_size", type=int, default=128)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--original_max_position_embeddings", type=int, default=40960)
    parser.add_argument("--load_in_8bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, metadata = pilot.load_model(args)
    wrapper_prefix, wrapper_suffix = pilot.chat_wrapper(tokenizer)
    catalog = pilot.build_catalog(tokenizer, args.seed)
    label_ids = [pilot.encode(tokenizer, " " + label)[0] for label in pilot.LABELS]
    target = next(concept for concept in pilot.CONCEPTS if concept["concept_id"] == TARGET_ID)
    gold_label = catalog["mapping"][TARGET_ID]
    wrong_label = catalog["mapping"][COMPETITOR_ID]
    gold_id = pilot.encode(tokenizer, " " + gold_label)[0]
    wrong_id = pilot.encode(tokenizer, " " + wrong_label)[0]
    if (gold_label, wrong_label) != ("D", "G"):
        raise AssertionError(f"expected scraper=D and toolbox=G, got {gold_label}/{wrong_label}")
    (output_dir / "design.json").write_text(
        json.dumps(
            {
                "target": TARGET_ID,
                "competitor": COMPETITOR_ID,
                "gold_label": gold_label,
                "wrong_label": wrong_label,
                "body_length": args.body_length,
                "gap": args.gap,
                "filler_types": parse_csv_strs(args.filler_types),
                "query_modes": parse_csv_strs(args.query_modes),
                "model": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result_path = output_dir / "rows.jsonl"
    completed = {
        json.loads(line)["case_id"]
        for line in result_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if result_path.exists() else set()
    for filler_type in parse_csv_strs(args.filler_types):
        prefix, spans, catalog_start = pilot.place_catalog(
            tokenizer,
            wrapper_prefix,
            args.body_length,
            args.gap,
            filler_type,
            args.seed + (0 if filler_type == "plain" else 1000),
            catalog,
        )
        cache, prefill_seconds = pilot.prefill_mutable(model, prefix, args.prefill_chunk_size)
        print(json.dumps({"stage": "prefill", "filler_type": filler_type, "seconds": prefill_seconds}), flush=True)
        for query_mode in parse_csv_strs(args.query_modes):
            case_id = f"{filler_type}_{query_mode}_g{args.gap}"
            if case_id in completed:
                continue
            suffix = (
                pilot.encode(tokenizer, pilot.query_text(target, query_mode))
                + list(wrapper_suffix)
                + pilot.encode(tokenizer, "LABEL:")
            )
            result, timing = run_final_probe(
                model,
                tokenizer,
                cache,
                len(prefix),
                suffix,
                spans,
                label_ids,
                gold_id,
                wrong_id,
                args.prefill_chunk_size,
            )
            row = {
                "case_id": case_id,
                "filler_type": filler_type,
                "query_mode": query_mode,
                "gap": args.gap,
                "catalog_start": catalog_start,
                "prefill_seconds": prefill_seconds,
                "timing": timing,
                **result,
            }
            pilot.append_jsonl(result_path, row)
            print(
                json.dumps(
                    {
                        "case_id": case_id,
                        "correct": row["scores"]["candidate_correct"],
                        "ppl": row["scores"]["gold_ppl"],
                        "D_minus_G": row["residual"]["exact_final_gold_minus_wrong"],
                    }
                ),
                flush=True,
            )
        del cache
        pilot.release_cuda()
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    write_report(output_dir, rows)
    (output_dir / "done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")


if __name__ == "__main__":
    main()
