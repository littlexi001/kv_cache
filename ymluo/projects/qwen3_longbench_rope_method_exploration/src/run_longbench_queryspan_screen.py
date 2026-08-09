from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve()
PROJECTS = HERE.parents[2]
RULE_SRC = PROJECTS / "qwen3_local_rule_failure_boundary" / "src"
ORACLE_SRC = PROJECTS / "qwen3_longbench_oracle_evidence" / "src"
for directory in (HERE.parent, RULE_SRC, ORACLE_SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_hotpot_oracle_pilot as oracle  # noqa: E402
import run_local_rule_failure_boundary as cache_runner  # noqa: E402
import run_queryspan_prerope_retrieval_probe_8b as queryspan  # noqa: E402
import run_rope_retrieval_repair_8b as rope_repair  # noqa: E402
from run_longbench_rope_sparse import (  # noqa: E402
    align_evidence_spans,
    answer_candidates,
    read_jsonl,
    write_csv,
    write_json,
)


VARIANTS = (
    "native_noop",
    "exact_final_pre_top2_postscore",
    "queryspan_tokenmax_top2_postscore",
    "queryspan_tokenmax_blend25",
    "queryspan_tokenmax_monotone25",
)


def tokenmax_scores(anchor_queries: torch.Tensor, key_pre: torch.Tensor) -> torch.Tensor:
    groups = queryspan._gqa_groups(anchor_queries, key_pre)
    batch, heads, anchors, dim = anchor_queries.shape
    kv_heads = int(key_pre.shape[1])
    query = F.normalize(anchor_queries.float(), dim=-1).reshape(
        batch, kv_heads, groups, anchors, dim
    )
    keys = F.normalize(key_pre.float(), dim=-1)
    scores = torch.einsum("bmgad,bmkd->bmgak", query, keys)
    return scores.amax(dim=3).reshape(batch, heads, key_pre.shape[-2])[0]


def tokenmax_blend_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_value: Any | None = None,
    cache_position: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    controller = queryspan._ACTIVE_CONTROLLER
    if controller is None or controller.variant not in {
        "queryspan_tokenmax_blend25",
        "queryspan_tokenmax_monotone25",
    }:
        return queryspan.queryspan_attention_forward(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
    if int(hidden_states.shape[-2]) != 1:
        return self._queryspan_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_pre = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_key_pre = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_post, current_key_post = self._queryspan_modeling_qwen3.apply_rotary_pos_emb(
        query_pre, current_key_pre, cos.to(query_pre.device), sin.to(query_pre.device)
    )
    key_post, value = queryspan.read_only_final_query_kv(
        past_key_value, int(self.layer_idx), current_key_post, current_value
    )
    cached_pre = self._queryspan_pre_key_cache
    anchors = self._queryspan_anchor_cache
    if cached_pre is None or anchors is None:
        raise RuntimeError("query-span caches are missing")
    key_pre = torch.cat((cached_pre.to(current_key_pre.device), current_key_pre), dim=2)
    anchors = anchors.to(query_pre.device)
    scale = float(getattr(self, "scaling", 1.0 / math.sqrt(query_post.shape[-1])))
    post_scores = queryspan.runner.add_attention_mask(
        queryspan.gqa_query_key_scores(query_post, key_post, scale), attention_mask
    )
    keep_count = min(
        int(key_post.shape[-2]),
        max(1, int(math.ceil(controller.ratio * int(key_post.shape[-2])))),
    )
    semantic = tokenmax_scores(anchors, key_pre)
    selected, layout = queryspan.exact_final_pre_selection(
        semantic, keep_count, controller.local_window, controller.sink_tokens
    )
    post_selected = post_scores[0, :, 0].gather(1, selected)
    if layout.remote_end - layout.remote_start >= 2:
        semantic_remote = semantic[:, layout.remote_start : layout.remote_end].float()
        post_remote = post_scores[0, :, 0, layout.remote_start : layout.remote_end].float()
        semantic_mean = semantic_remote.mean(dim=-1, keepdim=True)
        semantic_std = semantic_remote.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        post_mean = post_remote.mean(dim=-1, keepdim=True)
        post_std = post_remote.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        calibrated = (semantic.float() - semantic_mean) / semantic_std * post_std + post_mean
    else:
        calibrated = semantic.float()
    semantic_selected = calibrated.gather(1, selected)
    remote_mask = (selected >= layout.remote_start) & (selected < layout.remote_end)
    if controller.variant == "queryspan_tokenmax_blend25":
        remote_scores = 0.75 * post_selected.float() + 0.25 * semantic_selected
    else:
        remote_scores = post_selected.float() + 0.25 * torch.relu(
            semantic_selected - post_selected.float()
        )
    merged = torch.where(remote_mask, remote_scores, post_selected.float())
    weights = F.softmax(merged.unsqueeze(0).unsqueeze(2), dim=-1).to(query_post.dtype)
    groups = queryspan._gqa_groups(query_post, value)
    selected_value = queryspan.gather_per_query_head_gqa(value, selected, groups)
    exact_reference = getattr(self, "_queryspan_exact_support", None)
    controller.record(selected, weights, layout, exact_reference, None)
    attention_output = torch.matmul(weights, selected_value)
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self.o_proj(attention_output), weights


def patch_blend_forward(model: Any) -> None:
    for module in model.modules():
        if module.__class__.__name__ == "Qwen3Attention":
            module.forward = types.MethodType(tokenmax_blend_forward, module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--longbench-jsonl", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--evidence-mapping", type=Path, required=True)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--query-anchor-count", type=int, default=16)
    parser.add_argument("--score-chunk-blocks", type=int, default=32)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16"))
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def question_token_span(
    tokenizer: Any, prompt_text: str, question: str, prefix_length: int
) -> tuple[int, int]:
    char_start = prompt_text.rfind(question)
    if char_start < 0:
        raise RuntimeError("question is not a literal substring of the frozen prompt")
    char_end = char_start + len(question)
    encoded = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    positions = [
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if end > start and start < char_end and end > char_start and index < prefix_length
    ]
    if not positions:
        raise RuntimeError("question has no aligned prefix token positions")
    return min(positions), max(positions) + 1


def append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            continue
        def mean(key: str) -> float | None:
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            return sum(values) / len(values) if values else None
        first_nll = mean("first_token_nll")
        output.append(
            {
                "variant": variant,
                "sample_count": len(selected),
                "first_token_nll": first_nll,
                "first_token_ppl": math.exp(min(first_nll, 30.0)),
                "first_token_accuracy_percent": 100.0 * mean("first_token_correct"),
                "gold_evidence_token_recall_percent": (
                    100.0 * mean("gold_evidence_token_recall")
                    if mean("gold_evidence_token_recall") is not None else None
                ),
                "gold_chain_complete_percent": (
                    100.0 * mean("gold_evidence_line_hit_rate")
                    if mean("gold_evidence_line_hit_rate") is not None else None
                ),
                "gold_evidence_attention_mass_percent": (
                    100.0 * mean("gold_evidence_attention_mass")
                    if mean("gold_evidence_attention_mass") is not None else None
                ),
                "mean_query_seconds": mean("query_seconds"),
                "budget_violation_fraction": mean("token_budget_violation_fraction"),
                "duplicate_violation_fraction": mean("duplicate_support_violation_fraction"),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("invalid shard")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_jsonl(args.frozen_manifest)
    if args.max_samples > 0:
        manifest = manifest[: args.max_samples]
    shard = [row for index, row in enumerate(manifest) if index % args.shard_count == args.shard_index]
    evidence = {row["sample_id"]: row for row in read_jsonl(args.evidence_mapping)}
    full_predictions = {
        row["sample_id"]: row
        for row in read_jsonl(args.frozen_predictions)
        if row.get("condition") == "full"
    }
    longbench = {str(row.get("_id", index)): row for index, row in enumerate(read_jsonl(args.longbench_jsonl))}
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "variants": list(VARIANTS),
        "selector_uses_gold": False,
        "metric": "first answer token only",
        "consumer": "native post-RoPE scores and original V",
    }
    write_json(args.output_dir / "config.json", config)

    load_args = argparse.Namespace(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        original_max_position_embeddings=40960,
        global_max_position=40960,
        load_in_4bit=False,
    )
    model, tokenizer = queryspan.runner.load_model(load_args)
    queryspan.patch_model(model)
    patch_blend_forward(model)
    rows_path = args.output_dir / "rows.jsonl"
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = {row["sample_id"] for row in existing}

    for item in shard:
        sample_id = str(item["sample_id"])
        if sample_id in completed:
            continue
        source = longbench[sample_id]
        question = str(source["input"])
        context = str(source["context"])
        answers = [str(value) for value in source["answers"]]
        prompt_text = oracle.chat_prompt(tokenizer, question, context)
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        if prompt_hash != full_predictions[sample_id]["prompt_sha256"]:
            raise RuntimeError(f"prompt hash mismatch: {sample_id}")
        prompt_ids, evidence_spans, alignment_audit = align_evidence_spans(
            tokenizer, prompt_text, context, evidence[sample_id]
        )
        prompt = torch.tensor(prompt_ids, dtype=torch.long).view(1, -1)
        prefix_length = int(prompt.shape[1]) - 1
        q_span = question_token_span(tokenizer, prompt_text, question, prefix_length)
        anchors = queryspan.select_query_anchor_positions(q_span, prefix_length, args.query_anchor_count)
        legacy, prefill_seconds = queryspan.capture_prefill_sequence(
            model, prompt[:, :-1], args.prefill_chunk_size, anchors
        )
        cache = cache_runner.cache_from_legacy(legacy)
        del legacy
        first_ids = answer_candidates(tokenizer, answers[0])
        first_candidates = sorted({ids[0] for ids in first_ids})
        sample_rows = []
        for variant in VARIANTS:
            controller = None
            if variant != "native_noop":
                positions = tuple(position for start, end in evidence_spans for position in range(start, end))
                controller = queryspan.QuerySpanController(
                    variant=variant,
                    ratio=args.ratio,
                    minimum_keep_tokens=0,
                    maximum_keep_tokens=0,
                    local_window=args.local_window,
                    sink_tokens=args.sink_tokens,
                    block_size=args.block_size,
                    score_chunk_blocks=args.score_chunk_blocks,
                    evaluation_positions={
                        "gold_evidence": positions,
                        "conflict_evidence": (),
                        "lexical_format_distractor": (),
                    },
                    record_spans={
                        "gold_evidence": evidence_spans,
                        "conflict_evidence": (),
                        "lexical_format_distractor": (),
                    },
                )
            cache_runner.synchronize()
            started = time.perf_counter()
            with queryspan.activate(controller), torch.inference_mode():
                output = cache_runner.forward_with_cache(
                    model,
                    prompt[:, -1:].to(cache_runner.input_device(model)),
                    cache,
                    prefix_length,
                )
            cache_runner.synchronize()
            query_seconds = time.perf_counter() - started
            logits = output.logits[:, -1, :].float()
            log_probs = F.log_softmax(logits, dim=-1)
            losses = [-float(log_probs[0, token_id].item()) for token_id in first_candidates]
            metrics = controller.metrics.summary() if controller is not None else {}
            invalid = {
                key: value
                for key, value in metrics.items()
                if key.endswith("violation_fraction") and float(value) != 0.0
            }
            if invalid:
                raise RuntimeError(f"support audit failed: {sample_id}/{variant}: {invalid}")
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "variant": variant,
                    "question": question,
                    "answers": answers,
                    "prompt_tokens": len(prompt_ids),
                    "prompt_sha256": prompt_hash,
                    "question_token_span": list(q_span),
                    "anchor_positions": anchors,
                    "anchor_count": len(anchors),
                    "aligned_evidence_spans": [list(span) for span in evidence_spans],
                    "alignment_audit": alignment_audit,
                    "first_token_nll": min(losses),
                    "first_token_ppl": math.exp(min(min(losses), 30.0)),
                    "first_token_correct": int(int(logits.argmax(dim=-1).item()) in first_candidates),
                    "prediction_token_id": int(logits.argmax(dim=-1).item()),
                    "prediction_token_text": tokenizer.decode([int(logits.argmax(dim=-1).item())]),
                    "prefill_seconds": prefill_seconds,
                    "query_seconds": query_seconds,
                    **metrics,
                }
            )
            rope_repair.reset_dynamic_cache(cache, prefix_length)
            print(
                f"sample={sample_id[:8]} {variant} nll={min(losses):.4f} "
                f"recall={sample_rows[-1].get('gold_evidence_token_recall')}",
                flush=True,
            )
        append_rows(rows_path, sample_rows)
        all_rows = read_jsonl(rows_path)
        write_csv(args.output_dir / "rows.csv", all_rows)
        write_csv(args.output_dir / "summary.csv", summarize(all_rows))
        del cache, prompt
        queryspan.clear_allocator()

    all_rows = read_jsonl(rows_path)
    summary = summarize(all_rows)
    write_csv(args.output_dir / "rows.csv", all_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
