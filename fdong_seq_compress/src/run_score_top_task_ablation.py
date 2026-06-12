from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_loader import load_model_and_tokenizer  # noqa: E402
from run_answer_access_causal import (  # noqa: E402
    build_category_indices,
    locate_source_answer,
    normalize_text,
    overlapping_token_indices,
)
from run_top_token_category_ablation import build_prompt, load_sample  # noqa: E402


CATEGORIES = ("answer", "front", "end", "other")


def parse_categories(value: str) -> List[str]:
    categories = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(categories) - set(CATEGORIES))
    if invalid:
        raise ValueError(f"Unsupported categories: {invalid}; choose from {CATEGORIES}")
    return categories


class OracleTopAttention:
    def __init__(self, ratio: float, category_indices: Dict[str, List[int]]) -> None:
        self.ratio = ratio
        self.category_indices = category_indices
        self.enabled = False
        self.excluded_category: str | None = None
        self.selected_count = 0
        self.valid_count = 0
        self.call_count = 0
        self.record_union_stats = False
        self.phase = ""
        self.decode_step = -1
        self.union_rows: List[Dict] = []
        self.cumulative_layer_unions: Dict[int, set[int]] = {}
        self.cumulative_kv_unions: Dict[Tuple[int, int], set[int]] = {}

    def reset(self, enabled: bool, excluded_category: str | None) -> None:
        self.enabled = enabled
        self.excluded_category = excluded_category
        self.selected_count = 0
        self.valid_count = 0
        self.call_count = 0
        self.record_union_stats = False
        self.phase = ""
        self.decode_step = -1
        self.union_rows = []
        self.cumulative_layer_unions = {}
        self.cumulative_kv_unions = {}

    def set_recording(self, enabled: bool, phase: str = "", decode_step: int = -1) -> None:
        self.record_union_stats = enabled
        self.phase = phase
        self.decode_step = decode_step

    def record_selected_unions(self, module, selected: torch.Tensor, valid: torch.Tensor) -> None:
        # Record the current query only: prompt's final query or one decode query.
        selected_now = selected[0, :, -1].detach().cpu()
        valid_now = valid[0, :, -1].detach().cpu()
        visible_count = int(valid_now[0].sum().item())
        if visible_count <= 0:
            return

        num_q_heads = int(selected_now.shape[0])
        num_groups = int(module.num_key_value_groups)
        num_kv_heads = num_q_heads // num_groups
        q_head_counts = selected_now.sum(dim=-1).float()
        kv_union_counts: List[int] = []
        layer_idx = int(module.layer_idx)
        for kv_head_idx in range(num_kv_heads):
            start = kv_head_idx * num_groups
            end = start + num_groups
            union_mask = selected_now[start:end].any(dim=0)
            union_indices = set(torch.nonzero(union_mask, as_tuple=False).flatten().tolist())
            kv_union_counts.append(len(union_indices))
            self.cumulative_kv_unions.setdefault((layer_idx, kv_head_idx), set()).update(union_indices)

        layer_union_mask = selected_now.any(dim=0)
        layer_union_indices = set(torch.nonzero(layer_union_mask, as_tuple=False).flatten().tolist())
        self.cumulative_layer_unions.setdefault(layer_idx, set()).update(layer_union_indices)
        cumulative_kv_counts = [
            len(self.cumulative_kv_unions[(layer_idx, kv_head_idx)])
            for kv_head_idx in range(num_kv_heads)
        ]
        self.union_rows.append(
            {
                "phase": self.phase,
                "decode_step": self.decode_step,
                "layer": layer_idx,
                "key_length": int(selected_now.shape[-1]),
                "visible_count": visible_count,
                "q_head_selected_fraction_mean": float((q_head_counts / visible_count).mean().item()),
                "q_head_selected_count_mean": float(q_head_counts.mean().item()),
                "kv_head_union_fraction_mean": sum(kv_union_counts) / (num_kv_heads * visible_count),
                "kv_head_union_count_mean": sum(kv_union_counts) / num_kv_heads,
                "kv_head_union_count_min": min(kv_union_counts),
                "kv_head_union_count_max": max(kv_union_counts),
                "layer_shared_mask_union_fraction": len(layer_union_indices) / visible_count,
                "layer_shared_mask_union_count": len(layer_union_indices),
                "cumulative_kv_head_union_fraction_mean": (
                    sum(cumulative_kv_counts) / (num_kv_heads * visible_count)
                ),
                "cumulative_layer_union_fraction": (
                    len(self.cumulative_layer_unions[layer_idx]) / visible_count
                ),
            }
        )

    def forward(
        self,
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            scores = scores + causal_mask
            valid = causal_mask == 0
            valid = valid.expand(scores.shape[0], scores.shape[1], scores.shape[2], scores.shape[3])
        else:
            valid = torch.ones_like(scores, dtype=torch.bool)

        if self.enabled:
            valid_counts = valid.sum(dim=-1)
            keep_counts = torch.ceil(valid_counts.float() * self.ratio).long().clamp_min(1)
            max_keep = min(int(keep_counts.max().item()), scores.shape[-1])
            ranked = torch.topk(scores.masked_fill(~valid, torch.finfo(scores.dtype).min), k=max_keep, dim=-1).indices
            ranks = torch.arange(max_keep, device=scores.device).view(1, 1, 1, -1)
            rank_valid = ranks < keep_counts.unsqueeze(-1)
            selected = torch.zeros_like(valid)
            selected.scatter_(-1, ranked, rank_valid)
            selected &= valid

            eligible = valid.clone()
            if self.excluded_category is not None:
                excluded = [
                    idx
                    for idx in self.category_indices[self.excluded_category]
                    if idx < scores.shape[-1]
                ]
                if excluded:
                    excluded_tensor = torch.tensor(excluded, dtype=torch.long, device=scores.device)
                    selected[..., excluded_tensor] = False
                    eligible[..., excluded_tensor] = False

            # Avoid an undefined softmax if an early query loses its only selected key.
            empty = ~selected.any(dim=-1, keepdim=True)
            fallback_scores = scores.masked_fill(~eligible, torch.finfo(scores.dtype).min)
            fallback_idx = fallback_scores.argmax(dim=-1, keepdim=True)
            fallback = torch.zeros_like(selected).scatter_(-1, fallback_idx, True) & eligible
            no_eligible = ~eligible.any(dim=-1, keepdim=True)
            original_fallback = torch.zeros_like(selected).scatter_(-1, ranked[..., :1], True) & valid
            fallback = torch.where(no_eligible, original_fallback, fallback)
            selected = torch.where(empty, fallback, selected)
            scores = scores.masked_fill(~selected, torch.finfo(scores.dtype).min)
            self.selected_count += int(selected.sum().item())
            self.valid_count += int(valid.sum().item())
            self.call_count += 1
            if self.record_union_stats:
                self.record_selected_unions(module, selected, valid)

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=dropout, training=module.training)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return output, weights


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def answer_metrics(model, tokenizer, prompt: str, answer: str, device: torch.device) -> Dict:
    full_text = prompt + " " + answer
    encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded.input_ids.to(device)
    offsets = [(int(a), int(b)) for a, b in encoded.offset_mapping[0].tolist()]
    target_indices = overlapping_token_indices(offsets, len(prompt) + 1, len(full_text))
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0]
    positions = torch.tensor([idx - 1 for idx in target_indices], device=device)
    targets = ids[0, torch.tensor(target_indices, device=device)]
    losses = F.cross_entropy(logits[positions].float(), targets, reduction="none")
    predictions = logits[positions].argmax(dim=-1)
    return {
        "answer_nll": float(losses.mean().item()),
        "answer_perplexity": float(math.exp(min(20.0, losses.mean().item()))),
        "answer_token_accuracy": float((predictions == targets).float().mean().item()),
        "teacher_forced_greedy_answer": tokenizer.decode(predictions.tolist(), skip_special_tokens=True),
    }


def greedy_generate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    controller: OracleTopAttention,
) -> str:
    ids = prompt_ids.to(device)
    generated: List[int] = []
    eos_id = tokenizer.eos_token_id
    controller.set_recording(controller.enabled, phase="prompt_last_query", decode_step=-1)
    with torch.no_grad():
        outputs = model(input_ids=ids, use_cache=True)
    for step in range(max_new_tokens):
        next_id = int(outputs.logits[0, -1].argmax().item())
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        next_input = torch.tensor([[next_id]], dtype=ids.dtype, device=device)
        controller.set_recording(controller.enabled, phase="decode", decode_step=step)
        with torch.no_grad():
            outputs = model(
                input_ids=next_input,
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end oracle score-top attention and category ablations.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument(
        "--data-path",
        default="ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl",
    )
    parser.add_argument("--sample-id", default="niah_len2000_depth25")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--score-ratio", type=float, default=0.01)
    parser.add_argument("--position-ratio", type=float, default=0.01)
    parser.add_argument("--excluded-categories", default="answer,front,end,other")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if not 0 < args.score_ratio <= 1:
        raise ValueError("--score-ratio must be in (0, 1].")

    excluded_categories = parse_categories(args.excluded_categories)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/score_top_task_ablation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = load_sample(Path(args.data_path), args.sample_id)
    prompt = build_prompt(sample)
    answer = str(sample["expected_answer"])
    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path, device=args.device, dtype=args.dtype, attn_implementation="eager"
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = encoded.input_ids
    offsets = [(int(a), int(b)) for a, b in encoded.offset_mapping[0].tolist()]
    answer_indices = locate_source_answer(prompt, answer, offsets)
    category_indices = build_category_indices(int(prompt_ids.shape[1]), answer_indices, args.position_ratio)
    controller = OracleTopAttention(args.score_ratio, category_indices)
    original_forward = modeling_qwen3.eager_attention_forward
    modeling_qwen3.eager_attention_forward = controller.forward

    rows: List[Dict] = []
    conditions: List[Tuple[str, bool, str | None]] = [
        ("full", False, None),
        ("score_top_all", True, None),
        *[(f"score_top_without_{category}", True, category) for category in excluded_categories],
    ]
    try:
        for condition, enabled, excluded in conditions:
            controller.reset(enabled, excluded)
            started = time.perf_counter()
            controller.set_recording(False)
            metrics = answer_metrics(model, tokenizer, prompt, answer, device)
            generated = greedy_generate(
                model, tokenizer, prompt_ids, device, args.max_new_tokens, controller
            )
            expected_norm = normalize_text(answer)
            generated_norm = normalize_text(generated)
            row = {
                "sample_id": args.sample_id,
                "condition": condition,
                "excluded_category": excluded or "none",
                "score_ratio": args.score_ratio if enabled else 1.0,
                "effective_selected_fraction": (
                    controller.selected_count / controller.valid_count if controller.valid_count else 1.0
                ),
                **metrics,
                "generated_answer": generated,
                "exact_match": int(generated_norm == expected_norm),
                "contains_answer": int(expected_norm in generated_norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if enabled:
                write_csv(output_dir / f"head_union_stats_{condition}.csv", controller.union_rows)
    finally:
        modeling_qwen3.eager_attention_forward = original_forward

    write_csv(output_dir / "score_top_task_results.csv", rows)
    metadata = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "sample_id": args.sample_id,
        "device": str(device),
        "prompt_token_count": int(prompt_ids.shape[1]),
        "score_ratio": args.score_ratio,
        "position_ratio": args.position_ratio,
        "excluded_categories": excluded_categories,
        "category_token_counts": {key: len(value) for key, value in category_indices.items()},
        "expected_answer": answer,
        "definition": "Each layer/head/query keeps its oracle score-top ratio, then optionally removes one positional/semantic category.",
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
