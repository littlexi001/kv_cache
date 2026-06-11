from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_loader import load_model_and_tokenizer  # noqa: E402
from run_top_token_category_ablation import build_prompt, load_sample  # noqa: E402


CATEGORIES = ("answer", "front", "end", "other")


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def overlapping_token_indices(
    offsets: Sequence[Tuple[int, int]], char_start: int, char_end: int
) -> List[int]:
    return [
        idx
        for idx, (token_start, token_end) in enumerate(offsets)
        if token_start < char_end and token_end > char_start
    ]


def locate_source_answer(prompt: str, answer: str, offsets: Sequence[Tuple[int, int]]) -> List[int]:
    char_start = prompt.find(answer)
    if char_start < 0:
        raise ValueError("Expected answer was not found in the context prompt.")
    return overlapping_token_indices(offsets, char_start, char_start + len(answer))


def make_causal_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    blocked_key_indices: Sequence[int],
) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    if blocked_key_indices:
        blocked = torch.tensor(blocked_key_indices, dtype=torch.long, device=device)
        mask[:, blocked] = min_value
        # Keep self-attention valid for source-answer rows while preventing later tokens from reading them.
        for idx in blocked_key_indices:
            if 0 <= idx < seq_len:
                mask[idx, idx] = 0
    return mask[None, None, :, :]


def make_decode_mask(
    total_key_length: int,
    dtype: torch.dtype,
    device: torch.device,
    blocked_key_indices: Sequence[int],
) -> torch.Tensor:
    mask = torch.zeros((1, 1, 1, total_key_length), dtype=dtype, device=device)
    if blocked_key_indices:
        blocked = torch.tensor(blocked_key_indices, dtype=torch.long, device=device)
        mask[..., blocked] = torch.finfo(dtype).min
    return mask


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def parse_conditions(value: str) -> List[str]:
    conditions = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(conditions) - set(CATEGORIES))
    if invalid:
        raise ValueError(f"Unsupported categories: {invalid}; choose from {CATEGORIES}")
    return conditions


def build_category_indices(
    prompt_length: int,
    answer_indices: Sequence[int],
    position_ratio: float,
) -> Dict[str, List[int]]:
    front_count = max(1, math.ceil(position_ratio * prompt_length))
    end_count = max(1, math.ceil(position_ratio * prompt_length))
    answer_set = set(answer_indices)
    result: Dict[str, List[int]] = {category: [] for category in CATEGORIES}
    for idx in range(prompt_length):
        if idx in answer_set:
            category = "answer"
        elif idx < front_count:
            category = "front"
        elif idx >= prompt_length - end_count:
            category = "end"
        else:
            category = "other"
        result[category].append(idx)
    return result


def random_same_size_indices(
    prompt_length: int,
    count: int,
    seed: int,
) -> List[int]:
    generator = random.Random(seed)
    return sorted(generator.sample(range(prompt_length), k=min(count, prompt_length)))


def teacher_forced_answer_metrics(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    blocked_key_indices: Sequence[int],
    device: torch.device,
) -> Dict:
    full_text = prompt + " " + answer
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded.input_ids.to(device)
    offsets = [(int(start), int(end)) for start, end in encoded.offset_mapping[0].tolist()]
    target_start_char = len(prompt) + 1
    target_indices = overlapping_token_indices(offsets, target_start_char, len(full_text))
    if not target_indices or target_indices[0] == 0:
        raise ValueError("Could not locate teacher-forced answer tokens.")

    seq_len = int(input_ids.shape[1])
    attention_mask = make_causal_mask(
        seq_len,
        model.dtype,
        device,
        blocked_key_indices,
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0]

    prediction_positions = torch.tensor([idx - 1 for idx in target_indices], device=device)
    targets = input_ids[0, torch.tensor(target_indices, device=device)]
    token_losses = F.cross_entropy(logits[prediction_positions].float(), targets, reduction="none")
    greedy_ids = logits[prediction_positions].argmax(dim=-1)
    exact_token_accuracy = float((greedy_ids == targets).float().mean().item())
    return {
        "answer_token_count": len(target_indices),
        "answer_nll": float(token_losses.mean().item()),
        "answer_perplexity": float(math.exp(min(20.0, token_losses.mean().item()))),
        "answer_token_accuracy": exact_token_accuracy,
        "teacher_forced_greedy_answer": tokenizer.decode(greedy_ids.tolist(), skip_special_tokens=True),
    }


def greedy_generate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    blocked_key_indices: Sequence[int],
    device: torch.device,
    max_new_tokens: int,
) -> str:
    ids = prompt_ids.to(device)
    generated: List[int] = []
    eos_id = tokenizer.eos_token_id
    prompt_mask = make_causal_mask(
        int(ids.shape[1]),
        model.dtype,
        device,
        blocked_key_indices,
    )
    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=prompt_mask, use_cache=True)
    for _ in range(max_new_tokens):
        next_id = int(outputs.logits[0, -1].argmax().item())
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        next_input = torch.tensor([[next_id]], dtype=ids.dtype, device=device)
        total_key_length = int(prompt_ids.shape[1]) + len(generated)
        decode_mask = make_decode_mask(
            total_key_length,
            model.dtype,
            device,
            blocked_key_indices,
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_input,
                attention_mask=decode_mask,
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-level causal ablation of answer/front/end/other context tokens.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument(
        "--data-path",
        default="ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl",
    )
    parser.add_argument("--sample-id", default="niah_len2000_depth25")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--categories", default="answer,front,end,other")
    parser.add_argument("--position-ratio", type=float, default=0.01)
    parser.add_argument("--random-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    categories = parse_conditions(args.categories)
    if not 0 < args.position_ratio < 0.5:
        raise ValueError("--position-ratio must be in (0, 0.5).")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/answer_access_causal_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(Path(args.data_path), args.sample_id)
    prompt = build_prompt(sample)
    answer = str(sample["expected_answer"])
    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    prompt_encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_offsets = [(int(start), int(end)) for start, end in prompt_encoded.offset_mapping[0].tolist()]
    source_answer_indices = locate_source_answer(prompt, answer, prompt_offsets)
    prompt_ids = prompt_encoded.input_ids
    prompt_token_count = int(prompt_ids.shape[1])
    category_indices = build_category_indices(
        prompt_token_count,
        source_answer_indices,
        args.position_ratio,
    )
    model_max = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if model_max and prompt_ids.shape[1] + args.max_new_tokens > model_max:
        raise ValueError("Prompt plus generated answer exceeds max_position_embeddings.")

    conditions: List[Tuple[str, str, List[int]]] = [("full", "none", [])]
    for category in categories:
        blocked_indices = category_indices[category]
        conditions.append((f"block_{category}", category, blocked_indices))
        if args.random_controls:
            random_indices = random_same_size_indices(
                prompt_token_count,
                len(blocked_indices),
                args.random_seed + CATEGORIES.index(category),
            )
            conditions.append((f"block_random_like_{category}", f"random_like_{category}", random_indices))

    rows: List[Dict] = []
    for condition, blocked_category, blocked_indices in conditions:
        started = time.perf_counter()
        metrics = teacher_forced_answer_metrics(
            model, tokenizer, prompt, answer, blocked_indices, device
        )
        generated = greedy_generate(
            model,
            tokenizer,
            prompt_ids,
            blocked_indices,
            device,
            args.max_new_tokens,
        )
        expected_normalized = normalize_text(answer)
        generated_normalized = normalize_text(generated)
        rows.append(
            {
                "sample_id": args.sample_id,
                "condition": condition,
                "blocked_category": blocked_category,
                "blocked_token_count": len(blocked_indices),
                "blocked_fraction": len(blocked_indices) / prompt_token_count,
                **metrics,
                "generated_answer": generated,
                "exact_match": int(generated_normalized == expected_normalized),
                "contains_answer": int(expected_normalized in generated_normalized),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    write_csv(output_dir / "answer_access_results.csv", rows)
    metadata = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "sample_id": args.sample_id,
        "device": str(device),
        "prompt_token_count": prompt_token_count,
        "source_answer_token_indices": source_answer_indices,
        "position_ratio": args.position_ratio,
        "category_token_counts": {category: len(indices) for category, indices in category_indices.items()},
        "categories": categories,
        "random_controls": args.random_controls,
        "random_seed": args.random_seed,
        "expected_answer": answer,
        "intervention": "At every layer/head, block attention to all keys in one category; blocked source rows retain self-attention.",
        "claim_boundary": "This tests category-level task causality under positional partitioning, not score-top-only sparse attention.",
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
