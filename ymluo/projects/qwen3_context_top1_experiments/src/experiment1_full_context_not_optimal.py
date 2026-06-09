from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-0.6B"


FACTS = [
    ("Arden pilot", "blue quartz", "the hidden vault passphrase"),
    ("Bexley curator", "silver maple", "the museum alarm phrase"),
    ("Caspian engineer", "orange lantern", "the bridge unlock code"),
    ("Dorian analyst", "green comet", "the archive access key"),
    ("Elara medic", "violet river", "the emergency radio phrase"),
    ("Fenton clerk", "yellow harbor", "the customs clearance token"),
    ("Galen scout", "red compass", "the expedition checkpoint code"),
    ("Helena baker", "black orchid", "the bakery safe phrase"),
    ("Iris ranger", "white pebble", "the trail gate password"),
    ("Juno teacher", "copper cloud", "the classroom cabinet key"),
    ("Kellan sailor", "amber moon", "the dock authorization phrase"),
    ("Luna archivist", "glass meadow", "the manuscript retrieval code"),
]

IRRELEVANT_TOPICS = [
    "The town council discussed bridge repairs, library hours, and the price of winter fuel.",
    "A weather report described mild winds, scattered clouds, and a chance of evening rain.",
    "The bakery catalog listed breads, jams, ceramic bowls, and seasonal gift baskets.",
    "A travel diary mentioned train delays, mountain views, and a quiet hotel lobby.",
    "The gardening manual explained soil acidity, pruning schedules, and greenhouse humidity.",
    "A music review compared violin tone, stage lighting, and the applause after the encore.",
    "The lab notebook recorded battery voltage, cable labels, and cleaning instructions.",
    "The restaurant menu described soup, roasted vegetables, and a pear dessert.",
]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic QA experiment for testing when full context is worse than selected context."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", default="outputs/experiment1_full_context_not_optimal")
    parser.add_argument("--num_samples", type=int, default=48)
    parser.add_argument("--num_irrelevant", type=int, default=8)
    parser.add_argument("--num_semantic_distractors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--do_generate", type=str2bool, default=True)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def pick_input_device(model: torch.nn.Module, fallback_device: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def make_gold_passage(subject: str, answer: str, attribute: str) -> str:
    return (
        f"Record: {subject} is associated with {attribute}. "
        f"The exact value of {attribute} is {answer}. "
        f"If asked about {attribute}, answer only with {answer}."
    )


def make_semantic_distractor(subject: str, wrong_answer: str, attribute: str, idx: int) -> str:
    return (
        f"Conflicting note {idx}: {subject} was once rumored to use {wrong_answer} as {attribute}, "
        f"but this note is unverified and should not be used as the final answer."
    )


def build_prompt(passages: list[str], subject: str, attribute: str) -> str:
    numbered = "\n\n".join(f"Passage {idx + 1}: {passage}" for idx, passage in enumerate(passages))
    return (
        "Use the passages to answer the question. Some passages may be irrelevant or explicitly unverified.\n\n"
        f"{numbered}\n\n"
        f"Question: What is the exact value of {attribute} for {subject}?\n"
        "Answer:"
    )


def make_sample(sample_id: int, rng: random.Random, num_irrelevant: int, num_semantic: int) -> dict[str, Any]:
    subject, answer, attribute = FACTS[sample_id % len(FACTS)]
    gold = make_gold_passage(subject, answer, attribute)
    wrong_answers = [fact[1] for fact in FACTS if fact[1] != answer]
    rng.shuffle(wrong_answers)
    semantic = [
        make_semantic_distractor(subject, wrong_answers[idx], attribute, idx + 1)
        for idx in range(num_semantic)
    ]
    irrelevant = [rng.choice(IRRELEVANT_TOPICS) for _ in range(num_irrelevant)]
    return {
        "sample_id": f"synthetic_{sample_id:04d}",
        "subject": subject,
        "attribute": attribute,
        "answer": answer,
        "gold": gold,
        "semantic": semantic,
        "irrelevant": irrelevant,
    }


def prompt_variants(sample: dict[str, Any], rng: random.Random) -> dict[str, str]:
    gold = sample["gold"]
    semantic = list(sample["semantic"])
    irrelevant = list(sample["irrelevant"])
    all_distractors = semantic + irrelevant
    rng.shuffle(all_distractors)

    middle = list(all_distractors)
    middle.insert(len(middle) // 2, gold)
    random_one = [rng.choice(all_distractors)] if all_distractors else []
    semantic_only = semantic[:1] if semantic else []

    variants = {
        "gold_only": [gold],
        "full_gold_begin": [gold] + all_distractors,
        "full_gold_middle": middle,
        "full_gold_end": all_distractors + [gold],
        "irrelevant_plus_gold": [gold] + irrelevant,
        "semantic_plus_gold": [gold] + semantic,
        "oracle_top_chunk": [gold],
        "random_top_chunk": random_one,
        "semantic_only_wrong": semantic_only,
    }
    return {
        name: build_prompt(passages, sample["subject"], sample["attribute"])
        for name, passages in variants.items()
        if passages
    }


@torch.inference_mode()
def answer_nll(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    answer: str,
    input_device: torch.device,
) -> dict[str, float]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(input_device)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    answer_start = prompt_ids.shape[1] - 1
    answer_logits = logits[:, answer_start : answer_start + answer_ids.shape[1], :]
    answer_labels = labels[:, answer_start : answer_start + answer_ids.shape[1]]
    losses = F.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        answer_labels.reshape(-1),
        reduction="none",
    )
    loss = float(losses.mean())
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 80.0))),
        "answer_token_count": int(answer_ids.shape[1]),
        "prompt_token_count": int(prompt_ids.shape[1]),
    }


@torch.inference_mode()
def generate_answer(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    input_device: torch.device,
) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(input_device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    requested_device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map if requested_device.type != "cpu" else None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    if requested_device.type == "cpu":
        model.to(requested_device)
    model.eval()
    input_device = pick_input_device(model, requested_device)

    sample_path = output_dir / "per_sample_results.jsonl"
    rows: list[dict[str, Any]] = []
    with sample_path.open("w", encoding="utf-8") as handle:
        for sample_idx in range(args.num_samples):
            sample = make_sample(sample_idx, rng, args.num_irrelevant, args.num_semantic_distractors)
            variants = prompt_variants(sample, rng)
            gold_loss: float | None = None
            for mode, prompt in variants.items():
                metrics = answer_nll(model, tokenizer, prompt, sample["answer"], input_device)
                if mode == "gold_only":
                    gold_loss = metrics["loss"]
                generated = ""
                exact_match = ""
                if args.do_generate:
                    generated = generate_answer(model, tokenizer, prompt, args.max_new_tokens, input_device)
                    exact_match = str(sample["answer"].lower() in generated.lower())
                row = {
                    "sample_id": sample["sample_id"],
                    "mode": mode,
                    "answer": sample["answer"],
                    "loss": metrics["loss"],
                    "ppl": metrics["ppl"],
                    "delta_loss_vs_gold_only": metrics["loss"] - gold_loss if gold_loss is not None else 0.0,
                    "prompt_token_count": metrics["prompt_token_count"],
                    "answer_token_count": metrics["answer_token_count"],
                    "generated": generated,
                    "contains_answer": exact_match,
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"finished {sample['sample_id']}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    modes = sorted({row["mode"] for row in rows})
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        mean_loss = sum(float(row["loss"]) for row in mode_rows) / len(mode_rows)
        mean_delta = sum(float(row["delta_loss_vs_gold_only"]) for row in mode_rows) / len(mode_rows)
        contains = [row["contains_answer"] for row in mode_rows if row["contains_answer"] != ""]
        acc = sum(item == "True" for item in contains) / len(contains) if contains else ""
        summary_rows.append(
            {
                "mode": mode,
                "sample_count": len(mode_rows),
                "mean_loss": mean_loss,
                "mean_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_delta_loss_vs_gold_only": mean_delta,
                "contains_answer_rate": acc,
                "mean_prompt_token_count": sum(int(row["prompt_token_count"]) for row in mode_rows)
                / len(mode_rows),
            }
        )

    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "mode",
            "sample_count",
            "mean_loss",
            "mean_ppl",
            "mean_delta_loss_vs_gold_only",
            "contains_answer_rate",
            "mean_prompt_token_count",
        ],
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    print(f"wrote {sample_path} and {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
