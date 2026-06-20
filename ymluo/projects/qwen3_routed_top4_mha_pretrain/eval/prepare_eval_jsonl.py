from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

def load_dataset_checked(*args: Any, **kwargs: Any) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install with: pip install datasets") from exc
    return load_dataset(*args, **kwargs)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(rows: list[dict[str, Any]], max_examples: int, seed: int) -> list[dict[str, Any]]:
    if max_examples <= 0 or len(rows) <= max_examples:
        return rows
    rng = random.Random(seed)
    picked = list(rows)
    rng.shuffle(picked)
    return picked[:max_examples]


def prepare_piqa(split: str) -> list[dict[str, Any]]:
    dataset = load_dataset_checked("piqa", split=split)
    rows = []
    for item in dataset:
        rows.append(
            {
                "id": str(item.get("id", len(rows))),
                "prompt": f"Question: {item['goal']}\nAnswer:",
                "choices": [item["sol1"], item["sol2"]],
                "answer": int(item["label"]),
            }
        )
    return rows


def prepare_hellaswag(split: str) -> list[dict[str, Any]]:
    dataset = load_dataset_checked("hellaswag", split=split)
    rows = []
    for item in dataset:
        rows.append(
            {
                "id": str(item.get("ind", len(rows))),
                "prompt": f"{item['ctx'].strip()}\nThe most likely continuation is:",
                "choices": [str(ending) for ending in item["endings"]],
                "answer": int(item["label"]),
            }
        )
    return rows


def prepare_winogrande(split: str) -> list[dict[str, Any]]:
    dataset = load_dataset_checked("winogrande", "winogrande_xl", split=split)
    rows = []
    for item in dataset:
        sentence = item["sentence"].replace("_", "_____")
        rows.append(
            {
                "id": str(item.get("qID", len(rows))),
                "prompt": f"Fill in the blank with the best option.\nSentence: {sentence}\nAnswer:",
                "choices": [item["option1"], item["option2"]],
                "answer": int(item["answer"]) - 1,
            }
        )
    return rows


def prepare_arc(config_name: str, split: str) -> list[dict[str, Any]]:
    dataset = load_dataset_checked("ai2_arc", config_name, split=split)
    rows = []
    for item in dataset:
        labels = [str(label) for label in item["choices"]["label"]]
        choices = [str(text) for text in item["choices"]["text"]]
        answer_key = str(item["answerKey"])
        if answer_key not in labels:
            continue
        rows.append(
            {
                "id": str(item.get("id", len(rows))),
                "prompt": f"Question: {item['question']}\nAnswer:",
                "choices": choices,
                "answer": labels.index(answer_key),
            }
        )
    return rows


def prepare_boolq(split: str) -> list[dict[str, Any]]:
    dataset = load_dataset_checked("boolq", split=split)
    rows = []
    for item in dataset:
        rows.append(
            {
                "id": str(len(rows)),
                "prompt": f"Passage: {item['passage']}\nQuestion: {item['question']}\nAnswer:",
                "choices": ["No", "Yes"],
                "answer": 1 if bool(item["answer"]) else 0,
            }
        )
    return rows


TASK_BUILDERS: dict[str, Callable[[str], list[dict[str, Any]]]] = {
    "piqa": prepare_piqa,
    "hellaswag": prepare_hellaswag,
    "winogrande": prepare_winogrande,
    "arc_easy": lambda split: prepare_arc("ARC-Easy", split),
    "arc_challenge": lambda split: prepare_arc("ARC-Challenge", split),
    "boolq": prepare_boolq,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output/eval_data")
    parser.add_argument("--tasks", default="piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    manifest = []
    for task in tasks:
        if task not in TASK_BUILDERS:
            raise ValueError(f"Unknown task {task!r}. Available: {sorted(TASK_BUILDERS)}")
        rows = TASK_BUILDERS[task](args.split)
        rows = sample_rows(rows, args.max_examples, args.seed)
        output_path = output_dir / f"{task}_{args.split}.jsonl"
        write_jsonl(output_path, rows)
        manifest.append({"task": task, "split": args.split, "examples": len(rows), "path": str(output_path)})
        print(f"wrote {len(rows)} examples: {output_path}", flush=True)
    write_jsonl(output_dir / "manifest.jsonl", manifest)


if __name__ == "__main__":
    main()
