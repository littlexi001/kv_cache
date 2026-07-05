from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_full_information_stress_eval import StressCase, build_stress_cases  # noqa: E402
from run_static_summary_ppl_speed import content_words, word_tokens  # noqa: E402
from run_task_adaptive_memory_policy_eval import (  # noqa: E402
    Config as OracleConfig,
    GenerationCase,
    build_generation_cases,
    load_token_ids,
)


FEATURE_NAMES = [
    "task_is_generation",
    "task_is_exact",
    "query_len_chars",
    "query_len_words",
    "query_has_question_mark",
    "query_exact_keyword_count",
    "query_quote_keyword_count",
    "query_count_keyword_count",
    "query_list_keyword_count",
    "query_compare_keyword_count",
    "query_number_count",
    "query_all_keyword",
    "prefix_tokens",
    "older_tokens",
    "recent_tokens",
    "block_tokens",
    "num_older_blocks",
    "summary10_words",
    "summary100_words",
    "summary1000_words",
    "retriever_top1_overlap",
    "retriever_top2_overlap",
    "retriever_score_gap",
    "retriever_positive_blocks",
    "retriever_top1_norm",
    "retriever_gap_norm",
    "prefix_unique_word_ratio",
    "prefix_number_count_log",
    "prefix_capitalized_count_log",
]


@dataclass(frozen=True)
class Config:
    oracle_output_dir: str
    output_dir: str
    model_name_or_path: str
    text_paths: tuple[str, ...]
    dataset_names: tuple[str, ...]
    samples_per_dataset: int
    sample_stride_tokens: int
    prefill_tokens: int
    eval_tokens: int
    block_tokens: int
    recent_tokens: int
    summary10_words: int
    summary100_words: int
    summary1000_words: int
    max_text_tokens: int
    eval_start_tokens: int
    num_choices: int
    generation_nll_slack: float
    learned_summary_train_tokens: int
    learned_summary_epochs: int
    learned_summary_hidden_dim: int
    learned_summary_lr: float
    learned_summary_max_sentences: int
    learned_summary_seed: int
    seed: int
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    test_fraction: float

    @property
    def generation_methods(self) -> tuple[str, ...]:
        return ("summary10", "summary100", "summary1000", "static_hier", "full_raw")

    @property
    def exact_methods(self) -> tuple[str, ...]:
        return ("summary10", "summary100", "summary1000", "static_hier", "retrieval_raw_k1", "retrieval_raw_k2", "full_raw")


@dataclass
class RouterExample:
    task_family: str
    dataset: str
    sample_id: int
    start_token: int
    label: str
    features: list[float]


@dataclass
class PredictionRow:
    split: str
    task_family: str
    dataset: str
    sample_id: int
    start_token: int
    oracle_label: str
    predicted_label: str
    label_correct: int
    method_success: int
    token_ratio_vs_full_raw: float
    forward_seconds: float
    prompt_tokens: int


class TinyMemoryRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_tuple_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def config_from_oracle_output(args: argparse.Namespace) -> Config:
    oracle_dir = Path(args.oracle_output_dir)
    payload = json.loads((oracle_dir / "summary.json").read_text(encoding="utf-8"))
    source_config = payload["config"]
    text_paths = parse_tuple_csv(args.text_paths) if args.text_paths else tuple(source_config["text_paths"])
    dataset_names = parse_tuple_csv(args.dataset_names) if args.dataset_names else tuple(source_config["dataset_names"])
    return Config(
        oracle_output_dir=args.oracle_output_dir,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path or source_config["model_name_or_path"],
        text_paths=text_paths,
        dataset_names=dataset_names,
        samples_per_dataset=int(source_config["samples_per_dataset"]),
        sample_stride_tokens=int(source_config["sample_stride_tokens"]),
        prefill_tokens=int(source_config["prefill_tokens"]),
        eval_tokens=int(source_config["eval_tokens"]),
        block_tokens=int(source_config["block_tokens"]),
        recent_tokens=int(source_config["recent_tokens"]),
        summary10_words=int(source_config["summary10_words"]),
        summary100_words=int(source_config["summary100_words"]),
        summary1000_words=int(source_config["summary1000_words"]),
        max_text_tokens=int(source_config["max_text_tokens"]),
        eval_start_tokens=int(source_config["eval_start_tokens"]),
        num_choices=int(source_config["num_choices"]),
        generation_nll_slack=float(source_config["generation_nll_slack"]),
        learned_summary_train_tokens=int(source_config["learned_summary_train_tokens"]),
        learned_summary_epochs=int(source_config["learned_summary_epochs"]),
        learned_summary_hidden_dim=int(source_config["learned_summary_hidden_dim"]),
        learned_summary_lr=float(source_config["learned_summary_lr"]),
        learned_summary_max_sentences=int(source_config["learned_summary_max_sentences"]),
        learned_summary_seed=int(source_config["learned_summary_seed"]),
        seed=int(source_config["seed"]),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_fraction=args.test_fraction,
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Distill oracle memory-policy labels into a tiny inference router.")
    parser.add_argument("--oracle_output_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--text_paths", default="")
    parser.add_argument("--dataset_names", default="")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.35)
    args = parser.parse_args()
    return config_from_oracle_output(args)


def as_oracle_config(config: Config) -> OracleConfig:
    return OracleConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        adapter_path="",
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        generation_methods=config.generation_methods,
        exact_methods=config.exact_methods,
        samples_per_dataset=config.samples_per_dataset,
        sample_stride_tokens=config.sample_stride_tokens,
        prefill_tokens=config.prefill_tokens,
        eval_tokens=config.eval_tokens,
        block_tokens=config.block_tokens,
        recent_tokens=config.recent_tokens,
        summary10_words=config.summary10_words,
        summary100_words=config.summary100_words,
        summary1000_words=config.summary1000_words,
        max_text_tokens=config.max_text_tokens,
        eval_start_tokens=config.eval_start_tokens,
        num_choices=config.num_choices,
        generation_nll_slack=config.generation_nll_slack,
        device="cpu",
        dtype="float16",
        attn_implementation="sdpa",
        learned_summary_train_tokens=config.learned_summary_train_tokens,
        learned_summary_epochs=config.learned_summary_epochs,
        learned_summary_hidden_dim=config.learned_summary_hidden_dim,
        learned_summary_lr=config.learned_summary_lr,
        learned_summary_max_sentences=config.learned_summary_max_sentences,
        learned_summary_seed=config.learned_summary_seed,
        seed=config.seed,
    )


def keyword_count(text: str, words: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(1 for word in words if word in lower)


def block_texts_for_prefix(tokenizer: Any, prefix_ids: tuple[int, ...], config: Config) -> list[str]:
    recent_len = min(config.recent_tokens, len(prefix_ids))
    older_ids = list(prefix_ids[: max(0, len(prefix_ids) - recent_len)])
    return [
        tokenizer.decode(older_ids[idx : idx + config.block_tokens], skip_special_tokens=True)
        for idx in range(0, len(older_ids), config.block_tokens)
    ]


def lexical_retriever_features(query: str, block_texts: list[str]) -> tuple[float, float, float, float, float, float]:
    query_terms = set(content_words(query))
    if not query_terms or not block_texts:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    scores = []
    for text in block_texts:
        scores.append(float(len(query_terms & set(content_words(text)))))
    scores.sort(reverse=True)
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    gap = top1 - top2
    positive = float(sum(1 for score in scores if score > 0))
    denom = max(1.0, float(len(query_terms)))
    return top1, top2, gap, positive, top1 / denom, gap / denom


def prefix_text_features(text: str) -> tuple[float, float, float]:
    words = word_tokens(text)
    if not words:
        return 0.0, 0.0, 0.0
    unique_ratio = len({word.lower() for word in words}) / max(1, len(words))
    number_count = len(re.findall(r"\d+", text))
    capitalized_count = len(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
    return unique_ratio, math.log1p(number_count), math.log1p(capitalized_count)


def features_for_case(
    tokenizer: Any,
    case: GenerationCase | StressCase,
    config: Config,
) -> list[float]:
    is_exact = isinstance(case, StressCase)
    query = case.question if is_exact else ""
    prefix_ids = case.prefix_ids
    recent_len = min(config.recent_tokens, len(prefix_ids))
    older_tokens = max(0, len(prefix_ids) - recent_len)
    num_older_blocks = math.ceil(older_tokens / config.block_tokens) if config.block_tokens else 0
    block_texts = block_texts_for_prefix(tokenizer, prefix_ids, config)
    top1, top2, gap, positive, top1_norm, gap_norm = lexical_retriever_features(query, block_texts)
    prefix_sample_text = tokenizer.decode(list(prefix_ids[: min(len(prefix_ids), 2048)]), skip_special_tokens=True)
    unique_ratio, number_count_log, capitalized_count_log = prefix_text_features(prefix_sample_text)
    query_words = word_tokens(query)
    return [
        0.0 if is_exact else 1.0,
        1.0 if is_exact else 0.0,
        float(len(query)),
        float(len(query_words)),
        1.0 if "?" in query else 0.0,
        float(keyword_count(query, ("exact", "code", "access", "private", "value", "answer with only"))),
        float(keyword_count(query, ("quote", "verbatim", "span", "sentence"))),
        float(keyword_count(query, ("count", "how many", "number of", "total"))),
        float(keyword_count(query, ("list", "all", "every", "enumerate"))),
        float(keyword_count(query, ("compare", "contrast", "difference", "before", "after"))),
        float(len(re.findall(r"\d+", query))),
        1.0 if re.search(r"\ball\b", query.lower()) else 0.0,
        float(len(prefix_ids)),
        float(older_tokens),
        float(recent_len),
        float(config.block_tokens),
        float(num_older_blocks),
        float(config.summary10_words),
        float(config.summary100_words),
        float(config.summary1000_words),
        top1,
        top2,
        gap,
        positive,
        top1_norm,
        gap_norm,
        unique_ratio,
        number_count_log,
        capitalized_count_log,
    ]


def key_for(task_family: str, dataset: str, sample_id: str | int, start_token: str | int) -> tuple[str, str, int, int]:
    return (task_family, dataset, int(sample_id), int(start_token))


def build_examples(tokenizer: Any, config: Config) -> tuple[list[RouterExample], dict[tuple[str, str, int, int, str], dict[str, str]]]:
    oracle_dir = Path(config.oracle_output_dir)
    oracle_rows = read_csv_rows(oracle_dir / "oracle_rows.csv")
    method_rows = read_csv_rows(oracle_dir / "method_rows.csv")
    label_by_key = {
        key_for(row["task_family"], row["dataset"], row["sample_id"], row["start_token"]): row["selected_method"]
        for row in oracle_rows
    }
    method_lookup = {
        (*key_for(row["task_family"], row["dataset"], row["sample_id"], row["start_token"]), row["method"]): row
        for row in method_rows
    }

    oracle_config = as_oracle_config(config)
    token_ids_by_dataset = load_token_ids(tokenizer, oracle_config)
    generation_cases = build_generation_cases(token_ids_by_dataset, oracle_config)
    exact_cases = build_stress_cases(tokenizer, token_ids_by_dataset, oracle_config)

    examples: list[RouterExample] = []
    for case in generation_cases:
        lookup_key = key_for("generation", case.dataset, case.sample_id, case.start_token)
        if lookup_key not in label_by_key:
            continue
        examples.append(
            RouterExample(
                task_family="generation",
                dataset=case.dataset,
                sample_id=case.sample_id,
                start_token=case.start_token,
                label=label_by_key[lookup_key],
                features=features_for_case(tokenizer, case, config),
            )
        )
    for case in exact_cases:
        lookup_key = key_for("exact", case.dataset, case.sample_id, case.start_token)
        if lookup_key not in label_by_key:
            continue
        examples.append(
            RouterExample(
                task_family="exact",
                dataset=case.dataset,
                sample_id=case.sample_id,
                start_token=case.start_token,
                label=label_by_key[lookup_key],
                features=features_for_case(tokenizer, case, config),
            )
        )
    if not examples:
        raise ValueError("no router examples were matched from oracle output")
    return examples, method_lookup


def split_examples(examples: list[RouterExample], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    by_label: dict[str, list[int]] = {}
    for idx, example in enumerate(examples):
        by_label.setdefault(example.label, []).append(idx)
    train: list[int] = []
    test: list[int] = []
    for _, indices in sorted(by_label.items()):
        rng.shuffle(indices)
        if len(indices) == 1:
            train.extend(indices)
            continue
        test_count = max(1, int(round(len(indices) * config.test_fraction)))
        test_count = min(test_count, len(indices) - 1)
        test.extend(indices[:test_count])
        train.extend(indices[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def normalize_features(examples: list[RouterExample], train_indices: list[int]) -> tuple[list[float], list[float]]:
    dim = len(examples[0].features)
    mean = []
    std = []
    for col in range(dim):
        values = [examples[idx].features[col] for idx in train_indices]
        col_mean = sum(values) / max(1, len(values))
        variance = sum((value - col_mean) ** 2 for value in values) / max(1, len(values))
        mean.append(col_mean)
        std.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
    return mean, std


def tensorize(
    examples: list[RouterExample],
    indices: list[int],
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for idx in indices:
        row = examples[idx]
        xs.append([(value - mean[col]) / max(std[col], 1e-6) for col, value in enumerate(row.features)])
        ys.append(label_to_id[row.label])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def train_router(
    examples: list[RouterExample],
    train_indices: list[int],
    test_indices: list[int],
    config: Config,
) -> tuple[TinyMemoryRouter, dict[str, int], list[float], list[float], list[dict[str, Any]]]:
    label_names = sorted({example.label for example in examples})
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    mean, std = normalize_features(examples, train_indices)
    train_x, train_y = tensorize(examples, train_indices, label_to_id, mean, std)
    test_x, test_y = tensorize(examples, test_indices, label_to_id, mean, std) if test_indices else (train_x, train_y)

    torch.manual_seed(config.seed)
    model = TinyMemoryRouter(input_dim=train_x.shape[1], hidden_dim=config.hidden_dim, output_dim=len(label_names))
    counts = torch.bincount(train_y, minlength=len(label_names)).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    weights = weights / weights.mean().clamp_min(1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y, weight=weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.inference_mode():
                train_acc = float((model(train_x).argmax(dim=-1) == train_y).float().mean())
                test_acc = float((model(test_x).argmax(dim=-1) == test_y).float().mean()) if len(test_y) else 0.0
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "train_label_accuracy": train_acc,
                    "test_label_accuracy": test_acc,
                }
            )
    return model, label_to_id, mean, std, history


def evaluate_predictions(
    model: TinyMemoryRouter,
    examples: list[RouterExample],
    split_name: str,
    indices: list[int],
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    method_lookup: dict[tuple[str, str, int, int, str], dict[str, str]],
) -> list[PredictionRow]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    x, _ = tensorize(examples, indices, label_to_id, mean, std)
    model.eval()
    with torch.inference_mode():
        predictions = model(x).argmax(dim=-1).tolist()
    rows: list[PredictionRow] = []
    for local_idx, example_idx in enumerate(indices):
        example = examples[example_idx]
        predicted = id_to_label[int(predictions[local_idx])]
        method_row = method_lookup.get(
            key_for(example.task_family, example.dataset, example.sample_id, example.start_token) + (predicted,)
        )
        if method_row is None:
            method_success = 0
            token_ratio = 1.0
            seconds = 0.0
            prompt_tokens = 0
        else:
            method_success = int(float(method_row["success"]))
            token_ratio = float(method_row["token_ratio_vs_full_raw"])
            seconds = float(method_row["forward_seconds"])
            prompt_tokens = int(float(method_row["prompt_tokens"]))
        rows.append(
            PredictionRow(
                split=split_name,
                task_family=example.task_family,
                dataset=example.dataset,
                sample_id=example.sample_id,
                start_token=example.start_token,
                oracle_label=example.label,
                predicted_label=predicted,
                label_correct=int(predicted == example.label),
                method_success=method_success,
                token_ratio_vs_full_raw=token_ratio,
                forward_seconds=seconds,
                prompt_tokens=prompt_tokens,
            )
        )
    return rows


def summarize_prediction_rows(rows: list[PredictionRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[PredictionRow]] = {}
    for row in rows:
        grouped.setdefault((row.split, row.task_family), []).append(row)
        grouped.setdefault((row.split, "__overall__"), []).append(row)
    out: list[dict[str, Any]] = []
    for (split, task_family), items in sorted(grouped.items()):
        selections: dict[str, int] = {}
        for item in items:
            selections[item.predicted_label] = selections.get(item.predicted_label, 0) + 1
        payload: dict[str, Any] = {
            "split": split,
            "task_family": task_family,
            "samples": len(items),
            "oracle_label_accuracy": sum(item.label_correct for item in items) / len(items),
            "routed_success_rate": sum(item.method_success for item in items) / len(items),
            "avg_token_ratio_vs_full_raw": sum(item.token_ratio_vs_full_raw for item in items) / len(items),
            "avg_forward_seconds": sum(item.forward_seconds for item in items) / len(items),
            "avg_prompt_tokens": sum(item.prompt_tokens for item in items) / len(items),
        }
        for label, count in sorted(selections.items()):
            payload[f"select_{label}"] = count
            payload[f"select_{label}_rate"] = count / len(items)
        out.append(payload)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    from transformers import AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    examples, method_lookup = build_examples(tokenizer, config)
    train_indices, test_indices = split_examples(examples, config)
    model, label_to_id, mean, std, history = train_router(examples, train_indices, test_indices, config)

    train_predictions = evaluate_predictions(
        model, examples, "train", train_indices, label_to_id, mean, std, method_lookup
    )
    test_predictions = evaluate_predictions(
        model, examples, "test", test_indices, label_to_id, mean, std, method_lookup
    )
    prediction_rows = train_predictions + test_predictions
    prediction_summary = summarize_prediction_rows(prediction_rows)

    id_to_label = {idx: label for label, idx in label_to_id.items()}
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(FEATURE_NAMES),
            "hidden_dim": config.hidden_dim,
            "label_names": [id_to_label[idx] for idx in range(len(id_to_label))],
            "feature_names": FEATURE_NAMES,
            "mean": mean,
            "std": std,
            "config": asdict(config),
        },
        output_dir / "router.pt",
    )
    write_csv(output_dir / "examples.csv", [asdict(example) for example in examples])
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in prediction_rows])
    write_csv(output_dir / "prediction_summary.csv", prediction_summary)
    write_csv(output_dir / "train_history.csv", history)
    summary_payload = {
        "config": asdict(config),
        "feature_names": FEATURE_NAMES,
        "label_names": [id_to_label[idx] for idx in range(len(id_to_label))],
        "train_examples": len(train_indices),
        "test_examples": len(test_indices),
        "history_tail": history[-5:],
        "prediction_summary": prediction_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("split,task_family,samples,oracle_label_accuracy,routed_success_rate,avg_token_ratio_vs_full_raw,avg_forward_seconds")
    for row in prediction_summary:
        print(
            f"{row['split']},{row['task_family']},{row['samples']},"
            f"{row['oracle_label_accuracy']:.4f},{row['routed_success_rate']:.4f},"
            f"{row['avg_token_ratio_vs_full_raw']:.4f},{row['avg_forward_seconds']:.4f}"
        )
    print(f"saved router to {output_dir / 'router.pt'}")


if __name__ == "__main__":
    main()
