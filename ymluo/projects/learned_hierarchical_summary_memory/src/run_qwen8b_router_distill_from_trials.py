from __future__ import annotations

import argparse
import csv
import json
import math
import random
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

from memory_policy_router_runtime import TinyMemoryRouter  # noqa: E402
from run_qwen8b_paper_benchmarks import (  # noqa: E402
    Config as BenchConfig,
    SUMMARY_TASKS,
    load_longbench_cases,
    load_ruler_cases,
    parse_csv_tuple,
    parse_int_tuple,
    router_features,
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
    "retriever_top3_overlap",
    "retriever_score_gap",
    "retriever_positive_blocks",
    "retriever_top1_norm",
    "retriever_top2_norm",
    "retriever_gap_norm",
    "retriever_top1_position",
    "retriever_top2_position",
    "retriever_top1_is_recent",
    "prefix_unique_word_ratio",
    "prefix_number_count_log",
    "prefix_capitalized_count_log",
]


@dataclass(frozen=True)
class Config:
    benchmark_output_dir: str
    output_dir: str
    candidate_methods: tuple[str, ...]
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    test_fraction: float
    summary_rouge_slack: float
    seed: int


@dataclass
class RouterExample:
    benchmark: str
    task: str
    case_id: str
    task_family: str
    label: str
    oracle_score: float
    oracle_token_ratio: float
    features: list[float]


@dataclass
class PredictionRow:
    split: str
    benchmark: str
    task: str
    case_id: str
    task_family: str
    oracle_label: str
    predicted_label: str
    label_correct: int
    method_score: float
    method_success: int
    token_ratio_vs_full_raw: float
    seconds: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Distill a Qwen3-8B LongBench/RULER oracle into a memory router.")
    parser.add_argument("--benchmark_output_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--candidate_methods",
        default="full_raw,summary1_8,summary1_4,summary1_2,summary1000,static_hier,retrieval_raw_k1,retrieval_raw_k2",
    )
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.35)
    parser.add_argument("--summary_rouge_slack", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=2026070404)
    args = parser.parse_args()
    return Config(
        benchmark_output_dir=args.benchmark_output_dir,
        output_dir=args.output_dir,
        candidate_methods=parse_csv_tuple(args.candidate_methods),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_fraction=args.test_fraction,
        summary_rouge_slack=args.summary_rouge_slack,
        seed=args.seed,
    )


def bench_config_from_summary(path: Path) -> BenchConfig:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    raw = payload["config"]
    return BenchConfig(
        output_dir=raw["output_dir"],
        model_name_or_path=raw["model_name_or_path"],
        adapter_path=raw.get("adapter_path", ""),
        longbench_data_dir=raw["longbench_data_dir"],
        ruler_data_dir=raw["ruler_data_dir"],
        longbench_tasks=tuple(raw["longbench_tasks"]),
        ruler_tasks=tuple(raw["ruler_tasks"]),
        ruler_context_lengths=tuple(int(item) for item in raw["ruler_context_lengths"]),
        methods=tuple(raw["methods"]),
        max_examples_per_task=int(raw["max_examples_per_task"]),
        block_tokens=int(raw["block_tokens"]),
        recent_tokens=int(raw["recent_tokens"]),
        max_input_tokens=int(raw["max_input_tokens"]),
        summary10_words=int(raw["summary10_words"]),
        summary100_words=int(raw["summary100_words"]),
        summary1000_words=int(raw["summary1000_words"]),
        max_new_tokens_exact=int(raw["max_new_tokens_exact"]),
        max_new_tokens_summary=int(raw["max_new_tokens_summary"]),
        dtype=raw["dtype"],
        attn_implementation=raw["attn_implementation"],
        device_map=raw["device_map"],
        cuda_visible_devices=raw.get("cuda_visible_devices", ""),
        router_path=raw.get("router_path", ""),
        seed=int(raw["seed"]),
    )


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["benchmark"], row["task"], row["case_id"])


def choose_oracle(rows: list[dict[str, str]], config: Config) -> dict[str, Any]:
    candidates = [row for row in rows if row["method"] in config.candidate_methods]
    if not candidates:
        candidates = rows
    task = candidates[0]["task"]
    full_rows = [row for row in candidates if row["method"] == "full_raw"]
    full_score = float(full_rows[0]["score"]) if full_rows else max(float(row["score"]) for row in candidates)
    max_score = max(float(row["score"]) for row in candidates)

    if task in SUMMARY_TASKS:
        threshold = max(0.0, full_score - config.summary_rouge_slack)
    else:
        threshold = 1.0 if max_score >= 1.0 else max_score

    successful = [row for row in candidates if float(row["score"]) >= threshold and threshold > 0.0]
    if not successful:
        successful = [row for row in candidates if float(row["score"]) >= max_score]
    selected = min(
        successful,
        key=lambda row: (float(row["token_ratio_vs_full_raw"]), float(row["seconds"]), row["method"]),
    )
    return {
        "label": selected["method"],
        "score": float(selected["score"]),
        "token_ratio": float(selected["token_ratio_vs_full_raw"]),
        "threshold": threshold,
    }


def build_examples(tokenizer: Any, bench_config: BenchConfig, config: Config) -> tuple[list[RouterExample], dict[tuple[str, str, str, str], dict[str, str]]]:
    bench_dir = Path(config.benchmark_output_dir)
    trial_rows = read_csv(bench_dir / "trials.csv")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    method_lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in trial_rows:
        grouped.setdefault(key(row), []).append(row)
        method_lookup[(row["benchmark"], row["task"], row["case_id"], row["method"])] = row

    cases = load_longbench_cases(bench_config) + load_ruler_cases(bench_config)
    case_lookup = {(case.benchmark, case.task, case.case_id): case for case in cases}

    examples: list[RouterExample] = []
    for group_key, rows in sorted(grouped.items()):
        case = case_lookup.get(group_key)
        if case is None:
            continue
        oracle = choose_oracle(rows, config)
        features, task_family = router_features(tokenizer, case, bench_config)
        examples.append(
            RouterExample(
                benchmark=group_key[0],
                task=group_key[1],
                case_id=group_key[2],
                task_family=task_family,
                label=oracle["label"],
                oracle_score=oracle["score"],
                oracle_token_ratio=oracle["token_ratio"],
                features=features,
            )
        )
    if not examples:
        raise ValueError("no examples were built")
    return examples, method_lookup


def split_indices(examples: list[RouterExample], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    by_label: dict[str, list[int]] = {}
    for idx, example in enumerate(examples):
        by_label.setdefault(example.label, []).append(idx)
    train: list[int] = []
    test: list[int] = []
    for indices in by_label.values():
        rng.shuffle(indices)
        if len(indices) == 1:
            train.extend(indices)
            continue
        test_count = max(1, min(len(indices) - 1, round(len(indices) * config.test_fraction)))
        test.extend(indices[:test_count])
        train.extend(indices[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def normalize(examples: list[RouterExample], train_indices: list[int]) -> tuple[list[float], list[float]]:
    dim = len(examples[0].features)
    mean: list[float] = []
    std: list[float] = []
    for col in range(dim):
        vals = [examples[idx].features[col] for idx in train_indices]
        m = sum(vals) / max(1, len(vals))
        var = sum((val - m) ** 2 for val in vals) / max(1, len(vals))
        mean.append(m)
        std.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return mean, std


def tensorize(examples: list[RouterExample], indices: list[int], label_to_id: dict[str, int], mean: list[float], std: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for idx in indices:
        example = examples[idx]
        xs.append([(val - mean[col]) / max(std[col], 1e-6) for col, val in enumerate(example.features)])
        ys.append(label_to_id[example.label])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def train_model(examples: list[RouterExample], train_indices: list[int], test_indices: list[int], config: Config) -> tuple[TinyMemoryRouter, dict[str, int], list[float], list[float], list[dict[str, Any]]]:
    label_names = sorted({example.label for example in examples})
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    mean, std = normalize(examples, train_indices)
    train_x, train_y = tensorize(examples, train_indices, label_to_id, mean, std)
    test_x, test_y = tensorize(examples, test_indices, label_to_id, mean, std) if test_indices else (train_x, train_y)

    torch.manual_seed(config.seed)
    model = TinyMemoryRouter(train_x.shape[1], config.hidden_dim, len(label_names))
    counts = torch.bincount(train_y, minlength=len(label_names)).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    weights = weights / weights.mean().clamp_min(1e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y, weight=weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.inference_mode():
                train_acc = float((model(train_x).argmax(-1) == train_y).float().mean())
                test_acc = float((model(test_x).argmax(-1) == test_y).float().mean())
            history.append({"epoch": epoch, "loss": float(loss.detach()), "train_label_accuracy": train_acc, "test_label_accuracy": test_acc})
    return model, label_to_id, mean, std, history


def evaluate_split(
    model: TinyMemoryRouter,
    examples: list[RouterExample],
    indices: list[int],
    split: str,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    method_lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> list[PredictionRow]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    x, _ = tensorize(examples, indices, label_to_id, mean, std)
    model.eval()
    with torch.inference_mode():
        pred_ids = model(x).argmax(-1).tolist()
    rows: list[PredictionRow] = []
    for local_idx, example_idx in enumerate(indices):
        example = examples[example_idx]
        pred = id_to_label[int(pred_ids[local_idx])]
        method_row = method_lookup.get((example.benchmark, example.task, example.case_id, pred))
        if method_row is None:
            score = 0.0
            token_ratio = 1.0
            seconds = 0.0
        else:
            score = float(method_row["score"])
            token_ratio = float(method_row["token_ratio_vs_full_raw"])
            seconds = float(method_row["seconds"])
        if example.task in SUMMARY_TASKS:
            success = int(score >= max(0.0, example.oracle_score - config.summary_rouge_slack))
        else:
            success = int(score >= example.oracle_score and example.oracle_score > 0.0)
        rows.append(
            PredictionRow(
                split=split,
                benchmark=example.benchmark,
                task=example.task,
                case_id=example.case_id,
                task_family=example.task_family,
                oracle_label=example.label,
                predicted_label=pred,
                label_correct=int(pred == example.label),
                method_score=score,
                method_success=success,
                token_ratio_vs_full_raw=token_ratio,
                seconds=seconds,
            )
        )
    return rows


def summarize_predictions(rows: list[PredictionRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[PredictionRow]] = {}
    for row in rows:
        groups.setdefault((row.split, "__overall__"), []).append(row)
        groups.setdefault((row.split, row.task_family), []).append(row)
        groups.setdefault((row.split, row.benchmark), []).append(row)
    out: list[dict[str, Any]] = []
    for (split, group), items in sorted(groups.items()):
        payload: dict[str, Any] = {
            "split": split,
            "group": group,
            "samples": len(items),
            "label_accuracy": sum(row.label_correct for row in items) / len(items),
            "routed_success": sum(row.method_success for row in items) / len(items),
            "avg_score": sum(row.method_score for row in items) / len(items),
            "avg_token_ratio_vs_full_raw": sum(row.token_ratio_vs_full_raw for row in items) / len(items),
            "avg_seconds": sum(row.seconds for row in items) / len(items),
        }
        counts: dict[str, int] = {}
        for row in items:
            counts[row.predicted_label] = counts.get(row.predicted_label, 0) + 1
        for label, count in sorted(counts.items()):
            payload[f"select_{label}"] = count
            payload[f"select_{label}_rate"] = count / len(items)
        out.append(payload)
    return out


def main() -> None:
    from transformers import AutoTokenizer

    config = parse_args()
    bench_dir = Path(config.benchmark_output_dir)
    bench_config = bench_config_from_summary(bench_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(bench_config.model_name_or_path, trust_remote_code=True)
    examples, method_lookup = build_examples(tokenizer, bench_config, config)
    train_indices, test_indices = split_indices(examples, config)
    model, label_to_id, mean, std, history = train_model(examples, train_indices, test_indices, config)

    rows = []
    rows.extend(evaluate_split(model, examples, train_indices, "train", label_to_id, mean, std, method_lookup, config))
    rows.extend(evaluate_split(model, examples, test_indices, "test", label_to_id, mean, std, method_lookup, config))
    summary = summarize_predictions(rows)
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
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "prediction_summary.csv", summary)
    write_csv(output_dir / "train_history.csv", history)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "bench_config": asdict(bench_config),
                "label_names": [id_to_label[idx] for idx in range(len(id_to_label))],
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "history_tail": history[-5:],
                "prediction_summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("split,group,samples,label_accuracy,routed_success,avg_score,avg_token_ratio_vs_full_raw")
    for row in summary:
        print(
            f"{row['split']},{row['group']},{row['samples']},"
            f"{row['label_accuracy']:.4f},{row['routed_success']:.4f},"
            f"{row['avg_score']:.4f},{row['avg_token_ratio_vs_full_raw']:.4f}"
        )
    print(f"saved router to {output_dir / 'router.pt'}")


if __name__ == "__main__":
    main()
