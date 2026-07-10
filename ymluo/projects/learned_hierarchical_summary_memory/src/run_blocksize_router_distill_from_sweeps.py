from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
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
    router_features,
)
from run_qwen8b_router_distill_from_trials import FEATURE_NAMES, bench_config_from_summary  # noqa: E402


@dataclass(frozen=True)
class Config:
    benchmark_output_dirs: tuple[str, ...]
    output_dir: str
    feature_block_tokens: int
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    test_fraction: float
    summary_rouge_slack: float
    quality_mode: str
    allowed_label_regex: str
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
    full_score: float
    max_score: float
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
    parser = argparse.ArgumentParser(description="Distill block-size/topK sweep oracle into a runtime router.")
    parser.add_argument("--benchmark_output_dirs", required=True, help="Comma-separated sweep output directories.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_block_tokens", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--summary_rouge_slack", type=float, default=0.03)
    parser.add_argument("--quality_mode", choices=["full", "best", "best_or_full"], default="best_or_full")
    parser.add_argument(
        "--allowed_label_regex",
        default="",
        help="Optional full-match regex for candidate action labels. If no candidate matches, the full candidate set is used.",
    )
    parser.add_argument("--seed", type=int, default=2026070731)
    args = parser.parse_args()
    return Config(
        benchmark_output_dirs=parse_csv_tuple(args.benchmark_output_dirs),
        output_dir=args.output_dir,
        feature_block_tokens=args.feature_block_tokens,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_fraction=args.test_fraction,
        summary_rouge_slack=args.summary_rouge_slack,
        quality_mode=args.quality_mode,
        allowed_label_regex=args.allowed_label_regex,
        seed=args.seed,
    )


def case_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["benchmark"], row["task"], row["case_id"]


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def block_from_summary(path: Path) -> int:
    bench_config = bench_config_from_summary(path)
    return int(bench_config.block_tokens)


def label_for(row: dict[str, str], block_tokens: int) -> str:
    method = row["method"]
    if method == "full_raw":
        return "full_raw"
    if method.startswith("recent_plus_"):
        return f"recent_plus_b{block_tokens}_{method[len('recent_plus_'):]}"
    return f"b{block_tokens}_{method}"


def threshold_for(rows: list[dict[str, Any]], config: Config) -> tuple[float, float, float]:
    task = rows[0]["task"]
    full_scores = [finite_float(row["score"]) for row in rows if row["method"] == "full_raw"]
    full_scores = [score for score in full_scores if math.isfinite(score)]
    scores = [finite_float(row["score"]) for row in rows]
    scores = [score for score in scores if math.isfinite(score)]
    if not scores:
        return 0.0, 0.0, 0.0
    full_score = max(full_scores) if full_scores else max(scores)
    max_score = max(scores)
    if task in SUMMARY_TASKS:
        threshold = max(0.0, full_score - config.summary_rouge_slack)
    elif config.quality_mode == "full":
        threshold = full_score
    elif config.quality_mode == "best":
        threshold = max_score
    else:
        if max_score >= 1.0:
            threshold = 1.0
        elif full_score > 0:
            threshold = full_score
        else:
            threshold = max_score
    return threshold, full_score, max_score


def choose_oracle(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    rows = [
        row
        for row in rows
        if math.isfinite(finite_float(row.get("score")))
        and math.isfinite(finite_float(row.get("token_ratio_vs_full_raw")))
    ]
    if not rows:
        return {
            "label": "full_raw",
            "score": 0.0,
            "token_ratio": 1.0,
            "full_score": 0.0,
            "max_score": 0.0,
            "threshold": 0.0,
        }
    threshold, full_score, max_score = threshold_for(rows, config)
    candidates = [row for row in rows if row["method"] != "full_raw"]
    if config.allowed_label_regex:
        allowed = [
            row for row in candidates
            if re.fullmatch(config.allowed_label_regex, str(row.get("label", "")))
        ]
        if allowed:
            candidates = allowed
    if not candidates:
        selected = max(rows, key=lambda row: finite_float(row["score"]))
        return {
            "label": selected["label"],
            "score": finite_float(selected["score"]),
            "token_ratio": finite_float(selected["token_ratio_vs_full_raw"], 1.0),
            "full_score": full_score,
            "max_score": max_score,
            "threshold": threshold,
        }
    ok = [row for row in candidates if finite_float(row["score"]) + 1e-12 >= threshold]
    if not ok:
        ok = [row for row in candidates if finite_float(row["score"]) >= max_score]
    if not ok:
        ok = [max(candidates, key=lambda row: finite_float(row["score"]))]
    selected = min(
        ok,
        key=lambda row: (
            finite_float(row["token_ratio_vs_full_raw"], 1.0),
            finite_float(row["seconds"], 0.0),
            row["label"],
        ),
    )
    return {
        "label": selected["label"],
        "score": finite_float(selected["score"]),
        "token_ratio": finite_float(selected["token_ratio_vs_full_raw"], 1.0),
        "full_score": full_score,
        "max_score": max_score,
        "threshold": threshold,
    }


def load_sweep_rows(
    config: Config,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]], list[BenchConfig]]:
    all_rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    bench_configs: list[BenchConfig] = []
    for raw_dir in config.benchmark_output_dirs:
        path = Path(raw_dir)
        block_tokens = block_from_summary(path)
        bench_config = bench_config_from_summary(path)
        bench_configs.append(replace(bench_config, block_tokens=config.feature_block_tokens))
        for row in read_csv(path / "trials.csv"):
            labeled = dict(row)
            labeled["block_tokens"] = block_tokens
            labeled["label"] = label_for(labeled, block_tokens)
            all_rows.append(labeled)
            lookup[(row["benchmark"], row["task"], row["case_id"], labeled["label"])] = labeled
    if not bench_configs:
        raise ValueError("no benchmark dirs")
    return all_rows, lookup, bench_configs


def build_examples(tokenizer: Any, config: Config) -> tuple[list[RouterExample], dict[tuple[str, str, str, str], dict[str, Any]]]:
    rows, lookup, bench_configs = load_sweep_rows(config)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(case_key(row), []).append(row)

    case_lookup = {}
    for bench_config in bench_configs:
        for case in load_longbench_cases(bench_config) + load_ruler_cases(bench_config):
            case_lookup[(case.benchmark, case.task, case.case_id)] = case
    feature_bench_config = bench_configs[0]
    examples: list[RouterExample] = []
    for key, group_rows in sorted(grouped.items()):
        case = case_lookup.get(key)
        if case is None:
            continue
        oracle = choose_oracle(group_rows, config)
        features, task_family = router_features(tokenizer, case, feature_bench_config)
        examples.append(
            RouterExample(
                benchmark=key[0],
                task=key[1],
                case_id=key[2],
                task_family=task_family,
                label=oracle["label"],
                oracle_score=oracle["score"],
                oracle_token_ratio=oracle["token_ratio"],
                full_score=oracle["full_score"],
                max_score=oracle["max_score"],
                features=features,
            )
        )
    if not examples:
        raise ValueError("no examples")
    return examples, lookup


def split_indices(examples: list[RouterExample], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    by_group: dict[tuple[str, str, str], list[int]] = {}
    for idx, example in enumerate(examples):
        by_group.setdefault((example.benchmark, example.task, example.label), []).append(idx)
    train: list[int] = []
    test: list[int] = []
    for indices in by_group.values():
        rng.shuffle(indices)
        if len(indices) == 1:
            train.extend(indices)
            continue
        n_test = max(1, min(len(indices) - 1, round(len(indices) * config.test_fraction)))
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def normalize(examples: list[RouterExample], train_indices: list[int]) -> tuple[list[float], list[float]]:
    mean: list[float] = []
    std: list[float] = []
    for col in range(len(examples[0].features)):
        values = [examples[idx].features[col] for idx in train_indices]
        m = sum(values) / len(values)
        var = sum((value - m) ** 2 for value in values) / len(values)
        mean.append(float(m))
        std.append(float(math.sqrt(var)) if var > 1e-12 else 1.0)
    return mean, std


def tensorize(
    examples: list[RouterExample],
    indices: list[int],
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for idx in indices:
        example = examples[idx]
        xs.append([(value - mean[col]) / max(std[col], 1e-6) for col, value in enumerate(example.features)])
        ys.append(label_to_id[example.label])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def train_model(
    examples: list[RouterExample],
    train_indices: list[int],
    test_indices: list[int],
    config: Config,
) -> tuple[TinyMemoryRouter, dict[str, int], list[float], list[float], list[dict[str, Any]]]:
    labels = sorted({example.label for example in examples})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    mean, std = normalize(examples, train_indices)
    train_x, train_y = tensorize(examples, train_indices, label_to_id, mean, std)
    test_x, test_y = tensorize(examples, test_indices, label_to_id, mean, std) if test_indices else (train_x, train_y)
    torch.manual_seed(config.seed)
    model = TinyMemoryRouter(train_x.shape[1], config.hidden_dim, len(labels))
    counts = torch.bincount(train_y, minlength=len(labels)).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    weights = weights / weights.mean().clamp_min(1e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y, weight=weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.inference_mode():
                train_acc = float((model(train_x).argmax(-1) == train_y).float().mean())
                test_acc = float((model(test_x).argmax(-1) == test_y).float().mean())
            history.append({"epoch": epoch, "loss": float(loss.detach()), "train_label_accuracy": train_acc, "test_label_accuracy": test_acc})
    return model, label_to_id, mean, std, history


def evaluate(
    model: TinyMemoryRouter,
    examples: list[RouterExample],
    indices: list[int],
    split: str,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
) -> list[PredictionRow]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    x, _ = tensorize(examples, indices, label_to_id, mean, std)
    model.eval()
    with torch.inference_mode():
        preds = model(x).argmax(-1).tolist()
    rows = []
    for local_idx, example_idx in enumerate(indices):
        example = examples[example_idx]
        pred = id_to_label[int(preds[local_idx])]
        trial = lookup.get((example.benchmark, example.task, example.case_id, pred))
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
                method_score=float(trial["score"]) if trial else 0.0,
                token_ratio_vs_full_raw=float(trial["token_ratio_vs_full_raw"]) if trial else 1.0,
                seconds=float(trial["seconds"]) if trial else 0.0,
            )
        )
    return rows


def summarize(rows: list[PredictionRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[PredictionRow]] = {}
    for row in rows:
        groups.setdefault((row.split, "__overall__"), []).append(row)
        groups.setdefault((row.split, row.task_family), []).append(row)
        groups.setdefault((row.split, row.benchmark), []).append(row)
    out = []
    for (split, group), items in sorted(groups.items()):
        payload: dict[str, Any] = {
            "split": split,
            "group": group,
            "samples": len(items),
            "label_accuracy": sum(row.label_correct for row in items) / len(items),
            "avg_score": sum(row.method_score for row in items) / len(items),
            "avg_token_ratio_vs_full_raw": sum(row.token_ratio_vs_full_raw for row in items) / len(items),
            "avg_seconds": sum(row.seconds for row in items) / len(items),
        }
        for row in items:
            key = f"select_{row.predicted_label}"
            payload[key] = payload.get(key, 0) + 1
        out.append(payload)
    return out


def main() -> None:
    from transformers import AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_config = bench_config_from_summary(Path(config.benchmark_output_dirs[0]))
    tokenizer = AutoTokenizer.from_pretrained(bench_config.model_name_or_path, trust_remote_code=True)
    examples, lookup = build_examples(tokenizer, config)
    train_indices, test_indices = split_indices(examples, config)
    model, label_to_id, mean, std, history = train_model(examples, train_indices, test_indices, config)
    rows = evaluate(model, examples, train_indices, "train", label_to_id, mean, std, lookup)
    rows += evaluate(model, examples, test_indices, "test", label_to_id, mean, std, lookup)
    summary = summarize(rows)
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    label_names = [id_to_label[idx] for idx in range(len(id_to_label))]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(FEATURE_NAMES),
            "hidden_dim": config.hidden_dim,
            "label_names": label_names,
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
                "label_names": label_names,
                "examples": len(examples),
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
    print("split,group,samples,label_accuracy,avg_score,avg_token_ratio_vs_full_raw")
    for row in summary:
        print(
            f"{row['split']},{row['group']},{row['samples']},{row['label_accuracy']:.4f},"
            f"{row['avg_score']:.4f},{row['avg_token_ratio_vs_full_raw']:.4f}"
        )
    print(f"saved router to {output_dir / 'router.pt'}")


if __name__ == "__main__":
    main()
