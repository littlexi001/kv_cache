from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass
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
    load_longbench_cases,
    load_ruler_cases,
    parse_csv_tuple,
    parse_int_tuple,
    router_features,
)
from run_qwen8b_router_distill_from_trials import FEATURE_NAMES  # noqa: E402
from run_synthetic_router_distillation import (  # noqa: E402
    Config as SyntheticConfig,
    RouterExample,
    as_bench_case,
    build_synthetic_cases,
    load_text_ids,
)


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    text_paths: tuple[str, ...]
    dataset_names: tuple[str, ...]
    benchmark_output_dir: str
    candidate_methods: tuple[str, ...]
    cases_per_dataset: int
    prefill_token_lengths: tuple[int, ...]
    sample_stride_tokens: int
    eval_start_tokens: int
    block_tokens: int
    recent_tokens: int
    max_text_tokens: int
    max_input_tokens: int
    summary10_words: int
    summary100_words: int
    summary1000_words: int
    policy: str
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    test_fraction: float
    seed: int


@dataclass
class PredictionRow:
    split: str
    benchmark: str
    task: str
    case_id: str
    dataset: str
    kind: str
    task_family: str
    oracle_label: str
    predicted_label: str
    label_correct: int


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fast non-benchmark training for recent-plus runtime router.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_paths", required=True)
    parser.add_argument("--dataset_names", required=True)
    parser.add_argument("--benchmark_output_dir", required=True)
    parser.add_argument(
        "--candidate_methods",
        default=(
            "full_raw,recent_plus_summary1_8,recent_plus_summary1_4,recent_plus_summary1_2,"
            "recent_plus_static_hier,recent_plus_retrieval_raw_k1,recent_plus_retrieval_raw_k2,"
            "recent_plus_retrieval_raw_k3,recent_plus_retrieval_raw_k4,recent_plus_retrieval_raw_k8"
        ),
    )
    parser.add_argument("--cases_per_dataset", type=int, default=320)
    parser.add_argument("--prefill_token_lengths", default="4096,8192,16384,20000")
    parser.add_argument("--sample_stride_tokens", type=int, default=512)
    parser.add_argument("--eval_start_tokens", type=int, default=20000)
    parser.add_argument("--block_tokens", type=int, default=1024)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--max_text_tokens", type=int, default=260000)
    parser.add_argument("--max_input_tokens", type=int, default=24000)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--policy", choices=["budget", "balanced", "conservative"], default="balanced")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026070603)
    args = parser.parse_args()
    text_paths = tuple(item.strip() for item in args.text_paths.split(",") if item.strip())
    dataset_names = tuple(item.strip() for item in args.dataset_names.split(",") if item.strip())
    if len(text_paths) != len(dataset_names):
        raise ValueError("--text_paths and --dataset_names must have the same count")
    return Config(
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        text_paths=text_paths,
        dataset_names=dataset_names,
        benchmark_output_dir=args.benchmark_output_dir,
        candidate_methods=parse_csv_tuple(args.candidate_methods),
        cases_per_dataset=args.cases_per_dataset,
        prefill_token_lengths=parse_int_tuple(args.prefill_token_lengths),
        sample_stride_tokens=args.sample_stride_tokens,
        eval_start_tokens=args.eval_start_tokens,
        block_tokens=args.block_tokens,
        recent_tokens=args.recent_tokens,
        max_text_tokens=args.max_text_tokens,
        max_input_tokens=args.max_input_tokens,
        summary10_words=args.summary10_words,
        summary100_words=args.summary100_words,
        summary1000_words=args.summary1000_words,
        policy=args.policy,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )


def to_synthetic_config(config: Config) -> SyntheticConfig:
    return SyntheticConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        benchmark_output_dir=config.benchmark_output_dir,
        candidate_methods=config.candidate_methods,
        cases_per_dataset=config.cases_per_dataset,
        prefill_tokens=config.prefill_token_lengths[0],
        prefill_token_lengths=config.prefill_token_lengths,
        sample_stride_tokens=config.sample_stride_tokens,
        eval_start_tokens=config.eval_start_tokens,
        block_tokens=config.block_tokens,
        recent_tokens=config.recent_tokens,
        max_text_tokens=config.max_text_tokens,
        max_input_tokens=config.max_input_tokens,
        summary10_words=config.summary10_words,
        summary100_words=config.summary100_words,
        summary1000_words=config.summary1000_words,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        test_fraction=config.test_fraction,
        label_mode="length_aware",
        seed=config.seed,
    )


def bench_config(config: Config) -> BenchConfig:
    return BenchConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        adapter_path="",
        longbench_data_dir="",
        ruler_data_dir="",
        longbench_tasks=(),
        ruler_tasks=(),
        ruler_context_lengths=(),
        methods=config.candidate_methods,
        max_examples_per_task=0,
        block_tokens=config.block_tokens,
        recent_tokens=config.recent_tokens,
        max_input_tokens=config.max_input_tokens,
        summary10_words=config.summary10_words,
        summary100_words=config.summary100_words,
        summary1000_words=config.summary1000_words,
        max_new_tokens_exact=48,
        max_new_tokens_summary=160,
        dtype="float16",
        attn_implementation="sdpa",
        device_map="auto",
        cuda_visible_devices="",
        router_path="",
        seed=config.seed,
    )


def exact_label(kind: str, old_blocks: int, prefix_tokens: int, policy: str) -> str:
    if policy == "conservative":
        if kind in {"magic_multivalue", "magic_multiquery", "four_old"} or old_blocks >= 4:
            return "recent_plus_retrieval_raw_k4"
        return "recent_plus_retrieval_raw_k3"
    if policy == "budget":
        if old_blocks <= 1 and prefix_tokens >= 12_000:
            return "recent_plus_retrieval_raw_k1"
        if old_blocks <= 2:
            return "recent_plus_retrieval_raw_k2"
        if old_blocks == 3:
            return "recent_plus_retrieval_raw_k3"
        return "recent_plus_retrieval_raw_k4"
    if kind in {"cwe_k1", "fwe_k1"}:
        return "recent_plus_retrieval_raw_k2"
    if old_blocks <= 1:
        return "recent_plus_retrieval_raw_k2"
    if old_blocks == 2:
        return "recent_plus_retrieval_raw_k2"
    if old_blocks == 3:
        return "recent_plus_retrieval_raw_k3"
    return "recent_plus_retrieval_raw_k4"


def choose_label(kind: str, old_blocks: int, prefix_tokens: int, task_family: str, policy: str) -> str:
    if kind == "full_context":
        return "full_raw"
    if task_family == "generation":
        if kind == "summary_detailed":
            return "recent_plus_summary1_2" if policy == "conservative" else "recent_plus_summary1_4"
        if kind == "recent_generation":
            return "recent_plus_summary1_8"
        return "recent_plus_summary1_4" if policy == "conservative" else "recent_plus_summary1_8"
    if old_blocks <= 0:
        return "recent_plus_summary1_8"
    return exact_label(kind, old_blocks, prefix_tokens, policy)


def build_examples(tokenizer: Any, config: Config) -> list[RouterExample]:
    synth_cfg = to_synthetic_config(config)
    token_ids = load_text_ids(tokenizer, synth_cfg)
    cases = build_synthetic_cases(tokenizer, token_ids, synth_cfg)
    cfg = bench_config(config)
    examples: list[RouterExample] = []
    for case in cases:
        bench_case = as_bench_case(case)
        features, task_family = router_features(tokenizer, bench_case, cfg)
        prefix_tokens = int(features[12])
        label = choose_label(case.kind, len(case.old_target_blocks), prefix_tokens, task_family, config.policy)
        if label not in config.candidate_methods:
            label = "full_raw"
        examples.append(
            RouterExample(
                benchmark=case.benchmark,
                task=case.task,
                case_id=case.case_id,
                dataset=case.dataset,
                kind=case.kind,
                task_family=task_family,
                label=label,
                oracle_token_ratio=0.0,
                features=features,
            )
        )
    return examples


def split_indices(examples: list[RouterExample], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    by_key: dict[tuple[str, str, str], list[int]] = {}
    for idx, example in enumerate(examples):
        by_key.setdefault((example.dataset, example.kind, example.label), []).append(idx)
    train: list[int] = []
    test: list[int] = []
    for indices in by_key.values():
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
        mean.append(float(statistics.mean(vals)))
        std.append(float(statistics.pstdev(vals) or 1.0))
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
        example = examples[idx]
        xs.append([(value - mean[col]) / max(std[col], 1e-6) for col, value in enumerate(example.features)])
        ys.append(label_to_id[example.label])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def train_router(
    examples: list[RouterExample],
    train_indices: list[int],
    test_indices: list[int],
    config: Config,
) -> tuple[TinyMemoryRouter, dict[str, int], list[float], list[float], list[dict[str, Any]]]:
    label_names = sorted({example.label for example in examples})
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    mean, std = normalize(examples, train_indices)
    train_x, train_y = tensorize(examples, train_indices, label_to_id, mean, std)
    test_x, test_y = tensorize(examples, test_indices, label_to_id, mean, std)
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
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.inference_mode():
                train_acc = float((model(train_x).argmax(-1) == train_y).float().mean())
                test_acc = float((model(test_x).argmax(-1) == test_y).float().mean())
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "train_label_accuracy": train_acc,
                    "test_label_accuracy": test_acc,
                }
            )
    return model, label_to_id, mean, std, history


def evaluate_split(
    model: TinyMemoryRouter,
    examples: list[RouterExample],
    indices: list[int],
    split: str,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
) -> list[PredictionRow]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    x, _ = tensorize(examples, indices, label_to_id, mean, std)
    model.eval()
    with torch.inference_mode():
        preds = model(x).argmax(-1).tolist()
    rows: list[PredictionRow] = []
    for local_idx, example_idx in enumerate(indices):
        example = examples[example_idx]
        pred = id_to_label[int(preds[local_idx])]
        rows.append(
            PredictionRow(
                split=split,
                benchmark=example.benchmark,
                task=example.task,
                case_id=example.case_id,
                dataset=example.dataset,
                kind=example.kind,
                task_family=example.task_family,
                oracle_label=example.label,
                predicted_label=pred,
                label_correct=int(pred == example.label),
            )
        )
    return rows


def summarize_predictions(rows: list[PredictionRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[PredictionRow]] = {}
    for row in rows:
        groups.setdefault((row.split, "__overall__"), []).append(row)
        groups.setdefault((row.split, row.task_family), []).append(row)
        groups.setdefault((row.split, row.kind), []).append(row)
    out: list[dict[str, Any]] = []
    for (split, group), items in sorted(groups.items()):
        payload: dict[str, Any] = {
            "split": split,
            "group": group,
            "samples": len(items),
            "label_accuracy": sum(row.label_correct for row in items) / len(items),
        }
        counts: dict[str, int] = {}
        for row in items:
            counts[row.predicted_label] = counts.get(row.predicted_label, 0) + 1
        for label, count in sorted(counts.items()):
            payload[f"select_{label}"] = count
            payload[f"select_{label}_rate"] = count / len(items)
        out.append(payload)
    return out


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


def evaluate_heldout(
    model: TinyMemoryRouter,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    tokenizer: Any,
    config: Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bench_dir = Path(config.benchmark_output_dir)
    bench_cfg = bench_config_from_summary(bench_dir)
    trials = read_csv(bench_dir / "trials.csv")
    lookup = {(row["benchmark"], row["task"], row["case_id"], row["method"]): row for row in trials}
    cases = load_longbench_cases(bench_cfg) + load_ruler_cases(bench_cfg)
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    rows: list[dict[str, Any]] = []
    model.eval()
    for case in cases:
        features, task_family = router_features(tokenizer, case, bench_cfg)
        x = torch.tensor(
            [[(value - mean[idx]) / max(std[idx], 1e-6) for idx, value in enumerate(features)]],
            dtype=torch.float32,
        )
        with torch.inference_mode():
            pred = id_to_label[int(model(x).argmax(-1).item())]
        trial = lookup.get((case.benchmark, case.task, case.case_id, pred))
        full = lookup.get((case.benchmark, case.task, case.case_id, "full_raw"))
        if trial is None or full is None:
            continue
        rows.append(
            {
                "benchmark": case.benchmark,
                "task": case.task,
                "case_id": case.case_id,
                "task_family": task_family,
                "predicted_label": pred,
                "score": float(trial["score"]),
                "full_score": float(full["score"]),
                "token_ratio_vs_full_raw": float(trial["token_ratio_vs_full_raw"]),
                "seconds": float(trial["seconds"]),
            }
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault("__overall__", []).append(row)
        groups.setdefault(row["task_family"], []).append(row)
        groups.setdefault(row["benchmark"], []).append(row)
    summary: list[dict[str, Any]] = []
    for group, items in sorted(groups.items()):
        score = sum(row["score"] for row in items) / len(items)
        full = sum(row["full_score"] for row in items) / len(items)
        ratio = sum(row["token_ratio_vs_full_raw"] for row in items) / len(items)
        payload: dict[str, Any] = {
            "group": group,
            "samples": len(items),
            "avg_score": score,
            "avg_full_score": full,
            "relative_to_full": score / full if full else "",
            "avg_token_ratio_vs_full_raw": ratio,
        }
        counts: dict[str, int] = {}
        for row in items:
            counts[row["predicted_label"]] = counts.get(row["predicted_label"], 0) + 1
        for label, count in sorted(counts.items()):
            payload[f"select_{label}"] = count
            payload[f"select_{label}_rate"] = count / len(items)
        summary.append(payload)
    return rows, summary


def main() -> None:
    from transformers import AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    examples = build_examples(tokenizer, config)
    train_indices, test_indices = split_indices(examples, config)
    model, label_to_id, mean, std, history = train_router(examples, train_indices, test_indices, config)
    pred_rows = evaluate_split(model, examples, train_indices, "train", label_to_id, mean, std)
    pred_rows += evaluate_split(model, examples, test_indices, "test", label_to_id, mean, std)
    prediction_summary = summarize_predictions(pred_rows)
    heldout_rows, heldout_summary = evaluate_heldout(model, label_to_id, mean, std, tokenizer, config)
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
    write_csv(output_dir / "examples.csv", [asdict(row) for row in examples])
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in pred_rows])
    write_csv(output_dir / "prediction_summary.csv", prediction_summary)
    write_csv(output_dir / "train_history.csv", history)
    write_csv(output_dir / "heldout_benchmark_predictions.csv", heldout_rows)
    write_csv(output_dir / "heldout_benchmark_summary.csv", heldout_summary)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "label_names": label_names,
                "examples": len(examples),
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "history_tail": history[-5:],
                "prediction_summary": prediction_summary,
                "heldout_benchmark_summary": heldout_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("SYNTHETIC_LABEL")
    for row in prediction_summary:
        print(f"{row['split']},{row['group']},{row['samples']},{row['label_accuracy']:.4f}")
    print("HELDOUT_BENCHMARK_OFFLINE")
    for row in heldout_summary:
        print(
            f"{row['group']},{row['samples']},{row['avg_score']:.4f},"
            f"{row['avg_full_score']:.4f},{row['relative_to_full']:.4f},"
            f"{row['avg_token_ratio_vs_full_raw']:.4f}"
        )
    print(f"saved router to {output_dir / 'router.pt'}")


if __name__ == "__main__":
    main()
