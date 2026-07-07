from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_position_mode_planner_from_repack_results as base  # noqa: E402


ACTIONS = ("k2_compact", "k3_compact", "full")
ACTION_TO_METHOD = {
    "k2_compact": ("k2", "rope_delta_repack_compact_query_pos"),
    "k3_compact": ("k3", "rope_delta_repack_compact_query_pos"),
    "full": ("k2", "full_kv_cache"),
    "prompt_k2": ("k2", "prompt_rebuild_selected_pages"),
    "prompt_k3": ("k3", "prompt_rebuild_selected_pages"),
}


@dataclass(frozen=True)
class Config:
    benchmark_pairs: tuple[tuple[str, str], ...]
    output_dir: str
    label_target: str
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    test_fraction: float
    use_text_features: bool
    split_by_case: bool
    seed: int


@dataclass
class PairExample:
    source: str
    benchmark: str
    task: str
    case_id: str
    context_tokens: int
    query_tokens: int
    page_tokens: int
    k2_pages: str
    k3_pages: str
    label_best: str
    label_safe_vs_full: str
    features: list[float]


@dataclass
class PredictionRow:
    split: str
    policy: str
    source: str
    benchmark: str
    task: str
    case_id: str
    target_label: str
    predicted_action: str
    method: str
    score: float
    exact_correct: int
    answer_nll: float
    active_kv_tokens: int
    active_kv_ratio_vs_full: float
    speedup_vs_full_online: float
    label_correct: int


def parse_pairs(value: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "|" not in item:
            raise ValueError(f"benchmark pair must use k2_dir|k3_dir: {item}")
        left, right = item.split("|", 1)
        pairs.append((left.strip(), right.strip()))
    return tuple(pairs)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a two-stage k2/k3/full KV budget planner.")
    parser.add_argument("--benchmark_pairs", default="", help="Comma-separated k2_dir|k3_dir pairs.")
    parser.add_argument("--k2_dirs", default="", help="Comma-separated top-k=2 benchmark dirs; avoids shell pipes.")
    parser.add_argument("--k3_dirs", default="", help="Comma-separated top-k=3 benchmark dirs; avoids shell pipes.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_target", choices=["best", "safe_vs_full"], default="best")
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.35)
    parser.add_argument("--use_text_features", action="store_true")
    parser.add_argument("--split_by_case", action="store_true")
    parser.add_argument("--seed", type=int, default=2026070706)
    args = parser.parse_args()
    if args.k2_dirs or args.k3_dirs:
        k2_dirs = base.parse_csv_tuple(args.k2_dirs)
        k3_dirs = base.parse_csv_tuple(args.k3_dirs)
        if len(k2_dirs) != len(k3_dirs) or not k2_dirs:
            raise ValueError("--k2_dirs and --k3_dirs must be non-empty and have the same length")
        benchmark_pairs = tuple(zip(k2_dirs, k3_dirs))
    else:
        if not args.benchmark_pairs:
            raise ValueError("provide --benchmark_pairs or --k2_dirs/--k3_dirs")
        benchmark_pairs = parse_pairs(args.benchmark_pairs)
    return Config(
        benchmark_pairs=benchmark_pairs,
        output_dir=args.output_dir,
        label_target=args.label_target,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_fraction=args.test_fraction,
        use_text_features=args.use_text_features,
        split_by_case=args.split_by_case,
        seed=args.seed,
    )


def load_rows(directory: Path) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, dict[str, str]]]]:
    run_cfg = base.load_run_config(directory)
    rows = base.read_csv(directory / "results.csv")
    return run_cfg, base.group_rows(rows)


def page_layout_features(context_tokens: int, page_tokens: int, pages: list[int]) -> list[float]:
    total_pages = max(1, math.ceil(context_tokens / max(1, page_tokens)))
    denom = max(1.0, float(total_pages - 1))
    if not pages:
        return [0.0] * 9
    sorted_pages = sorted(pages)
    gaps = [b - a for a, b in zip(sorted_pages, sorted_pages[1:])]
    return [
        float(len(sorted_pages)),
        float(sorted_pages[0]) / denom,
        float(sorted_pages[-1]) / denom,
        sum(sorted_pages) / len(sorted_pages) / denom,
        float(sorted_pages[-1] - sorted_pages[0]) / denom,
        float(max(gaps)) / denom if gaps else 0.0,
        1.0 if 0 in sorted_pages else 0.0,
        1.0 if gaps and all(gap == 1 for gap in gaps) else (1.0 if len(sorted_pages) <= 1 else 0.0),
        float(len(sorted_pages) * page_tokens) / max(1.0, float(context_tokens)),
    ]


def feature_names(task_to_id: dict[str, int], benchmark_to_id: dict[str, int], source_to_id: dict[str, int]) -> list[str]:
    layout = [
        "pages",
        "first_page_norm",
        "last_page_norm",
        "mean_page_norm",
        "span_width_norm",
        "max_gap_norm",
        "has_page0",
        "all_pages_adjacent",
        "selected_kv_ratio",
    ]
    return (
        [
            "is_longbench",
            "is_ruler",
            "context_tokens_log",
            "query_tokens_log",
            "page_tokens_log",
        ]
        + [f"k2_{name}" for name in layout]
        + [f"k3_{name}" for name in layout]
        + [
            "k2_subset_k3",
            "page_jaccard",
            "k3_added_pages",
            "k3_added_page0",
        ]
        + [f"k2_{name}" for name in base.TEXT_FEATURE_NAMES]
        + [f"k3_{name}" for name in base.TEXT_FEATURE_NAMES]
        + [f"task={name}" for name in sorted(task_to_id, key=task_to_id.get)]
        + [f"benchmark={name}" for name in sorted(benchmark_to_id, key=benchmark_to_id.get)]
        + [f"source={name}" for name in sorted(source_to_id, key=source_to_id.get)]
    )


def one_hot(name: str, mapping: dict[str, int]) -> list[float]:
    values = [0.0] * len(mapping)
    if name in mapping:
        values[mapping[name]] = 1.0
    return values


def action_row(
    paired: dict[str, dict[str, dict[str, str]]],
    action: str,
) -> dict[str, str]:
    side, method = ACTION_TO_METHOD[action]
    return paired[side][method]


def full_tokens(paired: dict[str, dict[str, dict[str, str]]]) -> int:
    return base.row_kv(action_row(paired, "full"))


def choose_best_action(paired: dict[str, dict[str, dict[str, str]]]) -> str:
    return max(
        ACTIONS,
        key=lambda action: (
            base.row_score(action_row(paired, action)),
            -base.row_kv(action_row(paired, action)),
            -base.row_nll(action_row(paired, action)),
        ),
    )


def choose_safe_vs_full_action(paired: dict[str, dict[str, dict[str, str]]]) -> str:
    full_score = base.row_score(action_row(paired, "full"))
    for action in ("k2_compact", "k3_compact"):
        if base.row_score(action_row(paired, action)) + 1e-12 >= full_score:
            return action
    return "full"


def label_for(config: Config, example: PairExample) -> str:
    return example.label_safe_vs_full if config.label_target == "safe_vs_full" else example.label_best


def build_examples(config: Config) -> tuple[
    list[PairExample],
    dict[tuple[str, str, str, str], dict[str, dict[str, dict[str, str]]]],
    list[str],
]:
    raw: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str, str], dict[str, dict[str, dict[str, str]]]] = {}
    tokenizer_cache: dict[str, Any] = {}
    auto_tokenizer: Any | None = None
    for k2_dir_text, k3_dir_text in config.benchmark_pairs:
        k2_dir = Path(k2_dir_text)
        k3_dir = Path(k3_dir_text)
        source = f"{k2_dir.name}__{k3_dir.name}"
        k2_cfg, k2_rows = load_rows(k2_dir)
        k3_cfg, k3_rows = load_rows(k3_dir)
        page_tokens = int(k2_cfg.get("page_tokens", 512) or 512)
        max_context_tokens = int(k2_cfg.get("max_context_tokens", 4096) or 4096)
        case_lookup: dict[tuple[str, str, str], base.BenchCase] = {}
        tokenizer: Any | None = None
        if config.use_text_features:
            if auto_tokenizer is None:
                from transformers import AutoTokenizer

                auto_tokenizer = AutoTokenizer
            bcfg = base.bench_config_from_run(k2_cfg)
            case_lookup = {
                (case.benchmark, case.task, case.case_id): case
                for case in (base.load_longbench_cases(bcfg) + base.load_ruler_cases(bcfg))
            }
            model_path = str(k2_cfg.get("model_name_or_path", ""))
            if model_path:
                if model_path not in tokenizer_cache:
                    tokenizer_cache[model_path] = auto_tokenizer.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = tokenizer_cache[model_path]
        for key in sorted(set(k2_rows) & set(k3_rows)):
            k2_by_method = k2_rows[key]
            k3_by_method = k3_rows[key]
            needed = {"full_kv_cache", "rope_delta_repack_compact_query_pos", "prompt_rebuild_selected_pages"}
            if not needed.issubset(k2_by_method) or not needed.issubset(k3_by_method):
                continue
            ref = k2_by_method["full_kv_cache"]
            k2_pages = base.parse_pages(k2_by_method["rope_delta_repack_compact_query_pos"]["selected_pages"])
            k3_pages = base.parse_pages(k3_by_method["rope_delta_repack_compact_query_pos"]["selected_pages"])
            k2_text = base.zero_text_features()
            k3_text = base.zero_text_features()
            case = case_lookup.get(key)
            if tokenizer is not None and case is not None:
                k2_text = base.text_features_for_case(tokenizer, case, max_context_tokens, page_tokens, k2_pages)
                k3_text = base.text_features_for_case(tokenizer, case, max_context_tokens, page_tokens, k3_pages)
            paired = {
                "k2": k2_by_method,
                "k3": k3_by_method,
            }
            label_best = choose_best_action(paired)
            label_safe = choose_safe_vs_full_action(paired)
            source_key = f"{source}::{key[0]}::{key[1]}::{key[2]}"
            raw.append(
                {
                    "source": source,
                    "source_key": source_key,
                    "benchmark": key[0],
                    "task": key[1],
                    "case_id": key[2],
                    "context_tokens": int(float(ref["context_tokens"])),
                    "query_tokens": int(float(ref["query_tokens"])),
                    "page_tokens": page_tokens,
                    "k2_pages": k2_pages,
                    "k3_pages": k3_pages,
                    "k2_text": k2_text,
                    "k3_text": k3_text,
                    "label_best": label_best,
                    "label_safe_vs_full": label_safe,
                }
            )
            lookup[(source_key, key[0], key[1], key[2])] = paired
    if not raw:
        raise ValueError("no paired planner examples were built")

    task_to_id = {task: idx for idx, task in enumerate(sorted({case["task"] for case in raw}))}
    benchmark_to_id = {bench: idx for idx, bench in enumerate(sorted({case["benchmark"] for case in raw}))}
    source_to_id = {source: idx for idx, source in enumerate(sorted({case["source"] for case in raw}))}
    names = feature_names(task_to_id, benchmark_to_id, source_to_id)
    examples: list[PairExample] = []
    for case in raw:
        k2_set = set(case["k2_pages"])
        k3_set = set(case["k3_pages"])
        union = k2_set | k3_set
        added = k3_set - k2_set
        features = [
            1.0 if case["benchmark"] == "longbench" else 0.0,
            1.0 if case["benchmark"].startswith("ruler") else 0.0,
            math.log1p(case["context_tokens"]),
            math.log1p(case["query_tokens"]),
            math.log1p(case["page_tokens"]),
        ]
        features += page_layout_features(case["context_tokens"], case["page_tokens"], case["k2_pages"])
        features += page_layout_features(case["context_tokens"], case["page_tokens"], case["k3_pages"])
        features += [
            1.0 if k2_set and k2_set.issubset(k3_set) else 0.0,
            float(len(k2_set & k3_set)) / max(1.0, float(len(union))),
            float(len(added)),
            1.0 if 0 in added else 0.0,
        ]
        features += case["k2_text"] + case["k3_text"]
        features += one_hot(case["task"], task_to_id)
        features += one_hot(case["benchmark"], benchmark_to_id)
        features += one_hot(case["source"], source_to_id)
        examples.append(
            PairExample(
                source=case["source_key"],
                benchmark=case["benchmark"],
                task=case["task"],
                case_id=case["case_id"],
                context_tokens=case["context_tokens"],
                query_tokens=case["query_tokens"],
                page_tokens=case["page_tokens"],
                k2_pages=json.dumps(case["k2_pages"]),
                k3_pages=json.dumps(case["k3_pages"]),
                label_best=case["label_best"],
                label_safe_vs_full=case["label_safe_vs_full"],
                features=features,
            )
        )
    return examples, lookup, names


def split_indices(examples: list[PairExample], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    if config.split_by_case:
        groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for idx, example in enumerate(examples):
            groups[(example.benchmark, example.task, example.case_id)].append(idx)
        buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        for key, indices in groups.items():
            group_label = Counter(label_for(config, examples[idx]) for idx in indices).most_common(1)[0][0]
            buckets[(key[1], group_label)].append(key)
        train: list[int] = []
        test: list[int] = []
        for keys in buckets.values():
            rng.shuffle(keys)
            if len(keys) == 1:
                train.extend(idx for key in keys for idx in groups[key])
                continue
            n_test = max(1, min(len(keys) - 1, round(len(keys) * config.test_fraction)))
            for key in keys[:n_test]:
                test.extend(groups[key])
            for key in keys[n_test:]:
                train.extend(groups[key])
        rng.shuffle(train)
        rng.shuffle(test)
        return train, test

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        buckets[(example.task, label_for(config, example))].append(idx)
    train = []
    test = []
    for indices in buckets.values():
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


def labels(config: Config, examples: list[PairExample]) -> list[str]:
    return [label_for(config, example) for example in examples]


def train_model(
    examples: list[PairExample],
    train_indices: list[int],
    test_indices: list[int],
    mean: list[float],
    std: list[float],
    config: Config,
) -> tuple[base.MLP, dict[str, int], list[dict[str, Any]]]:
    label_names = sorted(set(labels(config, examples)))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}

    def xy(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [base.norm_features(examples[idx].features, mean, std) for idx in indices]
        ys = [label_to_id[label_for(config, examples[idx])] for idx in indices]
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)

    train_x, train_y = xy(train_indices)
    test_x, test_y = xy(test_indices) if test_indices else (train_x, train_y)
    torch.manual_seed(config.seed)
    model = base.MLP(train_x.shape[1], config.hidden_dim, len(label_names))
    counts = torch.bincount(train_y, minlength=len(label_names)).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    weights = weights / weights[weights > 0].mean().clamp_min(1e-6)
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
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "train_label_accuracy": train_acc,
                    "test_label_accuracy": test_acc,
                }
            )
    return model, label_to_id, history


def add_prediction(
    rows: list[PredictionRow],
    split: str,
    policy: str,
    example: PairExample,
    target: str,
    action: str,
    lookup: dict[tuple[str, str, str, str], dict[str, dict[str, dict[str, str]]]],
) -> None:
    paired = lookup[(example.source, example.benchmark, example.task, example.case_id)]
    row = action_row(paired, action)
    active = base.row_kv(row)
    ft = full_tokens(paired)
    rows.append(
        PredictionRow(
            split=split,
            policy=policy,
            source=example.source,
            benchmark=example.benchmark,
            task=example.task,
            case_id=example.case_id,
            target_label=target,
            predicted_action=action,
            method=ACTION_TO_METHOD[action][1],
            score=base.row_score(row),
            exact_correct=int(float(row["exact_correct"])),
            answer_nll=base.row_nll(row),
            active_kv_tokens=active,
            active_kv_ratio_vs_full=active / ft if ft else 0.0,
            speedup_vs_full_online=float(row["speedup_vs_full_online"]),
            label_correct=int(action == target),
        )
    )


def evaluate(
    model: base.MLP,
    examples: list[PairExample],
    indices: list[int],
    split: str,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, dict[str, dict[str, str]]]],
    config: Config,
) -> list[PredictionRow]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    out: list[PredictionRow] = []
    model.eval()
    for idx in indices:
        example = examples[idx]
        target = label_for(config, example)
        for action in ACTIONS:
            add_prediction(out, split, f"fixed_{action}", example, target, action, lookup)
        paired = lookup[(example.source, example.benchmark, example.task, example.case_id)]
        add_prediction(out, split, "prompt_k2", example, target, "prompt_k2", lookup)
        add_prediction(out, split, "prompt_k3", example, target, "prompt_k3", lookup)
        add_prediction(out, split, "oracle_best", example, target, choose_best_action(paired), lookup)
        add_prediction(out, split, "oracle_safe_vs_full", example, target, choose_safe_vs_full_action(paired), lookup)
        x = torch.tensor([base.norm_features(example.features, mean, std)], dtype=torch.float32)
        with torch.inference_mode():
            pred = id_to_label[int(model(x).argmax(-1).item())]
        add_prediction(out, split, "learned_planner", example, target, pred, lookup)
    return out


def summarize(rows: list[PredictionRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        groups[(row.split, row.policy, "__overall__")].append(row)
        groups[(row.split, row.policy, row.benchmark)].append(row)
    out: list[dict[str, Any]] = []
    for (split, policy, group), items in sorted(groups.items()):
        payload: dict[str, Any] = {
            "split": split,
            "policy": policy,
            "group": group,
            "samples": len(items),
            "avg_score": sum(item.score for item in items) / len(items),
            "exact_accuracy": sum(item.exact_correct for item in items) / len(items),
            "avg_answer_nll": sum(item.answer_nll for item in items) / len(items),
            "avg_active_kv_ratio_vs_full": sum(item.active_kv_ratio_vs_full for item in items) / len(items),
            "avg_speedup_vs_full_online": sum(item.speedup_vs_full_online for item in items) / len(items),
            "label_accuracy": sum(item.label_correct for item in items) / len(items),
        }
        counts = Counter(item.predicted_action for item in items)
        for action, count in sorted(counts.items()):
            payload[f"select_{action}_rate"] = count / len(items)
        out.append(payload)
    return out


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


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples, lookup, names = build_examples(config)
    train_indices, test_indices = split_indices(examples, config)
    mean, std = base.normalize(examples, train_indices)
    model, label_to_id, history = train_model(examples, train_indices, test_indices, mean, std, config)
    rows = evaluate(model, examples, train_indices, "train", label_to_id, mean, std, lookup, config)
    rows += evaluate(model, examples, test_indices, "test", label_to_id, mean, std, lookup, config)
    summary = summarize(rows)
    write_csv(output_dir / "examples.csv", [asdict(example) for example in examples])
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "prediction_summary.csv", summary)
    write_csv(output_dir / "train_history.csv", history)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(examples[0].features),
            "hidden_dim": config.hidden_dim,
            "label_to_id": label_to_id,
            "feature_names": names,
            "mean": mean,
            "std": std,
            "config": asdict(config),
        },
        output_dir / "two_stage_budget_planner.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "examples": len(examples),
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "label_counts_best": dict(Counter(example.label_best for example in examples)),
                "label_counts_safe_vs_full": dict(Counter(example.label_safe_vs_full for example in examples)),
                "history_tail": history[-5:],
                "prediction_summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("split,policy,group,samples,score,kv_ratio,label_acc")
    for row in summary:
        if row["split"] == "test" and row["group"] == "__overall__":
            print(
                f"{row['split']},{row['policy']},{row['group']},{row['samples']},"
                f"{row['avg_score']:.4f},{row['avg_active_kv_ratio_vs_full']:.4f},{row['label_accuracy']:.4f}"
            )
    print(f"saved planner to {output_dir / 'two_stage_budget_planner.pt'}")


if __name__ == "__main__":
    main()
