from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_variable_budget_planner_from_repack_results as vb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empirical-risk calibrated tail-risk KV budget planner.")
    parser.add_argument("--benchmark_dirs", default="")
    parser.add_argument("--benchmark_groups", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_target", choices=["min_safe", "best"], default="best")
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=2400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--calibration_fraction", type=float, default=0.25)
    parser.add_argument("--risk_alpha", type=float, default=0.05)
    parser.add_argument("--risk_delta", type=float, default=0.10)
    parser.add_argument("--risk_bound", choices=["addone", "hoeffding"], default="addone")
    parser.add_argument(
        "--selection_objective",
        choices=["min_kv", "min_risk", "risk_then_kv", "risk_kv_tradeoff"],
        default="min_kv",
        help=(
            "How to choose among calibration-feasible thresholds. min_kv is the most aggressive; "
            "min_risk is conservative; risk_then_kv chooses the lowest empirical risk then KV; "
            "risk_kv_tradeoff balances the selected risk upper bound and KV ratio."
        ),
    )
    parser.add_argument(
        "--candidate_thresholds",
        default="0,0.0001,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.35,0.5,0.8,1.0",
    )
    parser.add_argument("--use_text_features", action="store_true")
    parser.add_argument("--split_by_case", action="store_true")
    parser.add_argument("--seed", type=int, default=2026070810)
    return parser.parse_args()


def make_vb_config(args: argparse.Namespace) -> vb.Config:
    benchmark_dirs = vb.base.parse_csv_tuple(args.benchmark_dirs)
    benchmark_groups = vb.parse_benchmark_groups(args.benchmark_groups)
    if not benchmark_groups and benchmark_dirs:
        benchmark_groups = (benchmark_dirs,)
    if not benchmark_groups:
        raise ValueError("provide --benchmark_dirs or --benchmark_groups")
    thresholds = tuple(float(item) for item in vb.base.parse_csv_tuple(args.candidate_thresholds))
    return vb.Config(
        benchmark_dirs=benchmark_dirs,
        benchmark_groups=benchmark_groups,
        output_dir=args.output_dir,
        label_target=args.label_target,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=0.0,
        confidence_penalty=0.0,
        ce_loss_weight=1.0,
        expected_cost_weight=0.0,
        unsafe_cost_weight=2.0,
        best_gap_cost_weight=1.0,
        kv_cost_weight=0.05,
        include_full_action=False,
        test_fraction=args.test_fraction,
        use_text_features=args.use_text_features,
        split_by_case=args.split_by_case,
        risk_thresholds=thresholds,
        holdout_tasks=(),
        holdout_benchmarks=(),
        seed=args.seed,
    )


def three_way_split(
    examples: list[vb.VariableBudgetExample],
    config: vb.Config,
    calibration_fraction: float,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(config.seed)
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        groups[(example.benchmark, example.task, example.case_id)].append(idx)
    buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key, indices in groups.items():
        label = Counter(vb.label_for(config, examples[idx]) for idx in indices).most_common(1)[0][0]
        buckets[(key[1], label)].append(key)

    train: list[int] = []
    calibration: list[int] = []
    test: list[int] = []
    for keys in buckets.values():
        rng.shuffle(keys)
        if len(keys) == 1:
            train.extend(idx for key in keys for idx in groups[key])
            continue
        n_test = max(1, min(len(keys) - 1, round(len(keys) * config.test_fraction)))
        remaining = len(keys) - n_test
        n_cal = 0
        if remaining >= 2:
            n_cal = max(1, min(remaining - 1, round(len(keys) * calibration_fraction)))
        test_keys = keys[:n_test]
        cal_keys = keys[n_test : n_test + n_cal]
        train_keys = keys[n_test + n_cal :]
        test.extend(idx for key in test_keys for idx in groups[key])
        calibration.extend(idx for key in cal_keys for idx in groups[key])
        train.extend(idx for key in train_keys for idx in groups[key])
    if not calibration:
        raise ValueError("empty calibration split; reduce --test_fraction or --calibration_fraction")
    rng.shuffle(train)
    rng.shuffle(calibration)
    rng.shuffle(test)
    return train, calibration, test


def model_probs(
    model: vb.base.MLP,
    example: vb.VariableBudgetExample,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    x = torch.tensor([vb.norm_features(example.features, mean, std)], dtype=torch.float32)
    with torch.inference_mode():
        return torch.softmax(model(x), dim=-1)[0]


def tail_action_for_threshold(probs: torch.Tensor, label_to_id: dict[str, int], tau: float) -> str:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return vb.choose_tail_risk_action(probs, id_to_label, tau)


def action_metrics(
    example: vb.VariableBudgetExample,
    action: str,
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    payload = lookup[(example.source, example.benchmark, example.task, example.case_id)]
    row = vb.action_row(payload, action)
    full_score = vb.row_score(payload["full"])
    ft = max(1, vb.full_tokens(payload))
    score = vb.row_score(row)
    return {
        "score": score,
        "exact_correct": int(float(row["exact_correct"])),
        "answer_nll": vb.row_nll(row),
        "active_kv_tokens": vb.row_kv(row),
        "active_kv_ratio_vs_full": vb.row_kv(row) / ft,
        "failure_vs_full": int(score + 1e-12 < full_score),
    }


def evaluate_thresholds(
    *,
    model: vb.base.MLP,
    examples: list[vb.VariableBudgetExample],
    indices: list[int],
    split: str,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tau in thresholds:
        items: list[dict[str, Any]] = []
        action_counts: Counter[str] = Counter()
        for idx in indices:
            example = examples[idx]
            probs = model_probs(model, example, label_to_id, mean, std)
            action = tail_action_for_threshold(probs, label_to_id, tau)
            metrics = action_metrics(example, action, lookup)
            action_counts[action] += 1
            items.append(metrics)
        failures = sum(item["failure_vs_full"] for item in items)
        n = len(items)
        failure_rate = failures / max(1, n)
        payload: dict[str, Any] = {
            "split": split,
            "tau": tau,
            "samples": n,
            "failures_vs_full": failures,
            "failure_rate_vs_full": failure_rate,
            "risk_upper_addone": (failures + 1) / (n + 1) if n else 1.0,
            "risk_upper_hoeffding": failure_rate + math.sqrt(math.log(1.0 / 0.10) / (2 * max(1, n))),
            "avg_score": sum(item["score"] for item in items) / max(1, n),
            "exact_accuracy": sum(item["exact_correct"] for item in items) / max(1, n),
            "avg_active_kv_ratio_vs_full": sum(item["active_kv_ratio_vs_full"] for item in items) / max(1, n),
            "avg_answer_nll": sum(item["answer_nll"] for item in items) / max(1, n),
        }
        for action, count in sorted(action_counts.items(), key=lambda pair: (vb.action_budget(pair[0]), pair[0])):
            payload[f"select_{action}_rate"] = count / max(1, n)
        rows.append(payload)
    return rows


def select_threshold(
    cal_rows: list[dict[str, Any]],
    alpha: float,
    delta: float,
    risk_bound: str,
    selection_objective: str,
) -> dict[str, Any]:
    def risk_upper(row: dict[str, Any]) -> float:
        n = max(1, int(row["samples"]))
        if risk_bound == "addone":
            return (int(row["failures_vs_full"]) + 1) / (n + 1)
        failure_rate = float(row["failure_rate_vs_full"])
        return failure_rate + math.sqrt(math.log(1.0 / max(delta, 1e-12)) / (2 * n))

    annotated = []
    for row in cal_rows:
        item = dict(row)
        item["risk_upper_selected"] = risk_upper(row)
        annotated.append(item)
    feasible = [row for row in annotated if row["risk_upper_selected"] <= alpha + 1e-12]
    if feasible:
        if selection_objective == "min_risk":
            return min(
                feasible,
                key=lambda row: (
                    row["risk_upper_selected"],
                    row["failure_rate_vs_full"],
                    row["avg_active_kv_ratio_vs_full"],
                    -row["avg_score"],
                ),
            )
        if selection_objective == "risk_then_kv":
            return min(
                feasible,
                key=lambda row: (
                    row["failure_rate_vs_full"],
                    row["risk_upper_selected"],
                    row["avg_active_kv_ratio_vs_full"],
                    -row["avg_score"],
                ),
            )
        if selection_objective == "risk_kv_tradeoff":
            return min(
                feasible,
                key=lambda row: (
                    row["risk_upper_selected"] / max(alpha, 1e-12)
                    + 0.25 * row["avg_active_kv_ratio_vs_full"],
                    row["failure_rate_vs_full"],
                    row["avg_active_kv_ratio_vs_full"],
                    -row["avg_score"],
                ),
            )
        return min(
            feasible,
            key=lambda row: (
                row["avg_active_kv_ratio_vs_full"],
                -row["avg_score"],
                row["risk_upper_selected"],
            ),
        )
    return min(
        annotated,
        key=lambda row: (
            row["risk_upper_selected"],
            row["avg_active_kv_ratio_vs_full"],
            -row["avg_score"],
        ),
    )


def prediction_rows_for_policy(
    *,
    model: vb.base.MLP,
    examples: list[vb.VariableBudgetExample],
    indices: list[int],
    split: str,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    selected_tau: float,
    label_target: str,
    policy_name: str,
) -> list[vb.PredictionRow]:
    out: list[vb.PredictionRow] = []
    target_config = vb_config_for_target(label_target)
    for idx in indices:
        example = examples[idx]
        target = vb.label_for(target_config, example)
        payload = lookup[(example.source, example.benchmark, example.task, example.case_id)]
        for action in vb.available_actions(payload):
            vb.add_prediction(out, split, f"fixed_{action}", example, target, action, lookup)
        vb.add_prediction(out, split, "oracle_min_safe", example, target, vb.choose_min_safe(payload), lookup)
        vb.add_prediction(out, split, "oracle_best", example, target, vb.choose_best(payload), lookup)
        probs = model_probs(model, example, label_to_id, mean, std)
        action = tail_action_for_threshold(probs, label_to_id, selected_tau)
        vb.add_prediction(out, split, policy_name, example, target, action, lookup)
    return out


def vb_config_for_target(target: str) -> Any:
    class Minimal:
        label_target = target

    return Minimal()


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
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_vb_config(args)
    examples, lookup, names = vb.build_examples(config)
    train_indices, cal_indices, test_indices = three_way_split(examples, config, args.calibration_fraction)
    mean, std = vb.normalize(examples, train_indices)
    model, label_to_id, history = vb.train_model(
        examples,
        train_indices,
        cal_indices,
        mean,
        std,
        lookup,
        config,
    )
    thresholds = tuple(float(item) for item in vb.base.parse_csv_tuple(args.candidate_thresholds))
    cal_rows = evaluate_thresholds(
        model=model,
        examples=examples,
        indices=cal_indices,
        split="calibration",
        label_to_id=label_to_id,
        mean=mean,
        std=std,
        lookup=lookup,
        thresholds=thresholds,
    )
    selected = select_threshold(
        cal_rows,
        args.risk_alpha,
        args.risk_delta,
        args.risk_bound,
        args.selection_objective,
    )
    selected_tau = float(selected["tau"])
    test_rows = evaluate_thresholds(
        model=model,
        examples=examples,
        indices=test_indices,
        split="test",
        label_to_id=label_to_id,
        mean=mean,
        std=std,
        lookup=lookup,
        thresholds=thresholds,
    )
    policy_predictions = prediction_rows_for_policy(
        model=model,
        examples=examples,
        indices=test_indices,
        split="test",
        label_to_id=label_to_id,
        mean=mean,
        std=std,
        lookup=lookup,
        selected_tau=selected_tau,
        label_target=args.label_target,
        policy_name="conformal_tail_selected",
    )
    prediction_summary = vb.summarize(policy_predictions)
    write_csv(output_dir / "calibration_threshold_sweep.csv", cal_rows)
    write_csv(output_dir / "test_threshold_sweep.csv", test_rows)
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in policy_predictions])
    write_csv(output_dir / "prediction_summary.csv", prediction_summary)
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
            "selected_tau": selected_tau,
            "config": asdict(config),
            "calibration": {
                "risk_alpha": args.risk_alpha,
                "risk_delta": args.risk_delta,
                "risk_bound": args.risk_bound,
                "selection_objective": args.selection_objective,
                "selected": selected,
            },
        },
        output_dir / "conformal_tailrisk_planner.pt",
    )
    payload = {
        "config": asdict(config),
        "examples": len(examples),
        "train_examples": len(train_indices),
        "calibration_examples": len(cal_indices),
        "test_examples": len(test_indices),
        "selected_threshold": selected,
        "risk_bound": args.risk_bound,
        "selection_objective": args.selection_objective,
        "history_tail": history[-5:],
        "prediction_summary": prediction_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("selected_tau", selected_tau)
    print("split,policy,group,samples,score,kv_ratio,label_acc")
    for row in prediction_summary:
        if row["split"] == "test" and row["group"] == "__overall__":
            print(
                f"{row['split']},{row['policy']},{row['group']},{row['samples']},"
                f"{row['avg_score']:.4f},{row['avg_active_kv_ratio_vs_full']:.4f},{row['label_accuracy']:.4f}"
            )
    print(f"saved planner to {output_dir / 'conformal_tailrisk_planner.pt'}")


if __name__ == "__main__":
    main()
