from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(array.max()),
    }


def evaluate_trace(
    trace_path: Path,
    device: torch.device,
    budgets: tuple[float, ...],
    metrics: dict[str, dict[str, list[float]]],
) -> list[int]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    layers = []
    for record in payload["records"]:
        layers.append(int(record["layer"]))
        query = record["query"].to(device).float()[0, :, 0, :]
        key = record["key"].to(device).float()[0]
        value = record["value"].to(device).float()[0]
        scaling = float(record["scaling"])
        query_head_count = int(query.shape[0])
        kv_head_count = int(key.shape[0])
        groups = query_head_count // kv_head_count
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        value = value[:, :history_count]

        for head in range(query_head_count):
            kv_head = head // groups
            head_value = value[kv_head]
            scores = torch.mv(key[kv_head], query[head]) * scaling
            probabilities = torch.softmax(scores, dim=-1)
            value_norm = head_value.norm(dim=-1).clamp_min(1.0e-8)
            full_output = torch.mv(head_value.transpose(0, 1), probabilities)
            full_output_norm = full_output.norm().clamp_min(1.0e-8)
            weighted_norm = probabilities * value_norm
            total_weighted_norm = weighted_norm.sum().clamp_min(1.0e-8)
            ranking_scores = {
                "qk": scores,
                "value_bound": scores + value_norm.log(),
            }

            for budget in budgets:
                keep_count = max(1, math.ceil(budget * history_count))
                for method, priority in ranking_scores.items():
                    indices = torch.topk(priority, k=keep_count).indices
                    selected_scores = scores.index_select(0, indices)
                    selected_probabilities = torch.softmax(selected_scores, dim=-1)
                    selected_values = head_value.index_select(0, indices)
                    sparse_output = torch.mv(
                        selected_values.transpose(0, 1), selected_probabilities
                    )
                    relative_error = (
                        (sparse_output - full_output).norm() / full_output_norm
                    )
                    output_cosine = torch.nn.functional.cosine_similarity(
                        sparse_output.unsqueeze(0), full_output.unsqueeze(0)
                    )[0]
                    retained_mass = probabilities.index_select(0, indices).sum()
                    retained_weighted_norm = (
                        weighted_norm.index_select(0, indices).sum()
                        / total_weighted_norm
                    )
                    prefix = f"budget_{budget:g}"
                    metrics[method][f"{prefix}_relative_output_error"].append(
                        float(relative_error.item())
                    )
                    metrics[method][f"{prefix}_output_cosine"].append(
                        float(output_cosine.item())
                    )
                    metrics[method][f"{prefix}_retained_attention_mass"].append(
                        float(retained_mass.item())
                    )
                    metrics[method][f"{prefix}_retained_weighted_norm"].append(
                        float(retained_weighted_norm.item())
                    )
        del query, key, value
    return layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--budgets", default="0.005,0.01,0.02,0.04")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    budgets = tuple(float(item) for item in args.budgets.split(",") if item)
    if not budgets or min(budgets) <= 0.0 or max(budgets) > 1.0:
        raise ValueError("budgets must be in (0, 1]")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    traces = []
    for trace_path in args.trace_paths:
        layers = evaluate_trace(trace_path, device, budgets, metrics)
        traces.append({"path": str(trace_path), "layers": layers})

    report = {
        "traces": traces,
        "budgets": budgets,
        "selection_rules": {
            "qk": "rank by exact scaled q dot k",
            "value_bound": "rank by exact scaled q dot k plus log value norm",
        },
        "metrics": {
            method: {
                metric: summarize(values)
                for metric, values in sorted(method_metrics.items())
            }
            for method, method_metrics in sorted(metrics.items())
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
