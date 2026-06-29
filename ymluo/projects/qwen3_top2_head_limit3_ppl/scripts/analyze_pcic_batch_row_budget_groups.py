#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def budget_for_layer(row_map: dict[str, Any], layer_idx: int) -> dict[str, Any]:
    default_budget = row_map.get("default", {"type": "full"})
    layer_budgets = row_map.get("layers", {})
    budget = layer_budgets.get(str(layer_idx), layer_budgets.get(layer_idx, default_budget))
    if not isinstance(budget, dict):
        raise ValueError(f"layer {layer_idx} budget must be an object")
    return budget


def budget_key(budget: dict[str, Any]) -> str:
    return json.dumps(budget, sort_keys=True, separators=(",", ":"))


def budget_type(budget: dict[str, Any]) -> str:
    return str(budget.get("type", budget.get("kind", "full"))).lower()


def analyze_map(path: Path, num_layers: int) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    batch_maps = loaded.get("batch_maps", loaded.get("batch", []))
    if not isinstance(batch_maps, list) or not batch_maps:
        raise ValueError(f"{path} does not contain non-empty batch_maps")

    row_count = len(batch_maps)
    all_same_layers = 0
    mixed_layers = 0
    total_group_forwards = 0
    full_only_layers = 0
    landmark_mixed_layers = 0
    max_groups = 0
    group_count_hist: Counter[int] = Counter()
    group_type_hist: Counter[str] = Counter()
    mixed_layer_names: list[str] = []

    for layer_idx in range(num_layers):
        row_budgets = [budget_for_layer(row_map, layer_idx) for row_map in batch_maps]
        groups: dict[str, list[int]] = {}
        group_types: dict[str, str] = {}
        for row_idx, budget in enumerate(row_budgets):
            key = budget_key(budget)
            groups.setdefault(key, []).append(row_idx)
            group_types[key] = budget_type(budget)
        group_count = len(groups)
        max_groups = max(max_groups, group_count)
        group_count_hist[group_count] += 1
        total_group_forwards += group_count
        types = sorted(set(group_types.values()))
        group_type_hist["+".join(types)] += 1
        if group_count == 1:
            all_same_layers += 1
            if types == ["full"]:
                full_only_layers += 1
        else:
            mixed_layers += 1
            if "landmark" in types:
                landmark_mixed_layers += 1
            mixed_layer_names.append(str(layer_idx))

    filename = path.name
    stage = "unknown"
    if "_initial_" in filename:
        stage = "initial"
    elif "_extension_" in filename:
        stage = "extension"
    return {
        "path": str(path),
        "output": path.parts[path.parts.index("outputs") + 1] if "outputs" in path.parts else path.parent.parent.name,
        "map_name": filename,
        "stage": stage,
        "row_count": row_count,
        "num_layers": num_layers,
        "all_same_layers": all_same_layers,
        "mixed_layers": mixed_layers,
        "full_only_layers": full_only_layers,
        "landmark_mixed_layers": landmark_mixed_layers,
        "total_group_forwards": total_group_forwards,
        "avg_groups_per_layer": total_group_forwards / max(1, num_layers),
        "max_groups_per_layer": max_groups,
        "group_count_hist": ";".join(f"{count}:{group_count_hist[count]}" for count in sorted(group_count_hist)),
        "group_type_hist": ";".join(f"{key}:{group_type_hist[key]}" for key in sorted(group_type_hist)),
        "mixed_layers_list": ",".join(mixed_layer_names),
    }


def summarize_output(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_output.setdefault(str(row["output"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for output, output_rows in sorted(by_output.items()):
        map_count = len(output_rows)
        num_layers_total = sum(int(row["num_layers"]) for row in output_rows)
        all_same = sum(int(row["all_same_layers"]) for row in output_rows)
        mixed = sum(int(row["mixed_layers"]) for row in output_rows)
        group_forwards = sum(int(row["total_group_forwards"]) for row in output_rows)
        row_counts = sorted(set(int(row["row_count"]) for row in output_rows))
        summaries.append(
            {
                "output": output,
                "map_count": map_count,
                "row_counts": ",".join(str(count) for count in row_counts),
                "all_same_layers": all_same,
                "mixed_layers": mixed,
                "all_same_layer_fraction": all_same / max(1, num_layers_total),
                "avg_groups_per_layer": group_forwards / max(1, num_layers_total),
                "total_group_forwards": group_forwards,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", nargs="+", required=True, help="Output directories containing batch_maps/*.json")
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--csv_out", type=Path, required=True)
    parser.add_argument("--summary_csv_out", type=Path, required=True)
    parser.add_argument("--md_out", type=Path, required=True)
    args = parser.parse_args()

    map_paths: list[Path] = []
    for output in args.outputs:
        output_path = Path(output)
        map_paths.extend(sorted((output_path / "batch_maps").glob("*.json")))
    if not map_paths:
        raise SystemExit("no batch map json files found")

    rows = [analyze_map(path, args.num_layers) for path in map_paths]
    summaries = summarize_output(rows)
    write_csv(args.csv_out, rows)
    write_csv(args.summary_csv_out, summaries)

    lines = [
        "# Batch-row Budget Group 分析（2026-06-29）",
        "",
        "## 目的",
        "",
        "该分析不跑模型，只解析已有 `batch_maps/*.json`，量化 batch-row candidate gate 中哪些层真的需要 row-wise mixed sparse attention。",
        "",
        f"详细 CSV：`{args.csv_out}`",
        f"汇总 CSV：`{args.summary_csv_out}`",
        "",
        "## 输出级汇总",
        "",
        "| output | maps | row_counts | all-same layer frac | avg groups/layer | mixed layers |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['output']}` | {row['map_count']} | `{row['row_counts']}` | "
            f"{row['all_same_layer_fraction']:.3f} | {row['avg_groups_per_layer']:.3f} | "
            f"{row['mixed_layers']} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- `all-same layer` 表示该层所有 candidate rows 使用同一个 budget，可直接整批 forward。",
        "- `mixed layer` 表示该层不同 rows 使用 full / landmark 等不同 budget，是真正需要 fused/tensorized row-wise sparse attention 的位置。",
        "- 如果 all-same layer 占比高，当前 dispatch 优化是合理的；如果 optimized batched 仍慢，剩余瓶颈主要在 mixed layers 的候选维 cache 复制和 sparse attention 本体。",
    ]
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
