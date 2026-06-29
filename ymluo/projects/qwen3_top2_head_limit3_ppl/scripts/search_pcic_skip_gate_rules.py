#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable


def fnum(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def inum(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_features(row: dict[str, str]) -> dict[str, Any]:
    initial_combo = row.get("initial_combo", "")
    final_combo = row.get("final_combo", "")
    return {
        "case_id": row.get("case_id", ""),
        "task": row.get("task", ""),
        "block": row.get("block", ""),
        "extended": inum(row, "extended"),
        "no_change": inum(row, "no_change_after_extension"),
        "anchor_hit": inum(row, "anchor_hit"),
        "initial_combo": initial_combo,
        "final_combo": final_combo,
        "initial_margin": fnum(row, "initial_margin"),
        "abs_pairwise_delta": abs(fnum(row, "initial_pairwise_delta")),
        "horizon_gain": fnum(row, "horizon_gain"),
        "horizon_gain_ratio": fnum(row, "horizon_gain_ratio"),
        "extension_seconds": fnum(row, "extension_seconds"),
        "delta_ppl": fnum(row, "delta_ppl"),
        "initial_is_final": int(initial_combo == final_combo and bool(initial_combo)),
    }


def evaluate_rule(
    name: str,
    predicate: Callable[[dict[str, Any]], bool],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    extended_rows = [row for row in rows if row["extended"] == 1]
    selected = [row for row in extended_rows if predicate(row)]
    false_skip = [row for row in selected if row["no_change"] == 0]
    true_skip = [row for row in selected if row["no_change"] == 1]
    total_extension_seconds = sum(row["extension_seconds"] for row in extended_rows)
    saved_seconds = sum(row["extension_seconds"] for row in true_skip)
    false_skip_seconds = sum(row["extension_seconds"] for row in false_skip)
    return {
        "rule": name,
        "selected_blocks": len(selected),
        "true_skip_blocks": len(true_skip),
        "false_skip_blocks": len(false_skip),
        "saved_seconds": saved_seconds,
        "false_skip_seconds": false_skip_seconds,
        "saved_fraction": saved_seconds / max(total_extension_seconds, 1e-9),
        "selected_cases": ";".join(f"{row['case_id']}:{row['block']}" for row in selected),
        "false_skip_cases": ";".join(f"{row['case_id']}:{row['block']}" for row in false_skip),
    }


def build_rules(rows: list[dict[str, Any]]) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    margins = sorted({row["initial_margin"] for row in rows if row["extended"] == 1})
    pairwise = sorted({row["abs_pairwise_delta"] for row in rows if row["extended"] == 1})
    ratios = sorted({row["horizon_gain_ratio"] for row in rows if row["extended"] == 1})
    rules: list[tuple[str, Callable[[dict[str, Any]], bool]]] = []

    rules.append(("anchor_hit", lambda row: row["anchor_hit"] == 1))
    rules.append(("anchor_hit_and_pairwise_zero", lambda row: row["anchor_hit"] == 1 and row["abs_pairwise_delta"] <= 1e-12))
    rules.append(("anchor_hit_and_ratio_le_1", lambda row: row["anchor_hit"] == 1 and row["horizon_gain_ratio"] <= 1.0))

    for threshold in margins:
        rules.append(
            (
                f"margin_le_{threshold:.6g}",
                lambda row, threshold=threshold: row["initial_margin"] <= threshold,
            )
        )
        rules.append(
            (
                f"anchor_hit_and_margin_le_{threshold:.6g}",
                lambda row, threshold=threshold: row["anchor_hit"] == 1 and row["initial_margin"] <= threshold,
            )
        )
    for threshold in pairwise:
        rules.append(
            (
                f"pairwise_le_{threshold:.6g}",
                lambda row, threshold=threshold: row["abs_pairwise_delta"] <= threshold,
            )
        )
        rules.append(
            (
                f"anchor_hit_and_pairwise_le_{threshold:.6g}",
                lambda row, threshold=threshold: row["anchor_hit"] == 1 and row["abs_pairwise_delta"] <= threshold,
            )
        )
    for threshold in ratios:
        rules.append(
            (
                f"ratio_le_{threshold:.6g}",
                lambda row, threshold=threshold: row["horizon_gain_ratio"] <= threshold,
            )
        )
        rules.append(
            (
                f"anchor_hit_and_ratio_le_{threshold:.6g}",
                lambda row, threshold=threshold: row["anchor_hit"] == 1 and row["horizon_gain_ratio"] <= threshold,
            )
        )
    for margin in margins:
        for ratio in ratios:
            rules.append(
                (
                    f"margin_le_{margin:.6g}_and_ratio_le_{ratio:.6g}",
                    lambda row, margin=margin, ratio=ratio: row["initial_margin"] <= margin
                    and row["horizon_gain_ratio"] <= ratio,
                )
            )
            rules.append(
                (
                    f"anchor_hit_and_margin_le_{margin:.6g}_and_ratio_le_{ratio:.6g}",
                    lambda row, margin=margin, ratio=ratio: row["anchor_hit"] == 1
                    and row["initial_margin"] <= margin
                    and row["horizon_gain_ratio"] <= ratio,
                )
            )
    return rules


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_csv", type=Path, required=True)
    parser.add_argument("--rules_csv_out", type=Path, required=True)
    parser.add_argument("--md_out", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    rows = [row_features(row) for row in load_rows(args.block_csv)]
    evaluated = [evaluate_rule(name, predicate, rows) for name, predicate in build_rules(rows)]
    evaluated = [row for row in evaluated if int(row["selected_blocks"]) > 0]
    evaluated.sort(key=lambda row: (int(row["false_skip_blocks"]), -float(row["saved_seconds"]), row["rule"]))
    write_csv(args.rules_csv_out, evaluated)

    zero_false = [row for row in evaluated if int(row["false_skip_blocks"]) == 0]
    lines = [
        "# PCIC Skip-Gate Rule Search（2026-06-29）",
        "",
        "## 目的",
        "",
        "在 `pcic_extension_waste_blocks_2026_06_29.csv` 上搜索简单可解释规则。规则只使用 extension 前可见特征，目标是 zero false-skip 下最大化可省 extension 秒数。",
        "",
        f"完整规则 CSV：`{args.rules_csv_out}`",
        "",
        "## Zero False-Skip Top Rules",
        "",
        "| rule | selected | saved_s | saved_frac | selected_cases |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in zero_false[: args.topk]:
        lines.append(
            f"| `{row['rule']}` | {row['selected_blocks']} | {float(row['saved_seconds']):.3f} | "
            f"{float(row['saved_fraction']):.3f} | `{row['selected_cases']}` |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- `false-skip` 表示规则选择跳过 extension，但后验显示 extension 会改变最终 combo。",
        "- zero false-skip 规则只是当前样本上的候选规则；上线前必须在新样本上验证。",
        "- 如果最优规则只覆盖很少 block，说明 skip-gate 需要更多训练数据或更强特征。",
    ]
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
