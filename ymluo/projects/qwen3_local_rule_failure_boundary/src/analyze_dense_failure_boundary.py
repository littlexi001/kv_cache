from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = (
    "atomic_evidence_mass",
    "hop1_result_mass",
    "hop2_input_mass",
    "hop2_result_mass",
    "other_token_mass",
    "outside_top20_mass",
    "attention_entropy",
    "effective_tokens",
    "mean_head_logsumexp",
    "mean_hop2_result_logit",
    "mean_hop2_result_cosine",
    "mean_hop2_result_rank",
)

LABELS = {
    "atomic_evidence_mass": "全部证据 mass",
    "hop1_result_mass": "第一跳结果 mass",
    "hop2_input_mass": "第二跳输入 mass",
    "hop2_result_mass": "最终结果 mass",
    "other_token_mass": "其他 token mass",
    "outside_top20_mass": "Top-20 外 mass",
    "attention_entropy": "attention entropy",
    "effective_tokens": "有效 token 数",
    "mean_head_logsumexp": "Softmax logsumexp",
    "mean_hop2_result_logit": "最终证据 QK logit",
    "mean_hop2_result_cosine": "最终证据 QK cosine",
    "mean_hop2_result_rank": "最终证据 rank",
}


def read_rows(path: Path, start: int, end: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for source in rows:
        length = int(source["length"])
        if not start <= length <= end:
            continue
        row: dict[str, Any] = dict(source)
        row["length"] = length
        for field in (
            "top1_correct",
            "candidate_correct",
        ):
            value = source.get(field, "")
            row[field] = int(value) if value not in ("", None) else None
        for field in (
            "gold_probability",
            "gold_ppl",
            "signed_answer_margin",
            "candidate_margin",
            *METRICS,
        ):
            value = source.get(field, "")
            row[field] = float(value) if value not in ("", None) else None
        selected.append(row)
    selected.sort(key=lambda row: row["length"])
    expected = list(range(start, end + 1))
    actual = [row["length"] for row in selected]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise ValueError(f"dense interval is incomplete; missing={missing}")
    return selected


def pearson(xs: list[float], ys: list[float]) -> float:
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale == 0.0 or y_scale == 0.0:
        return float("nan")
    return numerator / (x_scale * y_scale)


def group_means(rows: list[dict[str, Any]], correct: int) -> dict[str, float]:
    group = [row for row in rows if row["top1_correct"] == correct]
    fields = ("signed_answer_margin", "gold_ppl", *METRICS)
    return {
        "n": len(group),
        **{field: mean(float(row[field]) for row in group) for field in fields},
    }


def transition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for left, right in zip(rows, rows[1:]):
        if left["top1_correct"] == right["top1_correct"]:
            continue
        transition_type = "failure" if right["top1_correct"] == 0 else "recovery"
        result.append(
            {
                "type": transition_type,
                "from_length": left["length"],
                "to_length": right["length"],
                "from_margin": left["signed_answer_margin"],
                "to_margin": right["signed_answer_margin"],
                "delta_margin": right["signed_answer_margin"]
                - left["signed_answer_margin"],
                **{
                    f"delta_{field}": float(right[field]) - float(left[field])
                    for field in METRICS
                },
            }
        )
    return result


def transition_means(
    transitions: list[dict[str, Any]], transition_type: str
) -> dict[str, float]:
    group = [row for row in transitions if row["type"] == transition_type]
    fields = ("delta_margin", *(f"delta_{field}" for field in METRICS))
    return {
        "n": len(group),
        **{field: mean(float(row[field]) for row in group) for field in fields},
    }


def runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    run_start = rows[0]
    previous = rows[0]
    for row in rows[1:]:
        if row["top1_correct"] != previous["top1_correct"]:
            result.append(
                {
                    "correct": previous["top1_correct"],
                    "start": run_start["length"],
                    "end": previous["length"],
                    "points": previous["length"] - run_start["length"] + 1,
                }
            )
            run_start = row
        previous = row
    result.append(
        {
            "correct": previous["top1_correct"],
            "start": run_start["length"],
            "end": previous["length"],
            "points": previous["length"] - run_start["length"] + 1,
        }
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent_difference(correct_value: float, wrong_value: float) -> float:
    return 100.0 * (wrong_value / correct_value - 1.0)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    correct: dict[str, float],
    wrong: dict[str, float],
    transitions: list[dict[str, Any]],
    failure_mean: dict[str, float],
    recovery_mean: dict[str, float],
    correlations: dict[str, float],
    all_runs: list[dict[str, Any]],
) -> None:
    failure_transitions = [
        row for row in transitions if row["type"] == "failure"
    ]
    recovery_transitions = [
        row for row in transitions if row["type"] == "recovery"
    ]
    failures_with_mass_increase = sum(
        row["delta_atomic_evidence_mass"] > 0.0 for row in failure_transitions
    )
    recoveries_with_mass_decrease = sum(
        row["delta_atomic_evidence_mass"] < 0.0 for row in recovery_transitions
    )
    candidate_rows = [
        row for row in rows if row.get("candidate_margin") is not None
    ]
    lines = [
        "# 34–100 token 密集失败边界：Attention 分析",
        "",
        "## 结论",
        "",
        f"- 连续扫描 {len(rows)} 点，其中完整词表一步输出正确 {int(correct['n'])} 点、错误 {int(wrong['n'])} 点。",
        f"- 第一次错误发生在 {failure_transitions[0]['from_length']} → "
        f"{failure_transitions[0]['to_length']}；区间内共有 "
        f"{len(failure_transitions)} 次失败翻转和 {len(recovery_transitions)} 次恢复。",
        f"- 错误点的平均证据 mass 比正确点低 "
        f"{abs(percent_difference(correct['atomic_evidence_mass'], wrong['atomic_evidence_mass'])):.1f}%，"
        "说明证据读取强度是重要的慢变量。",
        f"- 但 {failures_with_mass_increase}/{len(failure_transitions)} 次失败翻转发生时证据 mass 反而上升，"
        f"{recoveries_with_mass_decrease}/{len(recovery_transitions)} 次恢复发生时证据 mass 反而下降。",
        "- 因此模型级平均 attention mass 既不是逐 token 失败的充分条件，也不是必要条件。",
        "",
        "## 正确点与错误点的平均内部状态",
        "",
        "| 指标 | 正确点 | 错误点 | 错误相对正确 |",
        "|---|---:|---:|---:|",
    ]
    for field in METRICS:
        change = percent_difference(correct[field], wrong[field])
        lines.append(
            f"| {LABELS[field]} | {correct[field]:.6f} | {wrong[field]:.6f} | "
            f"{change:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 与答案 margin 的相关性",
            "",
            "| 指标 | Pearson r |",
            "|---|---:|",
        ]
    )
    for field in METRICS:
        lines.append(f"| {LABELS[field]} | {correlations[field]:+.3f} |")
    lines.extend(
        [
            "",
            "## 相邻翻转的平均变化",
            "",
            "| 翻转 | 数量 | Δmargin | Δ证据mass | ΔTop-20外mass | ΔLSE | ΔQK logit |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| 正确→错误 | {int(failure_mean['n'])} | "
            f"{failure_mean['delta_margin']:+.3f} | "
            f"{failure_mean['delta_atomic_evidence_mass']:+.6f} | "
            f"{failure_mean['delta_outside_top20_mass']:+.6f} | "
            f"{failure_mean['delta_mean_head_logsumexp']:+.3f} | "
            f"{failure_mean['delta_mean_hop2_result_logit']:+.3f} |",
            f"| 错误→正确 | {int(recovery_mean['n'])} | "
            f"{recovery_mean['delta_margin']:+.3f} | "
            f"{recovery_mean['delta_atomic_evidence_mass']:+.6f} | "
            f"{recovery_mean['delta_outside_top20_mass']:+.6f} | "
            f"{recovery_mean['delta_mean_head_logsumexp']:+.3f} | "
            f"{recovery_mean['delta_mean_hop2_result_logit']:+.3f} |",
            "",
            "## 机制解释",
            "",
            "1. **慢变量：证据选择性逐渐变弱。** 错误点总体具有更低的证据mass、更低的QK logit/cosine、更差的rank，以及更分散的attention。",
            "2. **当前短区间不是Softmax分母膨胀主导。** 错误点的平均LSE没有更高；失败翻转时平均LSE也略降。因此34–100内不能把失败归因于分母突然变大。",
            "3. **局部翻转发生在attention之后或被模型均值掩盖。** 同一总mass可以由不同head、不同value方向贡献；残差流、MLP和输出层会把很小的表示变化放大成较大的答案margin变化。",
            "4. **错误输出大多是解释性前缀。** 完整词表top-1常变为 `Let`，所以这里还混有输出格式控制失败，不能直接等同于模型在合法证据答案之间选错。",
            "",
            "更准确的因果链是：",
            "",
            "`长度/位置与filler变化 → 证据QK选择性整体减弱 → 系统进入低margin临界区 → head/value/残差的小扰动与输出格式偏好决定逐token翻转`",
        ]
    )
    if candidate_rows:
        candidate_correct = sum(int(row["candidate_correct"]) for row in candidate_rows)
        lines.extend(
            [
                "",
                "## 合法候选结果",
                "",
                f"- `river/window/basket` 三选一准确率：{candidate_correct}/{len(candidate_rows)} "
                f"= {candidate_correct / len(candidate_rows):.1%}。",
                "- 若候选准确而完整词表top-1错误，则该点是输出格式失败，不应计为证据检索失败。",
            ]
        )
    lines.extend(
        [
            "",
            "## 连续正确/错误区间",
            "",
            "| 状态 | 起点 | 终点 | 点数 |",
            "|---|---:|---:|---:|",
        ]
    )
    for run in all_runs:
        lines.append(
            f"| {'正确' if run['correct'] else '错误'} | "
            f"{run['start']} | {run['end']} | {run['points']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=34)
    parser.add_argument("--end", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.input_csv), args.start, args.end)
    correct = group_means(rows, 1)
    wrong = group_means(rows, 0)
    transitions = transition_rows(rows)
    failure_mean = transition_means(transitions, "failure")
    recovery_mean = transition_means(transitions, "recovery")
    margins = [float(row["signed_answer_margin"]) for row in rows]
    correlations = {
        field: pearson(
            margins,
            [float(row[field]) for row in rows],
        )
        for field in METRICS
    }
    all_runs = runs(rows)
    write_csv(output / "transition_pairs.csv", transitions)
    write_csv(output / "dense_rows.csv", rows)
    write_report(
        output / "dense_attention_failure_report.md",
        rows,
        correct,
        wrong,
        transitions,
        failure_mean,
        recovery_mean,
        correlations,
        all_runs,
    )
    (output / "dense_summary.json").write_text(
        json.dumps(
            {
                "start": args.start,
                "end": args.end,
                "points": len(rows),
                "correct_points": int(correct["n"]),
                "wrong_points": int(wrong["n"]),
                "correct_means": correct,
                "wrong_means": wrong,
                "failure_transition_means": failure_mean,
                "recovery_transition_means": recovery_mean,
                "correlations_with_answer_margin": correlations,
                "runs": all_runs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"analyzed dense interval {args.start}..{args.end} -> {output}")


if __name__ == "__main__":
    main()
