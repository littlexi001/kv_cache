from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


EVIDENCE_ROLE_INDICES = (0, 1)
TARGET_ROLE_INDEX = 1


def flatten(values: Iterable[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        if isinstance(value, list):
            output.extend(flatten(value))
        else:
            output.append(float(value))
    return output


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale == 0.0 or y_scale == 0.0:
        return float("nan")
    return numerator / (x_scale * y_scale)


def safe_group_mean(
    points: list[dict[str, Any]], field: str, *, correct: bool
) -> float | None:
    values = [
        float(point[field])
        for point in points
        if (float(point["full_vocab_margin"]) > 0.0) == correct
    ]
    return mean(values) if values else None


def strongest_wrong(answer: dict[str, Any]) -> dict[str, Any]:
    gold_id = int(answer["gold_token_scores"][0]["token_id"])
    rows = [
        row
        for row in answer["next_token_top5"]
        if int(row["token_id"]) != gold_id
    ]
    if not rows:
        raise ValueError("top-5 distribution contains no non-gold token")
    return max(rows, key=lambda row: float(row["probability"]))


def role_mean(values: list[Any], role_index: int) -> float:
    return mean(
        float(head[role_index])
        for layer in values
        for head in layer
    )


def read_point(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    answer = payload["answer"]
    attention = payload["attention"]
    gold_score = answer["gold_token_scores"][0]
    gold_probability = float(gold_score["probability"])
    wrong = strongest_wrong(answer)
    wrong_probability = float(wrong["probability"])
    full_margin = math.log(max(gold_probability, 1e-300)) - math.log(
        max(wrong_probability, 1e-300)
    )
    head_role_mass = attention["head_role_mass"]
    head_evidence_mass = [
        [
            sum(float(head[index]) for index in EVIDENCE_ROLE_INDICES)
            for head in layer
        ]
        for layer in head_role_mass
    ]
    head_target_mass = [
        [float(head[TARGET_ROLE_INDEX]) for head in layer]
        for layer in head_role_mass
    ]
    layer_evidence_mass = [mean(layer) for layer in head_evidence_mass]
    layer_target_mass = [mean(layer) for layer in head_target_mass]
    candidate_scores = {
        str(row["text"]).strip(): float(row["probability"])
        for row in answer.get("candidate_token_scores", [])
    }
    overall_roles = [float(value) for value in attention["overall_role_mass"]]
    point = {
        "length": int(payload["target_context_tokens"]),
        "prompt_tokens": int(payload["prompt_tokens"]),
        "gold_token": str(gold_score["token"]),
        "gold_probability": gold_probability,
        "gold_ppl": float(answer["gold_ppl"]),
        "strongest_wrong_token": str(wrong["token"]),
        "strongest_wrong_probability": wrong_probability,
        "full_vocab_margin": full_margin,
        "full_vocab_correct": int(full_margin > 0.0),
        "candidate_prediction": answer.get("candidate_prediction"),
        "candidate_margin": float(answer["candidate_margin"]),
        "candidate_correct": int(bool(answer["candidate_correct"])),
        "river_probability": candidate_scores.get("river"),
        "window_probability": candidate_scores.get("window"),
        "basket_probability": candidate_scores.get("basket"),
        "start_key_mass": overall_roles[0],
        "hop1_result_mass": overall_roles[1],
        "atomic_evidence_mass": overall_roles[0] + overall_roles[1],
        "other_token_mass": 1.0 - overall_roles[0] - overall_roles[1],
        "outside_top20_mass": 1.0
        - sum(float(value) for value in attention["overall_scores"][:20]),
        "attention_entropy": float(attention["overall_entropy"]),
        "effective_tokens": float(attention["overall_effective_tokens"]),
        "mean_head_logsumexp": mean(flatten(attention["head_logsumexp"])),
        "mean_target_logit": role_mean(
            attention["head_role_logit_mean"], TARGET_ROLE_INDEX
        ),
        "mean_target_cosine": role_mean(
            attention["head_role_cosine_mean"], TARGET_ROLE_INDEX
        ),
        "mean_target_rank": role_mean(
            attention["head_role_best_rank"], TARGET_ROLE_INDEX
        ),
        "mean_target_key_norm": role_mean(
            attention["head_role_key_norm_mean"], TARGET_ROLE_INDEX
        ),
        "mean_query_norm": mean(flatten(attention["head_query_norm"])),
        "head_evidence_mass": head_evidence_mass,
        "head_target_mass": head_target_mass,
        "layer_evidence_mass": layer_evidence_mass,
        "layer_target_mass": layer_target_mass,
    }
    return point


def scalar_row(point: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "head_evidence_mass",
        "head_target_mass",
        "layer_evidence_mass",
        "layer_target_mass",
    }
    return {key: value for key, value in point.items() if key not in excluded}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent_change(start: float | None, end: float | None) -> float | None:
    if start in {None, 0.0} or end is None:
        return None
    return 100.0 * (end / start - 1.0)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a strict single-rule, one-hop dense failure-boundary sweep."
    )
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--late_start", type=int, default=30)
    parser.add_argument("--late_end", type=int, default=33)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        raw_dir.glob("length_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if args.start is not None:
        paths = [
            path
            for path in paths
            if int(path.stem.split("_")[-1]) >= args.start
        ]
    if args.end is not None:
        paths = [
            path
            for path in paths
            if int(path.stem.split("_")[-1]) <= args.end
        ]
    if not paths:
        raise ValueError(f"no length JSON files found in {raw_dir}")
    points = [read_point(path) for path in paths]
    layers = len(points[0]["layer_evidence_mass"])
    heads = len(points[0]["head_evidence_mass"][0])
    if args.late_start < 0 or args.late_end >= layers:
        raise ValueError("late-layer range is outside the model")

    for point in points:
        point["late_evidence_mass"] = mean(
            point["layer_evidence_mass"][args.late_start : args.late_end + 1]
        )
        point["late_target_mass"] = mean(
            point["layer_target_mass"][args.late_start : args.late_end + 1]
        )

    margins = [float(point["full_vocab_margin"]) for point in points]
    candidate_margins = [float(point["candidate_margin"]) for point in points]
    gold_probabilities = [float(point["gold_probability"]) for point in points]
    global_evidence = [float(point["atomic_evidence_mass"]) for point in points]
    global_target = [float(point["hop1_result_mass"]) for point in points]
    late_evidence = [float(point["late_evidence_mass"]) for point in points]
    late_target = [float(point["late_target_mass"]) for point in points]

    layer_rows: list[dict[str, Any]] = []
    for layer in range(layers):
        evidence_values = [
            float(point["layer_evidence_mass"][layer]) for point in points
        ]
        target_values = [
            float(point["layer_target_mass"][layer]) for point in points
        ]
        layer_rows.append(
            {
                "layer": layer,
                "evidence_corr_full_margin": pearson(evidence_values, margins),
                "target_corr_full_margin": pearson(target_values, margins),
                "evidence_corr_candidate_margin": pearson(
                    evidence_values, candidate_margins
                ),
                "target_corr_candidate_margin": pearson(
                    target_values, candidate_margins
                ),
                "evidence_corr_gold_probability": pearson(
                    evidence_values, gold_probabilities
                ),
                "target_corr_gold_probability": pearson(
                    target_values, gold_probabilities
                ),
                "correct_mean_evidence_mass": mean(
                    value
                    for value, margin in zip(evidence_values, margins)
                    if margin > 0.0
                )
                if any(margin > 0.0 for margin in margins)
                else None,
                "wrong_mean_evidence_mass": mean(
                    value
                    for value, margin in zip(evidence_values, margins)
                    if margin <= 0.0
                )
                if any(margin <= 0.0 for margin in margins)
                else None,
            }
        )

    head_rows: list[dict[str, Any]] = []
    for layer in range(layers):
        for head in range(heads):
            evidence_values = [
                float(point["head_evidence_mass"][layer][head])
                for point in points
            ]
            target_values = [
                float(point["head_target_mass"][layer][head])
                for point in points
            ]
            head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "evidence_corr_full_margin": pearson(
                        evidence_values, margins
                    ),
                    "target_corr_full_margin": pearson(target_values, margins),
                    "evidence_corr_candidate_margin": pearson(
                        evidence_values, candidate_margins
                    ),
                    "target_corr_candidate_margin": pearson(
                        target_values, candidate_margins
                    ),
                    "mean_evidence_mass": mean(evidence_values),
                    "mean_target_mass": mean(target_values),
                }
            )

    transitions: list[dict[str, Any]] = []
    for left, right in zip(points, points[1:]):
        left_correct = bool(left["full_vocab_correct"])
        right_correct = bool(right["full_vocab_correct"])
        if left_correct == right_correct:
            continue
        transitions.append(
            {
                "type": "failure" if not right_correct else "recovery",
                "from_length": left["length"],
                "to_length": right["length"],
                "from_gold_probability": left["gold_probability"],
                "to_gold_probability": right["gold_probability"],
                "delta_full_vocab_margin": right["full_vocab_margin"]
                - left["full_vocab_margin"],
                "delta_candidate_margin": right["candidate_margin"]
                - left["candidate_margin"],
                "delta_global_evidence_mass": right["atomic_evidence_mass"]
                - left["atomic_evidence_mass"],
                "delta_global_target_mass": right["hop1_result_mass"]
                - left["hop1_result_mass"],
                "delta_late_evidence_mass": right["late_evidence_mass"]
                - left["late_evidence_mass"],
                "delta_late_target_mass": right["late_target_mass"]
                - left["late_target_mass"],
                "from_wrong_token": left["strongest_wrong_token"],
                "to_wrong_token": right["strongest_wrong_token"],
            }
        )

    correct_global = safe_group_mean(
        points, "atomic_evidence_mass", correct=True
    )
    wrong_global = safe_group_mean(points, "atomic_evidence_mass", correct=False)
    correct_late = safe_group_mean(points, "late_evidence_mass", correct=True)
    wrong_late = safe_group_mean(points, "late_evidence_mass", correct=False)
    failures = [row for row in transitions if row["type"] == "failure"]
    recoveries = [row for row in transitions if row["type"] == "recovery"]
    first_failure = failures[0] if failures else None
    summary = {
        "points": len(points),
        "length_min": points[0]["length"],
        "length_max": points[-1]["length"],
        "gold_answer": points[0]["gold_token"].strip(),
        "full_vocab_correct": sum(point["full_vocab_correct"] for point in points),
        "candidate_correct": sum(point["candidate_correct"] for point in points),
        "first_failure": first_failure,
        "global_evidence_corr_full_margin": pearson(global_evidence, margins),
        "global_target_corr_full_margin": pearson(global_target, margins),
        "late_layers": [args.late_start, args.late_end],
        "late_evidence_corr_full_margin": pearson(late_evidence, margins),
        "late_target_corr_full_margin": pearson(late_target, margins),
        "late_evidence_corr_candidate_margin": pearson(
            late_evidence, candidate_margins
        ),
        "global_correct_mean_evidence_mass": correct_global,
        "global_wrong_mean_evidence_mass": wrong_global,
        "global_wrong_vs_correct_pct": percent_change(
            correct_global, wrong_global
        ),
        "late_correct_mean_evidence_mass": correct_late,
        "late_wrong_mean_evidence_mass": wrong_late,
        "late_wrong_vs_correct_pct": percent_change(correct_late, wrong_late),
        "failure_transitions": len(failures),
        "failure_with_late_mass_decrease": sum(
            row["delta_late_evidence_mass"] < 0.0 for row in failures
        ),
        "recovery_transitions": len(recoveries),
        "recovery_with_late_mass_increase": sum(
            row["delta_late_evidence_mass"] > 0.0 for row in recoveries
        ),
    }

    write_csv(
        output_dir / "single_hop_trace.csv",
        [scalar_row(point) for point in points],
    )
    write_csv(output_dir / "layer_attention_margin_stats.csv", layer_rows)
    write_csv(output_dir / "head_attention_margin_stats.csv", head_rows)
    write_csv(output_dir / "transition_stats.csv", transitions)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    top_heads = sorted(
        head_rows,
        key=lambda row: float(row["evidence_corr_full_margin"]),
        reverse=True,
    )[:12]
    lines = [
        "# 严格单规则单跳失败边界分析",
        "",
        "## 实验定义",
        "",
        "- 规则：`river → window`；上下文中不放置第二条正确规则。",
        "- 查询：从 `river` 开始，只执行一步；正确答案为 `window`。",
        "- evidence mass：`start_key + hop1_result` 两个原子 token 的 attention mass。",
        f"- 长度：{points[0]['length']}–{points[-1]['length']}，共 {len(points)} 个整数长度点。",
        "",
        "## 总体结果",
        "",
        f"- 完整词表首 token：{summary['full_vocab_correct']}/{summary['points']} 正确；"
        f"三候选判断：{summary['candidate_correct']}/{summary['points']} 正确。",
        f"- 全局 evidence mass 与完整词表 margin："
        f"$r={summary['global_evidence_corr_full_margin']:+.3f}$；"
        f"只看结果 token：$r={summary['global_target_corr_full_margin']:+.3f}$。",
        f"- L{args.late_start}–{args.late_end} evidence mass 与完整词表 margin："
        f"$r={summary['late_evidence_corr_full_margin']:+.3f}$；"
        f"只看结果 token：$r={summary['late_target_corr_full_margin']:+.3f}$。",
        f"- 错误点相对正确点：全局 evidence mass "
        f"{fmt(summary['global_wrong_vs_correct_pct'], 1)}%；"
        f"L{args.late_start}–{args.late_end} evidence mass "
        f"{fmt(summary['late_wrong_vs_correct_pct'], 1)}%。",
        f"- {summary['failure_with_late_mass_decrease']}/{summary['failure_transitions']} 次失败翻转伴随"
        f"L{args.late_start}–{args.late_end} evidence mass 下降；"
        f"{summary['recovery_with_late_mass_increase']}/{summary['recovery_transitions']} 次恢复伴随其上升。",
        "",
    ]
    if first_failure is not None:
        left = next(
            point
            for point in points
            if point["length"] == first_failure["from_length"]
        )
        right = next(
            point
            for point in points
            if point["length"] == first_failure["to_length"]
        )
        lines.extend(
            [
                "## 第一次失败边界",
                "",
                "| 指标 | 边界前 | 边界后 | 变化 |",
                "|---|---:|---:|---:|",
                f"| 长度 | {left['length']} | {right['length']} | +{right['length'] - left['length']} |",
                f"| Gold 概率 | {left['gold_probability']:.4%} | {right['gold_probability']:.4%} | "
                f"{percent_change(left['gold_probability'], right['gold_probability']):+.1f}% |",
                f"| Gold PPL | {left['gold_ppl']:.4f} | {right['gold_ppl']:.4f} | "
                f"{percent_change(left['gold_ppl'], right['gold_ppl']):+.1f}% |",
                f"| 完整词表 margin | {left['full_vocab_margin']:+.4f} | {right['full_vocab_margin']:+.4f} | "
                f"{right['full_vocab_margin'] - left['full_vocab_margin']:+.4f} |",
                f"| 候选 margin | {left['candidate_margin']:+.4f} | {right['candidate_margin']:+.4f} | "
                f"{right['candidate_margin'] - left['candidate_margin']:+.4f} |",
                f"| 全局 evidence mass | {left['atomic_evidence_mass']:.4%} | {right['atomic_evidence_mass']:.4%} | "
                f"{percent_change(left['atomic_evidence_mass'], right['atomic_evidence_mass']):+.1f}% |",
                f"| 全局 result mass | {left['hop1_result_mass']:.4%} | {right['hop1_result_mass']:.4%} | "
                f"{percent_change(left['hop1_result_mass'], right['hop1_result_mass']):+.1f}% |",
                f"| L{args.late_start}–{args.late_end} evidence mass | {left['late_evidence_mass']:.4%} | "
                f"{right['late_evidence_mass']:.4%} | "
                f"{percent_change(left['late_evidence_mass'], right['late_evidence_mass']):+.1f}% |",
                f"| target QK logit | {left['mean_target_logit']:.4f} | {right['mean_target_logit']:.4f} | "
                f"{right['mean_target_logit'] - left['mean_target_logit']:+.4f} |",
                f"| target QK cosine | {left['mean_target_cosine']:.4f} | {right['mean_target_cosine']:.4f} | "
                f"{right['mean_target_cosine'] - left['mean_target_cosine']:+.4f} |",
                f"| softmax logsumexp | {left['mean_head_logsumexp']:.4f} | {right['mean_head_logsumexp']:.4f} | "
                f"{right['mean_head_logsumexp'] - left['mean_head_logsumexp']:+.4f} |",
                "",
            ]
        )

    lines.extend(
        [
            "## 与完整词表 margin 正相关最高的 Head",
            "",
            "| Layer | Head | evidence-mass r | result-mass r | 平均 evidence mass |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_heads:
        lines.append(
            f"| {row['layer']} | {row['head']} | "
            f"{float(row['evidence_corr_full_margin']):+.3f} | "
            f"{float(row['target_corr_full_margin']):+.3f} | "
            f"{float(row['mean_evidence_mass']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "相关性说明哪些层和 head 的证据读取与输出 margin 同步变化，但不能单独证明 causality。"
            "严格因果验证仍需在边界后的同一输入上恢复选定 head 的 evidence attention，"
            "并保持该输入自己的 Value 与其余计算不变，观察 `window` 相对最强竞争 token 的 margin 是否恢复。",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"analyzed {len(points)} points -> {output_dir}")


if __name__ == "__main__":
    main()
