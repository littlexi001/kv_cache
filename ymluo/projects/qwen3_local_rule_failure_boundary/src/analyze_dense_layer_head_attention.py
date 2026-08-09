from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def pearson(xs: list[float], ys: list[float]) -> float:
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale == 0.0 or y_scale == 0.0:
        return float("nan")
    return numerator / (x_scale * y_scale)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_point(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    answer = payload["answer"]
    attention = payload["attention"]
    gold = answer["gold_answer"]
    gold_probability = float(answer["gold_token_scores"][0]["probability"])
    wrong_probability = max(
        float(row["probability"])
        for row in answer["next_token_top5"]
        if row["token"].strip() != gold
    )
    head_role_mass = attention["head_role_mass"]
    head_mass = [
        [sum(float(value) for value in head[:4]) for head in layer]
        for layer in head_role_mass
    ]
    layer_mass = [mean(layer) for layer in head_mass]
    return {
        "length": int(payload["target_context_tokens"]),
        "gold_probability": gold_probability,
        "gold_ppl": float(answer["gold_ppl"]),
        "full_vocab_margin": math.log(gold_probability) - math.log(wrong_probability),
        "candidate_margin": float(answer["candidate_margin"]),
        "candidate_correct": int(answer["candidate_correct"]),
        "top_token": answer["next_token_top5"][0]["token"].strip(),
        "head_mass": head_mass,
        "layer_mass": layer_mass,
        "global_mass": mean(layer_mass),
    }


def group_mean(points: list[dict[str, Any]], field: str, correct: bool) -> float:
    return mean(
        float(point[field])
        for point in points
        if (float(point["full_vocab_margin"]) > 0.0) == correct
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=34)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--late_start", type=int, default=30)
    parser.add_argument("--late_end", type=int, default=33)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = [
        read_point(raw_dir / f"length_{length}.json")
        for length in range(args.start, args.end + 1)
    ]
    layers = len(points[0]["layer_mass"])
    heads = len(points[0]["head_mass"][0])
    margins = [float(point["full_vocab_margin"]) for point in points]
    candidate_margins = [float(point["candidate_margin"]) for point in points]
    gold_probabilities = [float(point["gold_probability"]) for point in points]

    layer_rows = []
    for layer in range(layers):
        masses = [float(point["layer_mass"][layer]) for point in points]
        correct_mass = mean(
            mass for mass, margin in zip(masses, margins) if margin > 0.0
        )
        wrong_mass = mean(
            mass for mass, margin in zip(masses, margins) if margin <= 0.0
        )
        layer_rows.append(
            {
                "layer": layer,
                "corr_full_vocab_margin": pearson(masses, margins),
                "corr_candidate_margin": pearson(masses, candidate_margins),
                "corr_gold_probability": pearson(masses, gold_probabilities),
                "correct_mean_mass": correct_mass,
                "wrong_mean_mass": wrong_mass,
                "wrong_vs_correct_pct": 100.0 * (wrong_mass / correct_mass - 1.0),
            }
        )

    head_rows = []
    for layer in range(layers):
        for head in range(heads):
            masses = [
                float(point["head_mass"][layer][head]) for point in points
            ]
            correct_mass = mean(
                mass for mass, margin in zip(masses, margins) if margin > 0.0
            )
            wrong_mass = mean(
                mass for mass, margin in zip(masses, margins) if margin <= 0.0
            )
            head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "corr_full_vocab_margin": pearson(masses, margins),
                    "corr_candidate_margin": pearson(masses, candidate_margins),
                    "corr_gold_probability": pearson(masses, gold_probabilities),
                    "correct_mean_mass": correct_mass,
                    "wrong_mean_mass": wrong_mass,
                    "wrong_vs_correct_pct": 100.0
                    * (wrong_mass / correct_mass - 1.0)
                    if correct_mass
                    else None,
                    "overall_mean_mass": mean(masses),
                }
            )

    for point in points:
        point["late_mass"] = mean(
            point["layer_mass"][args.late_start : args.late_end + 1]
        )
    transition_rows = []
    for left, right in zip(points, points[1:]):
        left_correct = float(left["full_vocab_margin"]) > 0.0
        right_correct = float(right["full_vocab_margin"]) > 0.0
        if left_correct == right_correct:
            continue
        transition_rows.append(
            {
                "type": "failure" if not right_correct else "recovery",
                "from_length": left["length"],
                "to_length": right["length"],
                "delta_full_vocab_margin": right["full_vocab_margin"]
                - left["full_vocab_margin"],
                "delta_global_evidence_mass": right["global_mass"]
                - left["global_mass"],
                "delta_late_evidence_mass": right["late_mass"]
                - left["late_mass"],
                "delta_candidate_margin": right["candidate_margin"]
                - left["candidate_margin"],
                "from_top_token": left["top_token"],
                "to_top_token": right["top_token"],
            }
        )

    late_masses = [float(point["late_mass"]) for point in points]
    global_masses = [float(point["global_mass"]) for point in points]
    failure_rows = [row for row in transition_rows if row["type"] == "failure"]
    recovery_rows = [row for row in transition_rows if row["type"] == "recovery"]
    summary = {
        "points": len(points),
        "candidate_correct": sum(point["candidate_correct"] for point in points),
        "full_vocab_correct": sum(margin > 0.0 for margin in margins),
        "global_mass_corr_full_vocab_margin": pearson(global_masses, margins),
        "late_layers": [args.late_start, args.late_end],
        "late_mass_corr_full_vocab_margin": pearson(late_masses, margins),
        "late_mass_corr_candidate_margin": pearson(
            late_masses, candidate_margins
        ),
        "global_correct_mean_mass": group_mean(
            points, "global_mass", True
        ),
        "global_wrong_mean_mass": group_mean(points, "global_mass", False),
        "late_correct_mean_mass": group_mean(points, "late_mass", True),
        "late_wrong_mean_mass": group_mean(points, "late_mass", False),
        "failure_transitions": len(failure_rows),
        "failure_with_late_mass_decrease": sum(
            row["delta_late_evidence_mass"] < 0.0 for row in failure_rows
        ),
        "recovery_transitions": len(recovery_rows),
        "recovery_with_late_mass_increase": sum(
            row["delta_late_evidence_mass"] > 0.0 for row in recovery_rows
        ),
    }

    write_csv(output_dir / "layer_attention_margin_stats.csv", layer_rows)
    write_csv(output_dir / "head_attention_margin_stats.csv", head_rows)
    write_csv(output_dir / "layer_aware_transition_stats.csv", transition_rows)
    (output_dir / "layer_head_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    top_heads = sorted(
        head_rows,
        key=lambda row: float(row["corr_full_vocab_margin"]),
        reverse=True,
    )[:10]
    lines = [
        "# 34–100 token：逐层逐 Head 的证据 Attention 分析",
        "",
        "## 核心结果",
        "",
        f"- 合法答案三选一：{summary['candidate_correct']}/{summary['points']} 正确；"
        f"完整词表第一 token：{summary['full_vocab_correct']}/{summary['points']} 正确。",
        f"- 全模型平均证据 mass 与完整词表答案 margin 的相关性："
        f"{summary['global_mass_corr_full_vocab_margin']:+.3f}。",
        f"- 第 {args.late_start}–{args.late_end} 层证据 mass 与同一 margin 的相关性："
        f"{summary['late_mass_corr_full_vocab_margin']:+.3f}。",
        f"- 错误点相对正确点：全局证据 mass 下降 "
        f"{100.0 * (summary['global_wrong_mean_mass'] / summary['global_correct_mean_mass'] - 1.0):+.1f}%；"
        f"第 {args.late_start}–{args.late_end} 层下降 "
        f"{100.0 * (summary['late_wrong_mean_mass'] / summary['late_correct_mean_mass'] - 1.0):+.1f}%。",
        f"- {summary['failure_with_late_mass_decrease']}/{summary['failure_transitions']} 次失败翻转伴随"
        f"第 {args.late_start}–{args.late_end} 层证据 mass 下降；"
        f"{summary['recovery_with_late_mass_increase']}/{summary['recovery_transitions']} 次恢复伴随其上升。",
        "",
        "## 解释",
        "",
        "全模型平均值会把不同功能的 Head 混在一起。晚层中与答案输出直接相关的 Head 对失败边界更敏感，"
        "所以晚层证据 mass 比全局平均 mass 更能解释答案 token 与 `Let` 等前缀 token 的竞争。"
        "但合法候选始终选择正确答案，说明这里观察到的完整词表错误主要是输出格式/生成策略翻转，"
        "不是证据语义被错误候选取代。",
        "",
        "## 与答案 margin 正相关最高的 Head",
        "",
        "| Layer | Head | r | 正确点 mass | 错误点 mass | 变化 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_heads:
        lines.append(
            f"| {row['layer']} | {row['head']} | "
            f"{float(row['corr_full_vocab_margin']):+.3f} | "
            f"{float(row['correct_mean_mass']):.6f} | "
            f"{float(row['wrong_mean_mass']):.6f} | "
            f"{float(row['wrong_vs_correct_pct']):+.1f}% |"
        )
    (output_dir / "layer_head_attention_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"analyzed {len(points)} raw points -> {output_dir}")


if __name__ == "__main__":
    main()
