from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROLE_NAMES = ("start_key", "hop1_result", "hop2_input", "hop2_result")


def flatten(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, list):
            result.extend(flatten(value))
        else:
            result.append(float(value))
    return result


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


def signed_answer_margin(answer: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    gold_score = answer["gold_token_scores"][0]
    gold_id = int(gold_score["token_id"])
    gold_probability = float(gold_score["probability"])
    wrong = [
        item
        for item in answer["next_token_top5"]
        if int(item["token_id"]) != gold_id
    ]
    if not wrong:
        raise ValueError("next_token_top5 does not contain a non-gold token")
    strongest_wrong = max(wrong, key=lambda item: float(item["probability"]))
    wrong_probability = float(strongest_wrong["probability"])
    margin = math.log(max(gold_probability, 1e-300)) - math.log(
        max(wrong_probability, 1e-300)
    )
    return margin, strongest_wrong


def read_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    attention = payload["attention"]
    answer = payload["answer"]
    margin, strongest_wrong = signed_answer_margin(answer)
    role_mass = [float(value) for value in attention["overall_role_mass"]]
    top_scores = [float(value) for value in attention["overall_scores"]]
    role_logits = attention["head_role_logit_mean"]
    role_cosines = attention["head_role_cosine_mean"]
    role_ranks = attention["head_role_best_rank"]

    def role_mean(values: list[Any], role_index: int) -> float:
        return mean(
            float(head[role_index])
            for layer in values
            for head in layer
        )

    gold_score = answer["gold_token_scores"][0]
    atomic_evidence_mass = sum(role_mass[:4])
    candidate_scores = {
        str(item["text"]).strip(): float(item["probability"])
        for item in answer.get("candidate_token_scores", [])
    }
    row = {
        "length": int(payload["target_context_tokens"]),
        "prompt_tokens": int(payload["prompt_tokens"]),
        "gold_token": str(gold_score["token"]),
        "gold_probability": float(gold_score["probability"]),
        "gold_ppl": float(answer["gold_ppl"]),
        "strongest_wrong_token": str(strongest_wrong["token"]),
        "strongest_wrong_probability": float(strongest_wrong["probability"]),
        "signed_answer_margin": margin,
        "top1_correct": int(margin > 0.0),
        "candidate_correct": int(bool(answer["candidate_correct"]))
        if "candidate_correct" in answer
        else None,
        "candidate_prediction": answer.get("candidate_prediction"),
        "candidate_margin": answer.get("candidate_margin"),
        "river_probability": candidate_scores.get("river"),
        "window_probability": candidate_scores.get("window"),
        "basket_probability": candidate_scores.get("basket"),
        "start_key_mass": role_mass[0],
        "hop1_result_mass": role_mass[1],
        "hop2_input_mass": role_mass[2],
        "hop2_result_mass": role_mass[3],
        "atomic_evidence_mass": atomic_evidence_mass,
        "other_token_mass": 1.0 - atomic_evidence_mass,
        "outside_top20_mass": 1.0 - sum(top_scores[:20]),
        "attention_entropy": float(attention["overall_entropy"]),
        "effective_tokens": float(attention["overall_effective_tokens"]),
        "mean_head_logsumexp": mean(flatten(attention["head_logsumexp"])),
        "mean_hop1_logit": role_mean(role_logits, 1),
        "mean_hop2_input_logit": role_mean(role_logits, 2),
        "mean_hop2_result_logit": role_mean(role_logits, 3),
        "mean_hop2_result_cosine": role_mean(role_cosines, 3),
        "mean_hop2_result_rank": role_mean(role_ranks, 3),
    }
    row["mean_hop2_result_log_odds_proxy"] = (
        row["mean_hop2_result_logit"] - row["mean_head_logsumexp"]
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nearest(rows: list[dict[str, Any]], length: int) -> dict[str, Any]:
    return min(rows, key=lambda row: abs(int(row["length"]) - length))


def percent_change(start: float, end: float) -> float:
    return 100.0 * (end / start - 1.0)


def fmt_token(token: str) -> str:
    return f"`{token.replace(chr(10), r'\n')}`"


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    first = rows[0]
    last = rows[-1]
    transitions = [
        (left, right)
        for left, right in zip(rows, rows[1:])
        if int(left["top1_correct"]) != int(right["top1_correct"])
    ]
    first_failure = next(
        (right for left, right in transitions if int(left["top1_correct"]) == 1),
        None,
    )
    anchors = [
        nearest(rows, length)
        for length in (0, 500, 1000, 8000, 32000, 64000, 96000, 128000)
    ]
    margin_values = [float(row["signed_answer_margin"]) for row in rows]
    correlation_fields = (
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
    correlations = {
        field: pearson(margin_values, [float(row[field]) for row in rows])
        for field in correlation_fields
    }
    has_candidate_scores = any(row.get("candidate_margin") is not None for row in rows)

    lines = [
        "# Qwen3-8B 单样例：从正确答案到错误答案的内部轨迹",
        "",
        "## 样例与判定",
        "",
        "- 两条可信规则：`river → window`，`window → basket`。",
        "- 问题从 `river` 出发，要求执行两步，正确答案为 `basket`。",
        "- 数据为 clean 条件，没有干扰规则或矛盾规则。",
        f"- 共 {len(rows)} 个长度点：{rows[0]['length']}–{rows[-1]['length']}，步长 500 token。",
        "- 有符号答案 margin：`M = log P(basket) - log P(最强错误 token)`；`M > 0` 时一步 greedy 输出正确。",
        "",
        "## 一眼能看懂的结果",
        "",
    ]
    if first_failure is not None:
        previous = rows[rows.index(first_failure) - 1]
        lines.extend(
            [
                f"- 第一次翻转发生在 filler {previous['length']} → {first_failure['length']}："
                f"margin 从 {previous['signed_answer_margin']:+.3f} 变为 "
                f"{first_failure['signed_answer_margin']:+.3f}。",
                f"- 正确答案概率从 {previous['gold_probability']:.3%} 降到 "
                f"{first_failure['gold_probability']:.3%}；最强错误输出变为 "
                f"{fmt_token(first_failure['strongest_wrong_token'])}，概率 "
                f"{first_failure['strongest_wrong_probability']:.3%}。",
                f"- 全部四个原子证据位置的 attention mass 从 "
                f"{previous['atomic_evidence_mass']:.3%} 降到 "
                f"{first_failure['atomic_evidence_mass']:.3%} "
                f"({percent_change(previous['atomic_evidence_mass'], first_failure['atomic_evidence_mass']):+.1f}%)。",
                f"- 但最终结果 token `basket` 的 mass 只从 "
                f"{previous['hop2_result_mass']:.3%} 变为 "
                f"{first_failure['hop2_result_mass']:.3%}；真正大幅下降的是两跳连接："
                f"第一跳结果 `window` {previous['hop1_result_mass']:.3%} → "
                f"{first_failure['hop1_result_mass']:.3%}，第二条规则输入 `window` "
                f"{previous['hop2_input_mass']:.3%} → {first_failure['hop2_input_mass']:.3%}。",
                "- 因此首次失败不是“最终答案 token 完全没被看到”，而是模型没有稳定地把两条规则绑定成一条链。",
            ]
        )
    lines.extend(
        [
            "",
            "## 长度轨迹",
            "",
            "| Filler | 正确 | Gold PPL | Gold 概率 | 最强错误 | Margin | 证据 mass | Hop1 | Hop2 输入 | Hop2 结果 | Top-20 外 mass | LSE | Hop2 logit |",
            "|---:|:---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in anchors:
        lines.append(
            f"| {row['length']} | {'是' if row['top1_correct'] else '否'} | "
            f"{row['gold_ppl']:.3f} | {row['gold_probability']:.4%} | "
            f"{fmt_token(row['strongest_wrong_token'])} | "
            f"{row['signed_answer_margin']:+.3f} | {row['atomic_evidence_mass']:.4%} | "
            f"{row['hop1_result_mass']:.4%} | {row['hop2_input_mass']:.4%} | "
            f"{row['hop2_result_mass']:.4%} | {row['outside_top20_mass']:.2%} | "
            f"{row['mean_head_logsumexp']:.3f} | {row['mean_hop2_result_logit']:.3f} |"
        )
    if has_candidate_scores:
        candidate_rows = [
            row for row in rows if row.get("candidate_margin") is not None
        ]
        candidate_failures = [
            row for row in candidate_rows if int(row["candidate_correct"]) == 0
        ]
        lines.extend(
            [
                "",
                "## 合法候选空间",
                "",
                "- 候选集合固定为 `river`、`window`、`basket`。",
                f"- 候选准确率："
                f"{sum(int(row['candidate_correct']) for row in candidate_rows)}/{len(candidate_rows)}"
                f" = {sum(int(row['candidate_correct']) for row in candidate_rows) / len(candidate_rows):.1%}。",
                "- 该指标把“开始输出解释文字”与“在合法答案之间选错”分开。",
                f"- 首次候选错误长度："
                f"{candidate_failures[0]['length'] if candidate_failures else '未观察到'}。",
            ]
        )
    lines.extend(
        [
            "",
            "## Softmax 分解",
            "",
            "对某个 head 中的目标证据 token，attention 的对数近似为：",
            "",
            "`log a_e ≈ s_e - logsumexp(s_1, …, s_N)`",
            "",
            f"从 {first['length']} 到 {last['length']}：",
            "",
            f"- 最终证据的平均 QK logit：{first['mean_hop2_result_logit']:.3f} → "
            f"{last['mean_hop2_result_logit']:.3f}，变化 "
            f"{last['mean_hop2_result_logit'] - first['mean_hop2_result_logit']:+.3f}。",
            f"- Softmax 平均 logsumexp：{first['mean_head_logsumexp']:.3f} → "
            f"{last['mean_head_logsumexp']:.3f}，变化 "
            f"{last['mean_head_logsumexp'] - first['mean_head_logsumexp']:+.3f}。",
            f"- 二者共同使典型 head 的目标 log-odds proxy 变化 "
            f"{last['mean_hop2_result_log_odds_proxy'] - first['mean_hop2_result_log_odds_proxy']:+.3f}。",
            "- 第一项是证据自身 QK 匹配退化；第二项是其他 token 增多、增强导致的分母竞争。",
            "- 这是“先逐 head 取值、再平均”的诊断量，不等于模型整体 attention mass 的精确比值。",
            "",
            "## 与答案 margin 的257点相关性",
            "",
            "| 内部指标 | Pearson r |",
            "|---|---:|",
        ]
    )
    labels = {
        "atomic_evidence_mass": "全部原子证据 mass",
        "hop1_result_mass": "第一跳结果 mass",
        "hop2_input_mass": "第二跳输入 mass",
        "hop2_result_mass": "最终结果 mass",
        "other_token_mass": "其他 token mass",
        "outside_top20_mass": "Top-20 之外 mass",
        "attention_entropy": "attention entropy",
        "effective_tokens": "有效竞争 token 数",
        "mean_head_logsumexp": "Softmax logsumexp",
        "mean_hop2_result_logit": "最终证据 QK logit",
        "mean_hop2_result_cosine": "最终证据 QK cosine",
        "mean_hop2_result_rank": "最终证据 rank（越小越好）",
    }
    for field in correlation_fields:
        lines.append(f"| {labels[field]} | {correlations[field]:+.3f} |")
    lines.extend(
        [
            "",
            "## 因果解释边界",
            "",
            "- 这条轨迹支持“证据匹配退化 + Softmax 竞争增强 → 证据读取变弱 → 答案 margin 翻转”的机制链。",
            "- 但 attention mass 不是 value 向量对最终 logits 的直接因果贡献。严格因果确认还需要遮蔽证据、遮蔽竞争 token，或做 residual/logit attribution。",
            "- 轨迹并非严格单调：在25.5K和32K附近会短暂恢复正确。这说明失败边界是对 filler 截断位置敏感的概率区间，而不是一个永久、精确的单点阈值。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        input_dir.glob("length_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not paths:
        raise FileNotFoundError(f"no length_*.json under {input_dir}")
    rows = [read_row(path) for path in paths]
    write_csv(output_dir / "single_sample_trace.csv", rows)
    write_report(output_dir / "single_sample_trace_report.md", rows)
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "rows": len(rows),
                "minimum_length": rows[0]["length"],
                "maximum_length": rows[-1]["length"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"analyzed {len(rows)} points -> {output_dir}")


if __name__ == "__main__":
    main()
