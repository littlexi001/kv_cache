from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


EPS = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full QK/attention traces for the 136K-correct and 144K-failed age cases."
    )
    parser.add_argument("--correct", required=True, help="136K detailed result JSON")
    parser.add_argument("--failed", required=True, help="144K detailed result JSON")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        body = json.load(handle)
    if body.get("attention") is None:
        raise ValueError(f"{path} has no detailed attention payload")
    return body


def avg(values: Iterable[float]) -> float:
    rows = list(values)
    return mean(rows) if rows else 0.0


def pct(value: float) -> float:
    return 100.0 * value


def rounded(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


def category_index(body: dict[str, Any], name: str) -> int:
    return body["attention"]["category_order"].index(name)


def layer_head_values(
    body: dict[str, Any],
    field: str,
    category: str | None = None,
) -> list[list[float]]:
    values = body["attention"][field]
    if category is None:
        return [[float(value) for value in layer] for layer in values]
    index = category_index(body, category)
    return [
        [float(head[index]) for head in layer]
        for layer in values
    ]


def category_count(body: dict[str, Any], category: str) -> int:
    return int(body["case"]["category_counts"][category])


def flatten(values: list[list[float]]) -> list[float]:
    return [value for layer in values for value in layer]


def elementwise(
    left: list[list[float]],
    right: list[list[float]],
    operation,
) -> list[list[float]]:
    return [
        [operation(a, b) for a, b in zip(left_layer, right_layer)]
        for left_layer, right_layer in zip(left, right)
    ]


def case_metrics(body: dict[str, Any]) -> dict[str, Any]:
    attention = body["attention"]
    categories = attention["category_order"]
    masses = {
        category: layer_head_values(body, "head_category_mass", category)
        for category in categories
    }
    mean_attention = {
        category: layer_head_values(body, "head_category_mean_attention", category)
        for category in categories
    }
    qk_mean = {
        category: layer_head_values(body, "head_category_mean_logit", category)
        for category in categories
    }
    qk_max = {
        category: layer_head_values(body, "head_category_max_logit", category)
        for category in categories
    }
    qk_lse = {
        category: layer_head_values(body, "head_category_logsumexp", category)
        for category in categories
    }
    gold = "gold_age"
    distractor = "distractor_ages"
    gold_mass = masses[gold]
    distractor_mass = masses[distractor]
    gold_mean_attention = mean_attention[gold]
    distractor_mean_attention = mean_attention[distractor]
    gold_score = qk_lse[gold]
    distractor_mean_score = qk_mean[distractor]
    distractor_lse = qk_lse[distractor]
    all_lse = layer_head_values(body, "head_logsumexp")
    distractor_count = category_count(body, distractor)
    log_d = math.log(distractor_count)

    log_mass_ratio = elementwise(
        gold_mass,
        distractor_mass,
        lambda g, d: math.log(max(g, EPS)) - math.log(max(d, EPS)),
    )
    per_token_selectivity = elementwise(
        gold_mean_attention,
        distractor_mean_attention,
        lambda g, d: g / max(d, EPS),
    )
    semantic_advantage = elementwise(
        gold_score,
        distractor_mean_score,
        lambda g, d: g - d,
    )
    distractor_tail = elementwise(
        distractor_lse,
        distractor_mean_score,
        lambda lse, mu: lse - mu - log_d,
    )
    strongest_distractor_gap = elementwise(
        gold_score,
        qk_max[distractor],
        lambda g, d: g - d,
    )

    category_summary = {}
    for category in categories:
        category_summary[category] = {
            "token_count": category_count(body, category),
            "mean_mass": rounded(avg(flatten(masses[category]))),
            "mean_per_token_attention": rounded(
                avg(flatten(mean_attention[category]))
            ),
            "mean_qk": rounded(avg(flatten(qk_mean[category]))),
            "mean_max_qk": rounded(avg(flatten(qk_max[category]))),
        }

    return {
        "total_tokens": int(body["case"]["total_tokens"]),
        "distractor_count": int(body["case"]["distractor_count"]),
        "answer": body["answer"],
        "categories": category_summary,
        "category_masses": masses,
        "gold_mass": gold_mass,
        "distractor_mass": distractor_mass,
        "gold_score": gold_score,
        "distractor_mean_score": distractor_mean_score,
        "distractor_lse": distractor_lse,
        "all_lse": all_lse,
        "log_mass_ratio": log_mass_ratio,
        "per_token_selectivity": per_token_selectivity,
        "semantic_advantage": semantic_advantage,
        "distractor_tail": distractor_tail,
        "strongest_distractor_gap": strongest_distractor_gap,
        "gold_rank": layer_head_values(
            body,
            "head_category_best_rank",
            gold,
        ),
    }


def compare(correct: dict[str, Any], failed: dict[str, Any]) -> dict[str, Any]:
    layer_count = len(correct["gold_mass"])
    head_count = len(correct["gold_mass"][0])
    if layer_count != len(failed["gold_mass"]) or head_count != len(failed["gold_mass"][0]):
        raise ValueError("trace shapes differ")

    layer_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for layer in range(layer_count):
        c_gold = correct["gold_mass"][layer]
        f_gold = failed["gold_mass"][layer]
        c_dist = correct["distractor_mass"][layer]
        f_dist = failed["distractor_mass"][layer]
        c_selectivity = correct["per_token_selectivity"][layer]
        f_selectivity = failed["per_token_selectivity"][layer]
        c_ratio = correct["log_mass_ratio"][layer]
        f_ratio = failed["log_mass_ratio"][layer]
        c_semantic = correct["semantic_advantage"][layer]
        f_semantic = failed["semantic_advantage"][layer]
        c_tail = correct["distractor_tail"][layer]
        f_tail = failed["distractor_tail"][layer]
        c_strongest_gap = correct["strongest_distractor_gap"][layer]
        f_strongest_gap = failed["strongest_distractor_gap"][layer]
        c_rank = correct["gold_rank"][layer]
        f_rank = failed["gold_rank"][layer]
        c_gold_score = correct["gold_score"][layer]
        f_gold_score = failed["gold_score"][layer]
        c_all_lse = correct["all_lse"][layer]
        f_all_lse = failed["all_lse"][layer]

        layer_rows.append(
            {
                "layer": layer,
                "gold_mass_correct": avg(c_gold),
                "gold_mass_failed": avg(f_gold),
                "gold_mass_delta": avg(f_gold) - avg(c_gold),
                "distractor_age_mass_correct": avg(c_dist),
                "distractor_age_mass_failed": avg(f_dist),
                "distractor_age_mass_delta": avg(f_dist) - avg(c_dist),
                "per_token_selectivity_correct": avg(c_selectivity),
                "per_token_selectivity_failed": avg(f_selectivity),
                "per_token_selectivity_log_delta": avg(
                    math.log(max(f, EPS)) - math.log(max(c, EPS))
                    for c, f in zip(c_selectivity, f_selectivity)
                ),
                "gold_vs_distractor_log_mass_ratio_correct": avg(c_ratio),
                "gold_vs_distractor_log_mass_ratio_failed": avg(f_ratio),
                "gold_vs_distractor_log_mass_ratio_delta": avg(
                    f - c for c, f in zip(c_ratio, f_ratio)
                ),
                "semantic_qk_advantage_correct": avg(c_semantic),
                "semantic_qk_advantage_failed": avg(f_semantic),
                "semantic_qk_advantage_delta": avg(
                    f - c for c, f in zip(c_semantic, f_semantic)
                ),
                "distractor_tail_correction_correct": avg(c_tail),
                "distractor_tail_correction_failed": avg(f_tail),
                "distractor_tail_correction_delta": avg(
                    f - c for c, f in zip(c_tail, f_tail)
                ),
                "gold_beats_strongest_distractor_correct_fraction": avg(
                    float(value > 0) for value in c_strongest_gap
                ),
                "gold_beats_strongest_distractor_failed_fraction": avg(
                    float(value > 0) for value in f_strongest_gap
                ),
                "gold_top20_correct_fraction": avg(float(value <= 20) for value in c_rank),
                "gold_top20_failed_fraction": avg(float(value <= 20) for value in f_rank),
                "gold_rank_median_correct": median(c_rank),
                "gold_rank_median_failed": median(f_rank),
                "gold_qk_delta": avg(
                    f - c for c, f in zip(c_gold_score, f_gold_score)
                ),
                "all_logsumexp_delta": avg(
                    f - c for c, f in zip(c_all_lse, f_all_lse)
                ),
                "predicted_log_gold_mass_delta": avg(
                    (fs - cs) - (fl - cl)
                    for cs, fs, cl, fl in zip(
                        c_gold_score,
                        f_gold_score,
                        c_all_lse,
                        f_all_lse,
                    )
                ),
                "observed_log_gold_mass_delta": avg(
                    math.log(max(f, EPS)) - math.log(max(c, EPS))
                    for c, f in zip(c_gold, f_gold)
                ),
            }
        )

        for head in range(head_count):
            head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "gold_mass_correct": c_gold[head],
                    "gold_mass_failed": f_gold[head],
                    "gold_log_mass_delta": math.log(max(f_gold[head], EPS))
                    - math.log(max(c_gold[head], EPS)),
                    "distractor_mass_correct": c_dist[head],
                    "distractor_mass_failed": f_dist[head],
                    "gold_vs_distractor_log_mass_ratio_delta": f_ratio[head]
                    - c_ratio[head],
                    "semantic_qk_advantage_delta": f_semantic[head]
                    - c_semantic[head],
                    "distractor_tail_correction_delta": f_tail[head]
                    - c_tail[head],
                    "gold_qk_delta": f_gold_score[head] - c_gold_score[head],
                    "all_logsumexp_delta": f_all_lse[head] - c_all_lse[head],
                    "gold_rank_correct": c_rank[head],
                    "gold_rank_failed": f_rank[head],
                }
            )

    distractor_count_ratio = failed["distractor_count"] / correct["distractor_count"]
    all_heads = layer_count * head_count
    sorted_by_correct_gold = sorted(
        head_rows,
        key=lambda row: row["gold_mass_correct"],
        reverse=True,
    )

    def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        correct_gold_sum = sum(row["gold_mass_correct"] for row in rows)
        failed_gold_sum = sum(row["gold_mass_failed"] for row in rows)
        all_correct_gold_sum = sum(
            row["gold_mass_correct"] for row in head_rows
        )
        return {
            "head_count": len(rows),
            "correct_gold_mass_share": correct_gold_sum
            / max(all_correct_gold_sum, EPS),
            "gold_mass_sum_correct": correct_gold_sum,
            "gold_mass_sum_failed": failed_gold_sum,
            "gold_mass_retained_fraction": failed_gold_sum
            / max(correct_gold_sum, EPS),
            "mean_gold_qk_delta": avg(row["gold_qk_delta"] for row in rows),
            "mean_all_logsumexp_delta": avg(
                row["all_logsumexp_delta"] for row in rows
            ),
            "mean_gold_log_mass_delta": avg(
                row["gold_log_mass_delta"] for row in rows
            ),
            "mean_gold_vs_distractor_log_mass_ratio_delta": avg(
                row["gold_vs_distractor_log_mass_ratio_delta"]
                for row in rows
            ),
            "mean_semantic_qk_advantage_delta": avg(
                row["semantic_qk_advantage_delta"] for row in rows
            ),
            "distractor_count_log_pressure": math.log(distractor_count_ratio),
            "mean_distractor_tail_correction_delta": avg(
                row["distractor_tail_correction_delta"] for row in rows
            ),
        }

    important_heads = [
        row for row in head_rows if row["gold_mass_correct"] >= 0.01
    ]
    critical_subsets = {
        "top2": subset_summary(sorted_by_correct_gold[:2]),
        "top10": subset_summary(sorted_by_correct_gold[:10]),
        "top20": subset_summary(sorted_by_correct_gold[:20]),
        "gold_mass_at_least_1pct": subset_summary(important_heads),
    }

    stage_ranges = [
        ("early_routing", 0, 19),
        ("route_formation", 20, 22),
        ("critical_retrieval", 23, 24),
        ("evidence_integration", 25, 28),
        ("output_preparation", 29, 35),
    ]
    stage_rows: list[dict[str, Any]] = []
    for name, first_layer, last_layer in stage_ranges:
        row: dict[str, Any] = {
            "stage": name,
            "first_layer": first_layer,
            "last_layer": last_layer,
        }
        for category in correct["category_masses"]:
            correct_values = [
                correct["category_masses"][category][layer][head]
                for layer in range(first_layer, last_layer + 1)
                for head in range(head_count)
            ]
            failed_values = [
                failed["category_masses"][category][layer][head]
                for layer in range(first_layer, last_layer + 1)
                for head in range(head_count)
            ]
            row[f"{category}_correct"] = avg(correct_values)
            row[f"{category}_failed"] = avg(failed_values)
            row[f"{category}_delta"] = avg(failed_values) - avg(correct_values)
        row["gold_evidence_correct"] = (
            row["gold_other_correct"] + row["gold_age_correct"]
        )
        row["gold_evidence_failed"] = (
            row["gold_other_failed"] + row["gold_age_failed"]
        )
        row["gold_evidence_delta"] = (
            row["gold_evidence_failed"] - row["gold_evidence_correct"]
        )
        stage_rows.append(row)

    largest_absolute_losses = sorted(
        head_rows,
        key=lambda row: row["gold_mass_failed"] - row["gold_mass_correct"],
    )[:20]
    summary = {
        "correct": {
            "total_tokens": correct["total_tokens"],
            "distractor_count": correct["distractor_count"],
            "answer": correct["answer"],
            "categories": correct["categories"],
        },
        "failed": {
            "total_tokens": failed["total_tokens"],
            "distractor_count": failed["distractor_count"],
            "answer": failed["answer"],
            "categories": failed["categories"],
        },
        "shape": {"layers": layer_count, "heads": head_count, "layer_heads": all_heads},
        "global": {
            "mean_gold_mass_correct": avg(flatten(correct["gold_mass"])),
            "mean_gold_mass_failed": avg(flatten(failed["gold_mass"])),
            "mean_distractor_age_mass_correct": avg(flatten(correct["distractor_mass"])),
            "mean_distractor_age_mass_failed": avg(flatten(failed["distractor_mass"])),
            "mean_per_token_selectivity_correct": avg(
                flatten(correct["per_token_selectivity"])
            ),
            "mean_per_token_selectivity_failed": avg(
                flatten(failed["per_token_selectivity"])
            ),
            "mean_gold_vs_distractor_log_mass_ratio_delta": avg(
                row["gold_vs_distractor_log_mass_ratio_delta"] for row in head_rows
            ),
            "mean_semantic_qk_advantage_delta": avg(
                row["semantic_qk_advantage_delta"] for row in head_rows
            ),
            "distractor_count_log_pressure": math.log(distractor_count_ratio),
            "mean_distractor_tail_correction_delta": avg(
                row["distractor_tail_correction_delta"] for row in head_rows
            ),
            "mean_gold_qk_delta": avg(row["gold_qk_delta"] for row in head_rows),
            "mean_all_logsumexp_delta": avg(
                row["all_logsumexp_delta"] for row in head_rows
            ),
            "mean_gold_log_mass_delta": avg(
                row["gold_log_mass_delta"] for row in head_rows
            ),
            "heads_gold_mass_decreased_fraction": avg(
                float(row["gold_mass_failed"] < row["gold_mass_correct"])
                for row in head_rows
            ),
            "heads_gold_rank_worsened_fraction": avg(
                float(row["gold_rank_failed"] > row["gold_rank_correct"])
                for row in head_rows
            ),
        },
        "critical_subsets": critical_subsets,
        "stages": stage_rows,
        "largest_absolute_gold_mass_losses": largest_absolute_losses,
        "layers": layer_rows,
        "heads": head_rows,
    }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent_text(value: float) -> str:
    return f"{pct(value):.4f}%"


def report_text(result: dict[str, Any]) -> str:
    correct = result["correct"]
    failed = result["failed"]
    global_metrics = result["global"]
    critical = result["critical_subsets"]["gold_mass_at_least_1pct"]
    stages = result["stages"]
    critical_losses = result["largest_absolute_gold_mass_losses"][:10]
    layers = result["layers"]
    worst_gold_layers = sorted(layers, key=lambda row: row["gold_mass_delta"])[:8]
    worst_ratio_layers = sorted(
        layers,
        key=lambda row: row["gold_vs_distractor_log_mass_ratio_delta"],
    )[:8]
    semantic_layers = sorted(
        layers,
        key=lambda row: row["semantic_qk_advantage_delta"],
    )[:8]
    tail_layers = sorted(
        layers,
        key=lambda row: row["distractor_tail_correction_delta"],
        reverse=True,
    )[:8]

    lines = [
        "# 136K 正确样例 vs 144K 错误样例：attention/QK 分解",
        "",
        "## 输出翻转",
        "",
        "| 条件 | 干扰数 | P(nine) | Gold PPL | 全词表 margin | 年龄候选 margin | 年龄候选预测 |",
        "|---|---:|---:|---:|---:|---:|---|",
        (
            f"| 136K | {correct['distractor_count']} | "
            f"{percent_text(correct['answer']['gold_probability'])} | "
            f"{correct['answer']['gold_ppl']:.4f} | "
            f"{correct['answer']['full_vocab_margin']:+.4f} | "
            f"{correct['answer']['candidate_margin']:+.4f} | "
            f"{correct['answer']['candidate_prediction']} |"
        ),
        (
            f"| 144K | {failed['distractor_count']} | "
            f"{percent_text(failed['answer']['gold_probability'])} | "
            f"{failed['answer']['gold_ppl']:.4f} | "
            f"{failed['answer']['full_vocab_margin']:+.4f} | "
            f"{failed['answer']['candidate_margin']:+.4f} | "
            f"{failed['answer']['candidate_prediction']} |"
        ),
        "",
        "## 全模型平均 attention 分配",
        "",
        "| 类别 | 136K mass | 144K mass | 变化 |",
        "|---|---:|---:|---:|",
    ]
    category_labels = {
        "gold_other": "正确证据句（除 nine）",
        "gold_age": "正确年龄 token：nine",
        "distractor_other": "干扰句（除年龄）",
        "distractor_ages": "干扰年龄 token",
        "irrelevant_periods": "无关句号",
        "query": "问题与回答指令",
    }
    for category, label in category_labels.items():
        c = correct["categories"][category]["mean_mass"]
        f = failed["categories"][category]["mean_mass"]
        lines.append(
            f"| {label} | {percent_text(c)} | {percent_text(f)} | "
            f"{pct(f - c):+.4f} pp |"
        )

    lines.extend(
        [
            "",
            "## 分层过程：主导地位如何被夺走",
            "",
            "| 阶段 | 层 | 正确证据总 mass 136K→144K | nine mass 136K→144K | 干扰年龄 mass 136K→144K | filler 变化 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    stage_labels = {
        "early_routing": "早期路由",
        "route_formation": "检索路径形成",
        "critical_retrieval": "关键直接检索",
        "evidence_integration": "证据整合",
        "output_preparation": "输出准备",
    }
    for row in stages:
        lines.append(
            f"| {stage_labels[row['stage']]} | "
            f"{row['first_layer']}–{row['last_layer']} | "
            f"{percent_text(row['gold_evidence_correct'])} → "
            f"{percent_text(row['gold_evidence_failed'])} | "
            f"{percent_text(row['gold_age_correct'])} → "
            f"{percent_text(row['gold_age_failed'])} | "
            f"{percent_text(row['distractor_ages_correct'])} → "
            f"{percent_text(row['distractor_ages_failed'])} | "
            f"{pct(row['irrelevant_periods_delta']):+.4f} pp |"
        )
    lines.extend(
        [
            "",
            "## 关键检索 head 的精确分解",
            "",
            (
                "这里把 136K 时给 `nine` 至少 1% attention 的 head 定义为关键 head。"
                f"共 {critical['head_count']} 个，它们承载了全模型 "
                f"{percent_text(critical['correct_gold_mass_share'])} 的 `nine` mass。"
            ),
            "",
            r"\[\Delta\log m_{\text{nine}}=\Delta s_{\text{nine}}-\Delta\operatorname{LSE}(s_{\text{all}})\]",
            "",
            (
                f"- 关键 head 的 `nine` mass 仅保留："
                f"{percent_text(critical['gold_mass_retained_fraction'])}"
            ),
            (
                f"- `nine` 自身 QK 分数平均变化："
                f"{critical['mean_gold_qk_delta']:+.5f}"
            ),
            (
                f"- 全部 token 的 softmax log-sum-exp 平均变化："
                f"{critical['mean_all_logsumexp_delta']:+.5f}"
            ),
            (
                f"- 因此 `nine` mass 的平均 log 变化："
                f"{critical['mean_gold_log_mass_delta']:+.5f}"
            ),
            "",
            "相对所有干扰年龄：",
            "",
            r"\[\log\frac{m_{\text{nine}}}{m_{\text{dist-age}}}=(s_{\text{nine}}-\mu_d)-\log D-h_d\]",
            "",
            (
                f"- QK 语义优势变化："
                f"{critical['mean_semantic_qk_advantage_delta']:+.5f}"
            ),
            (
                f"- 干扰数量增长的 log 压力："
                f"{critical['distractor_count_log_pressure']:+.5f}"
            ),
            (
                f"- 高分干扰长尾修正变化："
                f"{critical['mean_distractor_tail_correction_delta']:+.5f}"
            ),
            (
                f"- 最终 `log(m_nine / m_dist-age)` 变化："
                f"{critical['mean_gold_vs_distractor_log_mass_ratio_delta']:+.5f}"
            ),
            "",
            "### 失守最严重的关键 head",
            "",
            "| 层-Head | nine mass 136K→144K | 干扰年龄 mass 136K→144K | Δnine QK | Δ总分母LSE | nine排名 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in critical_losses:
        lines.append(
            f"| L{row['layer']}H{row['head']} | "
            f"{percent_text(row['gold_mass_correct'])} → "
            f"{percent_text(row['gold_mass_failed'])} | "
            f"{percent_text(row['distractor_mass_correct'])} → "
            f"{percent_text(row['distractor_mass_failed'])} | "
            f"{row['gold_qk_delta']:+.4f} | "
            f"{row['all_logsumexp_delta']:+.4f} | "
            f"{int(row['gold_rank_correct'])} → {int(row['gold_rank_failed'])} |"
        )
    lines.extend(
        [
            "",
            "## 全部 head 等权平均（仅作完整性检查）",
            "",
            (
                "注意：多数 head 原本几乎不看 `nine`；其极小质量的相对变化会主导 log 平均，"
                "不能用来代表承担检索功能的关键 head。"
            ),
            "",
            "对每个 layer/head：",
            "",
            r"\[\Delta\log m_{\text{nine}}=\Delta s_{\text{nine}}-\Delta\operatorname{LSE}(s_{\text{all}})\]",
            "",
            (
                f"- `nine` 自身 QK 分数平均变化："
                f"{global_metrics['mean_gold_qk_delta']:+.5f}"
            ),
            (
                f"- 全部 token 的 softmax log-sum-exp 平均变化："
                f"{global_metrics['mean_all_logsumexp_delta']:+.5f}"
            ),
            (
                f"- `nine` attention mass 的平均 log 变化："
                f"{global_metrics['mean_gold_log_mass_delta']:+.5f}"
            ),
            "",
            "相对所有干扰年龄：",
            "",
            r"\[\log\frac{m_{\text{nine}}}{m_{\text{dist-age}}}=(s_{\text{nine}}-\mu_d)-\log D-h_d\]",
            "",
            (
                f"- QK 语义优势变化："
                f"{global_metrics['mean_semantic_qk_advantage_delta']:+.5f}"
            ),
            (
                f"- 干扰数量增长的 log 压力："
                f"{global_metrics['distractor_count_log_pressure']:+.5f}"
            ),
            (
                f"- 高分干扰长尾修正变化："
                f"{global_metrics['mean_distractor_tail_correction_delta']:+.5f}"
            ),
            (
                f"- 最终 `log(m_nine / m_dist-age)` 平均变化："
                f"{global_metrics['mean_gold_vs_distractor_log_mass_ratio_delta']:+.5f}"
            ),
            (
                f"- `nine` mass 下降的 head 比例："
                f"{percent_text(global_metrics['heads_gold_mass_decreased_fraction'])}"
            ),
            (
                f"- `nine` 全序列排名变差的 head 比例："
                f"{percent_text(global_metrics['heads_gold_rank_worsened_fraction'])}"
            ),
            "",
            "## 变化最大的层",
            "",
            "### nine attention mass 降幅最大",
            "",
            "| 层 | 136K | 144K | 变化 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in worst_gold_layers:
        lines.append(
            f"| {row['layer']} | {percent_text(row['gold_mass_correct'])} | "
            f"{percent_text(row['gold_mass_failed'])} | "
            f"{pct(row['gold_mass_delta']):+.4f} pp |"
        )
    lines.extend(
        [
            "",
            "### nine 相对干扰年龄的质量比下降最大",
            "",
            "| 层 | Δ log mass ratio | Δ QK语义优势 | Δ干扰长尾 |",
            "|---:|---:|---:|---:|",
        ]
    )
    by_layer = {row["layer"]: row for row in layers}
    for row in worst_ratio_layers:
        lines.append(
            f"| {row['layer']} | "
            f"{row['gold_vs_distractor_log_mass_ratio_delta']:+.5f} | "
            f"{row['semantic_qk_advantage_delta']:+.5f} | "
            f"{row['distractor_tail_correction_delta']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "### QK 语义优势退化最大",
            "",
            "| 层 | Δ QK语义优势 | Δ gold QK | Δ全分母 LSE |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in semantic_layers:
        lines.append(
            f"| {row['layer']} | {row['semantic_qk_advantage_delta']:+.5f} | "
            f"{row['gold_qk_delta']:+.5f} | {row['all_logsumexp_delta']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "### 高分干扰长尾增强最大",
            "",
            "| 层 | Δ长尾修正 | 136K gold胜过最强干扰的head | 144K |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in tail_layers:
        lines.append(
            f"| {row['layer']} | {row['distractor_tail_correction_delta']:+.5f} | "
            f"{percent_text(row['gold_beats_strongest_distractor_correct_fraction'])} | "
            f"{percent_text(row['gold_beats_strongest_distractor_failed_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "完整逐层、逐 head 数值见 `layers.csv`、`heads.csv` 和 `summary.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    correct = case_metrics(load(args.correct))
    failed = case_metrics(load(args.failed))
    result = compare(correct, failed)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    write_csv(output_dir / "layers.csv", result["layers"])
    write_csv(output_dir / "heads.csv", result["heads"])
    (output_dir / "report.md").write_text(
        report_text(result),
        encoding="utf-8",
    )
    print(json.dumps(result["global"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
